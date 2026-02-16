import streamlit as st

# *** THIS MUST BE THE VERY FIRST STREAMLIT COMMAND ***
st.set_page_config(layout="wide")
st.title("🎬 LTX i2v Video Studio - Image-to-Video Generation")

# NOW we can proceed with other imports and initialization
import requests
import json
import time
import os
import uuid
import websocket
import subprocess
import cv2
import copy
import sys
import shutil
import random
import logging
import numpy as np

"""
LTX i2v (Image-to-Video) Video Generation Studio
==================================================

This application generates videos using the LTX i2v model, which creates video content
from input images combined with text prompts. Unlike text-to-video models, i2v requires:

1. INPUT IMAGES: Each scene MUST have a reference image
   - Can be automatically generated from descriptions
   - Can be uploaded by the user
   - Can be extracted from previous scenes for continuity

2. TEXT PROMPTS: Describe what should happen in the video
   - Based on the script
   - Optionally can use LLM (Ollama or Gemini)

3. WORKFLOW: Uses ComfyUI with the LTX 2.0 i2v model

WORKFLOW:
1. Parse script into scenes (text description + visual prompt)
2. For each scene:
   a. Generate or upload a reference image
   b. Create a text prompt describing the motion/animation
   c. Pass both to ComfyUI for i2v video generation
   d. Extract last frame for continuity to next scene
3. Stitch all videos together

KEY DIFFERENCES FROM T2V:
- Image is PRIMARY input (not optional)
- Each scene NEEDS a starting image
- Better character consistency with image input
- Prompt describes motion/changes, not just content
"""

# --- Dependency Check (note: actual imports, no st.* calls here) ---
# st.* calls must come AFTER st.set_page_config()
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    # Import the Google Generative AI library (package: google-generativeai)
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    # This is an optional dependency, so we'll just warn the user if they try to use it.
    GEMINI_AVAILABLE = False



# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# --- Configuration ---
COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_WS_URL = f"ws://{COMFYUI_URL.split('//')[1]}/ws"
WORKFLOW_FILE = "workfllows/video_ltx2_i2v.json"  # LTX i2v (Image-to-Video) model for visual continuation
CONFIG_FILE = "config.json"

# --- IMPORTANT: SET THESE PATHS TO YOUR COMFYUI'S ACTUAL DIRECTORIES ---
COMFYUI_REAL_INPUT_DIR = "D:\\StabilityMatrix-win-x64\\Data\\Packages\\ComfyUI\\input"
COMFYUI_REAL_OUTPUT_DIR = "D:\\StabilityMatrix-win-x64\\Data\\Packages\\ComfyUI\\output"

# These are for the application's local organization
APP_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FINAL_VIDEO_FILE = "final_story.mp4"

# Create base directories if they don't exist
os.makedirs(APP_OUTPUT_DIR, exist_ok=True)
os.makedirs(COMFYUI_REAL_INPUT_DIR, exist_ok=True) # Ensure Comfy's input dir exists

def load_config():
    """Loads configuration from a JSON file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error loading {CONFIG_FILE}: {e}")
            return {}
    return {}

def save_config(config_data):
    """Saves configuration to a JSON file."""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
        logging.info(f"Configuration saved to {CONFIG_FILE}")
    except IOError as e:
        logging.error(f"Error saving {CONFIG_FILE}: {e}")
        st.error(f"Could not save configuration file: {e}")

def create_blank_image(directory, width=1280, height=720, filename="blank.png"):
    """Creates a simple black PNG image to be used as a neutral starting frame."""
    image_path = os.path.join(directory, filename)
    if not os.path.exists(image_path):
        try:
            # Create a black image using numpy and opencv
            black_image = np.zeros((height, width, 3), np.uint8)
            cv2.imwrite(image_path, black_image)
            logging.info(f"Created blank starting image at: {image_path}")
            return filename
        except Exception as e:
            logging.error(f"Failed to create blank image: {e}")
            return None
    # If it already exists, just return the filename
    return filename

# --- Default Node Titles ---
# These titles are used to pre-select the correct nodes in the sidebar.
# They should match the titles in your `video_ltx2_i2v.json` workflow file.
# NOTE: For i2v (Image-to-Video) model, the IMAGE INPUT is CRITICAL - it's not optional!
DEFAULT_TITLES = {
    "prompt": "Positive Prompt",
    "negative_prompt": "Negative Prompt",  # Negative prompt node
    "seed": "RandomNoise",
    "image": "Load Image",  # CRITICAL for i2v - this is the MAIN input image from which video is generated
    "save": "Save Video",
    "frame_count": "Length",
    "frame_rate": "LTXVConditioning",  # Frame rate is a parameter of the conditioning node
    "ipadapter": "None"  # i2v doesn't use IPAdapter for character consistency - uses the input image directly
}

# --- Pre-flight check: Create the blank image at startup ---
BLANK_IMAGE_FILENAME = create_blank_image(COMFYUI_REAL_INPUT_DIR)

def validate_and_sanitize_prompt(prompt_text, max_length=4000):
    """
    Validates and sanitizes a prompt string before sending to ComfyUI.
    
    - Checks for reasonable length
    - Validates JSON encoding compatibility
    - Handles special characters properly
    - Returns (is_valid, sanitized_text, warnings)
    """
    warnings = []
    
    # 1. Check length
    if len(prompt_text) > max_length:
        warnings.append(f"⚠️ Prompt exceeds recommended length ({len(prompt_text)}/{max_length} chars). This may cause generation issues.")
        # For now, we'll allow it but warn the user
    
    # 2. Verify JSON serializability
    try:
        test_json = json.dumps({"text": prompt_text})
        logging.debug(f"Prompt JSON validation passed. Serialized size: {len(test_json)} bytes")
    except (TypeError, UnicodeDecodeError) as e:
        return False, prompt_text, [f"❌ Prompt contains characters that cannot be serialized to JSON: {e}"]
    
    # 3. Check for problematic escape sequences
    if prompt_text.count("\\") > 10:
        warnings.append("⚠️ Prompt contains many backslashes. This might cause parsing issues.")
    
    # 4. Validate no unescaped control characters
    control_chars = [c for c in prompt_text if ord(c) < 32 and c not in '\n\t\r']
    if control_chars:
        warnings.append(f"⚠️ Prompt contains {len(control_chars)} control characters that may cause issues.")
        # Remove problematic control characters
        prompt_text = "".join(c if ord(c) >= 32 or c in '\n\t\r' else ' ' for c in prompt_text)
    
    return True, prompt_text, warnings

def get_character_description(char_data):
    """
    Extract character description from various possible field names.
    Handles flexible JSON structures from different AI models.
    """
    if not isinstance(char_data, dict):
        return "character"
    
    # Try multiple field names (LLMs use different conventions)
    for field_name in ["description", "desc", "voice_description", "characteristics", "appearance", "info"]:
        if field_name in char_data and char_data[field_name]:
            return str(char_data[field_name])
    
    # Fallback: concatenate any text-like fields
    text_fields = []
    for key, value in char_data.items():
        if isinstance(value, str) and key not in ["name", "type", "role"]:
            text_fields.append(value)
    
    if text_fields:
        return ", ".join(text_fields)
    
    return "character"

def check_comfyui_health(status_text):
    """Checks if the ComfyUI server is alive and responsive."""
    try:
        response = requests.get(f"{COMFYUI_URL}/queue", timeout=3)
        response.raise_for_status()
        logging.info("ComfyUI health check PASSED.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"ComfyUI health check FAILED. Error: {e}")
        status_text.error("🔴 **ComfyUI is unreachable.** Please ensure it is running and accessible.")
        return False


def _wait_for_file_to_be_written(file_path, status_text, timeout_sec=120):
    """
    Waits for a specific file to appear and be fully written to disk by checking
    for a stable file size.
    """
    start_time = time.time()
    logging.info(f"Waiting for specific file to be written: '{os.path.basename(file_path)}'")
    while time.time() - start_time < timeout_sec:
        try:
            if os.path.exists(file_path):
                # Check if file size is stable
                initial_size = os.path.getsize(file_path)
                time.sleep(0.5) # Wait a moment to see if it's still being written
                final_size = os.path.getsize(file_path)

                if initial_size > 0 and initial_size == final_size:
                    logging.info(f"File '{os.path.basename(file_path)}' is complete.")
                    return True

            elapsed = int(time.time() - start_time)
            status_text.info(f"⏳ Waiting for output file to be written... {elapsed}s")
            time.sleep(1)
        except FileNotFoundError:
            # This can happen in a race condition, just continue waiting
            time.sleep(1)
            continue
        except Exception as e:
            logging.error(f"An error occurred while waiting for file: {e}", exc_info=True)
            time.sleep(1)

    logging.error(f"Timeout: File '{os.path.basename(file_path)}' not found or not complete after {timeout_sec}s.")
    return None

def sanitize_filename(filename):
    """Sanitizes a string to make it a valid filename."""
    # Remove invalid characters
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    # Limit length (filesystems typically limit to 255 chars)
    filename = filename[:200]
    return filename

def generate_video_title_with_gemini(detailed_script, api_key, model_name, status_text):
    """
    Uses Gemini API to generate a concise, catchy title for the video from the story script.
    """
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        logging.error(f"Failed to configure Gemini API for title generation: {e}")
        return None
    
    models_to_try = [model_name, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-pro"]
    actual_model_name = model_name
    
    try:
        available_models = [m.name for m in genai.list_models()]
        
        def normalize_model_name(name):
            if name.startswith('models/'):
                return name[7:]
            return name
        
        for potential_model in models_to_try:
            normalized_potential = normalize_model_name(potential_model)
            for available in available_models:
                normalized_available = normalize_model_name(available)
                if normalized_potential.lower() in normalized_available.lower():
                    actual_model_name = available
                    break
            if actual_model_name != model_name:
                break
    except Exception as e:
        logging.warning(f"Could not list models for title generation: {e}")
    
    try:
        model = genai.GenerativeModel(actual_model_name)
        prompt = f"""Read this story/script and generate a SHORT, CATCHY YouTube-friendly title (5-8 words max). 
The title should be compelling, descriptive, and suitable as a video title.
Respond with ONLY the title text, nothing else.

Story:
{detailed_script[:1000]}"""
        
        response = model.generate_content(prompt)
        title = response.text.strip()
        logging.info(f"Generated title with Gemini: {title}")
        return title
    except Exception as e:
        logging.error(f"Failed to generate title with Gemini: {e}")
        return None

def generate_video_title_with_ollama(detailed_script, model, status_text):
    """
    Uses Ollama to generate a concise, catchy title for the video from the story script.
    """
    try:
        prompt = f"""Read this story/script and generate a SHORT, CATCHY YouTube-friendly title (5-8 words max). 
The title should be compelling, descriptive, and suitable as a video title.
Respond with ONLY the title text, nothing else.

Story:
{detailed_script[:1000]}"""
        
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            options={'temperature': 0.7}
        )
        
        title = response['message']['content'].strip()
        logging.info(f"Generated title with Ollama: {title}")
        return title
    except Exception as e:
        logging.error(f"Failed to generate title with Ollama: {e}")
        return None

def parse_detailed_script_with_gemini(detailed_script, api_key, model_name, status_text, num_scenes=1):
    """
    Uses the Google Gemini API to parse a detailed narrative script into a structured JSON format.
    """
    status_text.info("🤖 Contacting AI Script Parser (Gemini API)...")
    logging.info(f"Parsing detailed script with Gemini using model: {model_name}, target scenes: {num_scenes}")

    # Explicitly configure the genai library with the API key
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Failed to configure Gemini API: {e}")
        logging.error(f"genai.configure() failed: {e}", exc_info=True)
        return None
    
    # Try to list available models and find one that works
    actual_model_name = model_name
    models_to_try = [model_name, "gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash", "gemini-pro-latest", "gemini-pro"]  # Fallback chain
    
    try:
        available_models = [m.name for m in genai.list_models()]
        logging.info(f"Available models: {available_models}")
        
        # Filter to text generation models (exclude embedding and image-only models)
        text_models = [m for m in available_models 
                      if 'generateContent' in m 
                      or ('gemini' in m.lower() and 'embedding' not in m.lower() and 'image' not in m.lower())
                      or 'gemma' in m.lower()]
        logging.info(f"Text generation models available: {text_models}")
        
        # Normalize model names for matching (remove 'models/' prefix and add if needed)
        def normalize_model_name(name):
            """Remove models/ prefix if present"""
            if name.startswith('models/'):
                return name[7:]  # Remove 'models/' prefix
            return name
        
        # Find a working model
        found_model = None
        for potential_model in models_to_try:
            normalized_potential = normalize_model_name(potential_model)
            # Check if this model exists in available models
            for available in text_models:
                normalized_available = normalize_model_name(available)
                if normalized_potential.lower() in normalized_available.lower() or normalized_available.lower() in normalized_potential.lower():
                    found_model = available
                    break
            if found_model:
                break
        
        if found_model:
            actual_model_name = found_model
            if found_model != model_name:
                logging.warning(f"Requested model {model_name} not available. Using {actual_model_name} instead.")
                status_text.warning(f"⚠️ Using compatible model: '{actual_model_name}'")
        elif text_models:
            # Use the first available model
            actual_model_name = text_models[0]
            logging.warning(f"Using first available model: {actual_model_name}")
            status_text.info(f"✓ Using available model: '{actual_model_name}'")
    except Exception as e:
        logging.warning(f"Could not list available models: {e}. Proceeding with model name: {model_name}")
    
    system_prompt = (
        "You are an expert script parser. Your task is to take a detailed video script "
        "and extract its components into a structured JSON format. The script may describe a story, a tutorial, an advertisement, or another concept. "
        "Identify a 'global_visual_description' that sets the overall style (e.g., '3D animation', 'live-action cooking show', 'cinematic product ad'). "
        "If specific, recurring characters are defined, list them in a 'characters' array. If no characters are defined, this key can be omitted. "
        f"CRITICAL SCENE COUNT: You MUST split the script into EXACTLY {num_scenes} scenes in the 'scenes' array. Each scene represents a 12-second video segment. "
        f"If the script has fewer natural breaks, split dialogue and action evenly across {num_scenes} scenes. If the script is short, create variations/continuations to fill all {num_scenes} scenes. "
        "Break the script into a 'scenes' array. Each scene MUST be a JSON object with 'type' and other fields. "
        "For each scene, provide a 'visual_prompt' that summarizes the key visual elements for the AI to generate. "
        "CRITICAL - DIALOGUE PRESERVATION: If a scene contains dialogue, include 'character' and 'dialogue_text' fields WITHIN THE SAME SCENE OBJECT. "
        "DIALOGUE TEXT MUST BE COPIED EXACTLY AS WRITTEN IN THE SOURCE SCRIPT — WORD FOR WORD, CHARACTER FOR CHARACTER. "
        "DO NOT translate, paraphrase, rewrite, edit, rephrase, or 'improve' ANY dialogue. "
        "If the user wrote dialogue in Hinglish (Hindi words in English/Roman script like 'Arre tum itna lamba muh kaise bana lete ho'), "
        "you MUST copy that EXACT Hinglish text into dialogue_text. Do NOT convert it to English or formal Hindi. "
        "If the user wrote dialogue in any language using Roman script, preserve it exactly as-is. "
        "When dialogue has special characters, quotes, or non-English text, ensure proper JSON escaping. "
        "If a scene contains on-screen text or titles, use a 'title_card' type with a 'text' field. "
        ""
        "ABSOLUTE RULE - TYPE FIELD: ANY scene that contains 'dialogue_text' MUST have its 'type' set to 'dialogue'. "
        "NEVER use 'visual_segment' for a scene that has dialogue_text. This is the #1 most important rule. "
        "If a character speaks in a scene, that scene's type MUST be 'dialogue', period. "
        ""
        "CRITICAL - EVERY SCENE MUST HAVE DIALOGUE: The user requires that ALL scenes contain dialogue. "
        "Distribute the script's dialogue evenly across ALL scenes. Do NOT leave any scene without dialogue_text. "
        "If the script has limited dialogue, split existing dialogue into smaller parts so every scene has some speech. "
        "Every single scene object MUST have a non-empty 'dialogue_text' field and type='dialogue'. Zero silent scenes allowed. "
        "CRITICAL - DIALOGUE QUALITY & LANGUAGE PRESERVATION: "
        "Each scene's dialogue MUST be contextually relevant to the visual action in that scene. "
        "COPY the dialogue VERBATIM from the original script — do NOT translate, rephrase, or rewrite it. "
        "If the original script uses Hinglish (Hindi written in Roman/English script, e.g. 'Yaar, kya scene hai?'), "
        "then ALL dialogue_text MUST also be in Hinglish exactly as written. Do NOT convert Hinglish to English or to Devanagari Hindi. "
        "If the script is in Hindi, keep Hindi. If English, keep English. NEVER change the language of the dialogue. "
        "The dialogue must make narrative sense — split at natural pause points (sentence boundaries like '.', '!', '?', '|') "
        "so each chunk is a complete, meaningful thought. "
        ""
        "CRITICAL - DIALOGUE DISTRIBUTION: If the script is dialogue-heavy, do NOT create long stretches of silent 'visual_segment' scenes. "
        "Interleave dialogue throughout the scenes naturally. A 10-15 second visual-only segment in the middle of a conversation looks unnatural. "
        "If dialogue exists nearby in the script, split it so each scene has some dialogue rather than bunching all dialogue into a few scenes and leaving others silent. "
        "Short visual transitions (1-2 scenes) between dialogue are fine, but avoid 3+ consecutive silent visual segments when dialogue is available. "
        ""
        "CRITICAL - LANGUAGE & ACCENT CONSISTENCY: Detect the language style from the script. "
        "If the script contains Hinglish (Hindi words written in Roman script like 'kya', 'hai', 'yaar', 'bhai', 'arre', 'accha'), "
        "set language to 'Hinglish' and accent to 'native Hindi desi'. "
        "If the script is in pure Hindi (Devanagari), set language to 'Hindi' and accent to 'native Hindi desi'. "
        "If the script is in English, set language to 'English' and accent to 'neutral American'. "
        "Add top-level 'language' and 'accent' fields to the JSON output. "
        ""
        "CRITICAL - VISUAL PROMPT MUST NOT CONTAIN SILENCE MARKERS FOR DIALOGUE SCENES: "
        "If a scene has 'dialogue_text', the 'visual_prompt' field MUST NOT contain any phrases like "
        "'No speech', 'No dialogue', 'No voiceover', 'No talking', 'Silent scene', or 'No human voice'. "
        "These silence instructions destroy the dialogue during video generation. Only describe the visual action in 'visual_prompt'. "
        ""
        "CRITICAL - SCENE UNIQUENESS: Each scene MUST have a visually DISTINCT 'visual_prompt'. "
        "Do NOT repeat or closely paraphrase the same visual description across multiple scenes. "
        "Each scene must describe a DIFFERENT camera angle, character action, setting detail, or moment in the story. "
        "If two scenes involve the same characters talking, differentiate them by: camera angle (close-up vs wide), character positioning, expressions, gestures, background elements, or lighting changes. "
        ""
        "IMPORTANT: Every scene object must have this complete structure: {\"type\": \"...\", \"character\": \"...\", \"dialogue_text\": \"...\", \"visual_prompt\": \"...\"} "
        "The top-level JSON must include: {\"global_visual_description\": \"...\", \"language\": \"...\", \"accent\": \"...\", \"characters\": [...], \"scenes\": [...]} "
        "ENSURE OUTPUT IS VALID, PROPERLY FORMATTED JSON. Each array element is a complete object separated by commas. "
        "Your entire response must be ONLY the JSON object. Do not include any conversational text, introductions, or explanations. Your response must start with `{` and end with `}`."
    )

    # Conditionally remove the "EVERY SCENE MUST HAVE DIALOGUE" block if force_dialogue is off
    # (It's always included above; we strip it when not needed)
    if not st.session_state.get("force_dialogue_all_scenes", False):
        system_prompt = system_prompt.replace(
            "CRITICAL - EVERY SCENE MUST HAVE DIALOGUE: The user requires that ALL scenes contain dialogue. "
            "Distribute the script's dialogue evenly across ALL scenes. Do NOT leave any scene without dialogue_text. "
            "If the script has limited dialogue, split existing dialogue into smaller parts so every scene has some speech. "
            "Every single scene object MUST have a non-empty 'dialogue_text' field and type='dialogue'. Zero silent scenes allowed. ",
            ""
        )

    try:
        model = genai.GenerativeModel(
            model_name=actual_model_name,
            system_instruction=system_prompt,
            generation_config={"response_mime_type": "application/json"} # Enforce JSON output
        )
        response = model.generate_content(
            detailed_script,
            generation_config=genai.GenerationConfig(temperature=0.5)
        )

        content = response.text
        logging.info(f"Gemini response received (partial): {content[:500]}...")

        # Even with enforced JSON, it's good practice to have robust parsing.
        parsed_data = json.loads(content)

        # Basic validation of the parsed structure
        if "global_visual_description" not in parsed_data or "scenes" not in parsed_data:
            raise ValueError("Parsed JSON is missing required top-level keys (global_visual_description, scenes).")

        status_text.info(f"✅ AI Script parsed successfully! Found {len(parsed_data['scenes'])} events. Now enhancing for cinematic quality...")
        logging.info(f"Successfully parsed script into {len(parsed_data['scenes'])} events.")
    except AttributeError as e:
        st.error(f"❌ Gemini API Error: The 'google-generativeai' library is not properly installed or configured. Please ensure you have installed it with: pip install google-generativeai")
        logging.error(f"Gemini API AttributeError (likely import issue): {e}", exc_info=True)
        return None
    except (ValueError, json.JSONDecodeError) as e:
        st.error(f"Failed to parse Gemini response as JSON. This may indicate an API key issue or rate limiting. Error: {e}")
        logging.error(f"Gemini JSON parsing failed: {e}", exc_info=True)
        return None
    except Exception as e:
        # Check for authentication/API key errors
        error_str = str(e)
        error_upper = error_str.upper()
        
        # Handle specific error cases
        if "NOT_FOUND" in error_upper or "404" in error_str:
            st.error(f"❌ Model Not Found: The model '{actual_model_name}' is not supported for text generation.\n\n" + 
                    f"**This often happens with the deprecated 'google.generativeai' package.**\n\n" +
                    f"**Alternatives:**\n" +
                    f"1. Try using **Ollama (Local)** parser instead - it's free and doesn't need an API key\n" +
                    f"2. The app tried to auto-select a compatible model but couldn't find one\n" +
                    f"3. Re-run the app - it may automatically pick a working model\n" +
                    f"4. Upgrade to the newer 'google-genai' package:\n   ```pip install --upgrade google-genai```")
        elif any(keyword in error_upper for keyword in ["API_KEY", "PERMISSION_DENIED", "UNAUTHENTICATED"]):
            st.error(f"❌ Gemini API Authentication Error: {e}\n\n**Troubleshooting:**\n1. Get your API key from https://ai.google.dev/aistudio\n2. Make sure you've enabled the Gemini API in Google Cloud\n3. Check that your API key is correct (no extra spaces or characters)")
        elif "RESOURCE_EXHAUSTED" in error_upper or "429" in error_str:
            st.error(f"❌ Rate Limited: You've hit Google's API rate limit. Please wait a moment and try again.")
        else:
            st.error(f"Failed during Gemini API parsing: {e}\n\n**Tip:** Try using the **Ollama (Local)** parser instead for better compatibility.")
        logging.error(f"Gemini initial parsing failed: {e}", exc_info=True)
        return None

    enhancement_prompt = (
        "You are a world-class cinematographer and creative director. Your task is to take a structured video plan (JSON) and make it more cinematic. "
        "You will be given a 'global_visual_description' and a list of 'scenes'. "
        "For each scene, you must REWRITE the 'visual_prompt' to be more vivid, detailed, and evocative. "
        "Incorporate professional cinematography terms. Add details about camera angles (e.g., 'low-angle shot', 'dutch angle'), camera movement (e.g., 'slow dolly in', 'crane shot down'), lighting (e.g., 'dramatic Rembrandt lighting', 'soft golden hour glow'), and character emotion (e.g., 'a look of quiet determination', 'a joyful, uninhibited smile'). "
        "CRITICAL RULES - DO NOT MODIFY THESE FIELDS: 'type', 'character', 'dialogue_text', 'language', 'accent' must be PRESERVED EXACTLY as written. Copy them verbatim without any changes, rewording, formatting, or additions. These are SACRED fields and must appear in output unchanged. "
        "CRITICAL - DO NOT ADD SILENCE MARKERS TO DIALOGUE SCENES: If a scene has 'dialogue_text' that is non-empty, the 'visual_prompt' MUST NOT contain any phrases like 'No speech', 'No dialogue', 'No voiceover', 'No talking', 'Silent scene', or 'No human voice'. These silence markers DESTROY the dialogue during video generation. Only add silence markers to scenes that have NO dialogue_text. "
        "CRITICAL - SCENE UNIQUENESS: Each scene's enhanced 'visual_prompt' MUST be visually distinct from every other scene. Use DIFFERENT camera angles, movements, lighting setups, and compositions for each scene. Never repeat the same cinematography for two scenes. "
        "Only rewrite the 'visual_prompt' for each scene. Keep all other fields unchanged. "
        "Return the complete, updated JSON object. Your entire response must be ONLY the JSON object, starting with `{` and ending with `}`."
    )

    # --- Cinematic Enhancement Pass ---
    logging.info("Starting cinematic enhancement pass with Gemini.")

    try:
        # Use a new model instance with the cinematographer system prompt
        enhancer_model = genai.GenerativeModel(
            model_name=actual_model_name,
            system_instruction=enhancement_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        enhancement_response = enhancer_model.generate_content(json.dumps(parsed_data))
        enhanced_content = enhancement_response.text
        logging.info(f"Gemini enhancement response received (partial): {enhanced_content[:500]}...")
        enhanced_data = json.loads(enhanced_content)

        if "scenes" not in enhanced_data or len(enhanced_data["scenes"]) != len(parsed_data["scenes"]):
            raise ValueError("Enhanced JSON is malformed or has a different number of scenes.")

        # --- POST-ENHANCEMENT: Strip silence markers from dialogue scenes ---
        # The enhancement LLM sometimes adds "No speech. No dialogue..." to visual_prompts
        # even for scenes that have dialogue_text. This override kills the dialogue during generation.
        import re as _re_enhance
        silence_strip_patterns = [
            r'No speech\.?\s*', r'No dialogue\.?\s*', r'No voiceover\.?\s*',
            r'No talking\.?\s*', r'No human voice\.?\s*',
            r'Silent scene[^.]*\.?\s*',
            r'only soft ambient environmental sounds\.?\s*',
        ]
        stripped_count = 0
        for scene in enhanced_data.get("scenes", []):
            if scene.get("dialogue_text") and scene.get("dialogue_text", "").strip():
                vp = scene.get("visual_prompt", "")
                original_vp = vp
                for pattern in silence_strip_patterns:
                    vp = _re_enhance.sub('', vp, flags=_re_enhance.IGNORECASE)
                vp = _re_enhance.sub(r'\s{2,}', ' ', vp).strip()
                if vp != original_vp:
                    scene["visual_prompt"] = vp
                    stripped_count += 1
                    logging.info(f"Enhancement post-processing: Stripped silence markers from dialogue scene visual_prompt")
        if stripped_count > 0:
            logging.info(f"Enhancement post-processing: Cleaned silence markers from {stripped_count} dialogue scene(s)")

        status_text.success(f"✅ AI Script parsed and cinematically enhanced! Found {len(enhanced_data['scenes'])} events.")
        logging.info(f"Successfully enhanced script with {len(enhanced_data['scenes'])} events.")
        return enhanced_data
    except (json.JSONDecodeError, ValueError) as e:
        st.warning(f"Cinematic enhancement pass failed due to data validation: {e}. Falling back to the basic parsed script.")
        logging.warning(f"Enhancement phase data validation failed: {e}", exc_info=True)
        return parsed_data
    except Exception as e:
        st.warning(f"Cinematic enhancement pass failed: {e}. Falling back to the basic parsed script.")
        logging.warning(f"Cinematic enhancement failed, returning original parse. Error: {e}", exc_info=True)
        return parsed_data # Fallback to the original parsed data if enhancement fails


def safely_parse_json_with_control_chars(json_str):
    """
    Parse JSON string while properly handling control characters (newlines, tabs) that
    can appear in LLM responses. These characters break standard JSON parsing.
    
    Args:
        json_str: Raw JSON string potentially containing literal newlines in values
        
    Returns:
        Parsed JSON object
        
    Raises:
        ValueError: If JSON cannot be parsed
    """
    import re
    
    # Step 1: Replace literal newlines/tabs/carriage returns with spaces
    # This prevents "Invalid control character" errors from json.loads()
    sanitized = json_str.replace('\n', ' ')
    sanitized = sanitized.replace('\r', ' ')
    sanitized = sanitized.replace('\t', ' ')
    
    # Step 2: Handle smart/curly quotes that LLMs sometimes produce
    sanitized = sanitized.replace('"', '"').replace('"', '"')  # Left and right double quotes
    sanitized = sanitized.replace(''', "'").replace(''', "'")  # Left and right single quotes
    
    # Step 3: Remove excessive spaces (normalize multiple spaces to single)
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # Step 4: Try to parse the JSON
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError as e:
        # If still failing, provide diagnostic information
        error_pos = e.pos if hasattr(e, 'pos') else 0
        context_start = max(0, error_pos - 100)
        context_end = min(len(sanitized), error_pos + 100)
        error_context = sanitized[context_start:context_end]
        
        logging.error(f"JSON parsing failed at position {error_pos}")
        logging.error(f"Error message: {e.msg}")
        logging.error(f"Context around error: ...{error_context}...")
        
        raise ValueError(f"Failed to parse JSON response: {e}") from e


def parse_detailed_script_with_ollama(detailed_script, model, status_text, num_scenes=1):
    """
    Uses Ollama to parse a detailed narrative script into a structured JSON format
    for sequential video generation.
    """
    status_text.info("🤖 Contacting AI Script Parser (Ollama)...")
    logging.info(f"Parsing detailed script with Ollama using model: {model}, target scenes: {num_scenes}")

    # The system prompt instructs Ollama to output a specific JSON structure
    system_prompt = (
        "You are an expert script parser. Your task is to take a detailed video script "
        "and extract its components into a structured JSON format. The script may describe a story, a tutorial, an advertisement, or another concept. "
        "Identify a 'global_visual_description' that sets the overall style (e.g., '3D animation', 'live-action cooking show', 'cinematic product ad'). "
        "If specific, recurring characters are defined, list them in a 'characters' array. If no characters are defined, this key can be omitted. "
        f"CRITICAL SCENE COUNT: You MUST split the script into EXACTLY {num_scenes} scenes in the 'scenes' array. Each scene represents a 12-second video segment. "
        f"If the script has fewer natural breaks, split dialogue and action evenly across {num_scenes} scenes. If the script is short, create variations/continuations to fill all {num_scenes} scenes. "
        "Break the script into a 'scenes' array. Each scene MUST be a JSON object with 'type' and other fields. "
        "For each scene, provide a 'visual_prompt' that summarizes the key visual elements for the AI to generate. "
        "CRITICAL - DIALOGUE PRESERVATION: If a scene contains dialogue, ALWAYS include 'character' and 'dialogue_text' fields WITHIN THE SAME SCENE OBJECT. "
        "DIALOGUE TEXT MUST BE COPIED EXACTLY AS WRITTEN IN THE SOURCE SCRIPT — WORD FOR WORD, CHARACTER FOR CHARACTER. "
        "DO NOT translate, paraphrase, rewrite, edit, rephrase, or 'improve' ANY dialogue. "
        "If the user wrote dialogue in Hinglish (Hindi words in English/Roman script like 'Arre tum itna lamba muh kaise bana lete ho'), "
        "you MUST copy that EXACT Hinglish text into dialogue_text. Do NOT convert it to English or formal Hindi. "
        "If the user wrote dialogue in any language using Roman script, preserve it exactly as-is. "
        "When dialogue has special characters, quotes, or non-English text, ensure proper JSON escaping. "
        ""
        "ABSOLUTE RULE - TYPE FIELD: ANY scene that contains 'dialogue_text' MUST have its 'type' set to 'dialogue'. "
        "NEVER use 'visual_segment' for a scene that has dialogue_text. This is the #1 most important rule. "
        "If a character speaks in a scene, that scene's type MUST be 'dialogue', period. "
        ""
        "CRITICAL - EVERY SCENE MUST HAVE DIALOGUE: The user requires that ALL scenes contain dialogue. "
        "Distribute the script's dialogue evenly across ALL scenes. Do NOT leave any scene without dialogue_text. "
        "If the script has limited dialogue, split existing dialogue into smaller parts so every scene has some speech. "
        "Every single scene object MUST have a non-empty 'dialogue_text' field and type='dialogue'. Zero silent scenes allowed. "
        "CRITICAL - DIALOGUE QUALITY & LANGUAGE PRESERVATION: "
        "Each scene's dialogue MUST be contextually relevant to the visual action in that scene. "
        "COPY the dialogue VERBATIM from the original script — do NOT translate, rephrase, or rewrite it. "
        "If the original script uses Hinglish (Hindi written in Roman/English script, e.g. 'Yaar, kya scene hai?'), "
        "then ALL dialogue_text MUST also be in Hinglish exactly as written. Do NOT convert Hinglish to English or to Devanagari Hindi. "
        "If the script is in Hindi, keep Hindi. If English, keep English. NEVER change the language of the dialogue. "
        "The dialogue must make narrative sense — split at natural pause points (sentence boundaries like '.', '!', '?', '|') "
        "so each chunk is a complete, meaningful thought. "
        ""
        "CRITICAL - DIALOGUE DISTRIBUTION: If the script is dialogue-heavy, do NOT create long stretches of silent 'visual_segment' scenes. "
        "Interleave dialogue throughout the scenes naturally. A 10-15 second visual-only segment in the middle of a conversation looks unnatural. "
        "If dialogue exists nearby in the script, split it so each scene has some dialogue rather than bunching all dialogue into a few scenes and leaving others silent. "
        "Short visual transitions (1-2 scenes) between dialogue are fine, but avoid 3+ consecutive silent visual segments when dialogue is available. "
        ""
        "CRITICAL - LANGUAGE & ACCENT CONSISTENCY: Detect the language style from the script. "
        "If the script contains Hinglish (Hindi words written in Roman script like 'kya', 'hai', 'yaar', 'bhai', 'arre', 'accha'), "
        "set language to 'Hinglish' and accent to 'native Hindi desi'. "
        "If the script is in pure Hindi (Devanagari), set language to 'Hindi' and accent to 'native Hindi desi'. "
        "If the script is in English, set language to 'English' and accent to 'neutral American'. "
        "Add top-level 'language' and 'accent' fields to the JSON output. "
        ""
        "CRITICAL - VISUAL PROMPT MUST NOT CONTAIN SILENCE MARKERS FOR DIALOGUE SCENES: "
        "If a scene has 'dialogue_text', the 'visual_prompt' field MUST NOT contain any phrases like "
        "'No speech', 'No dialogue', 'No voiceover', 'No talking', 'Silent scene', or 'No human voice'. "
        "These silence instructions destroy the dialogue during video generation. Only describe the visual action in 'visual_prompt'. "
        ""
        "CRITICAL - SCENE UNIQUENESS: Each scene MUST have a visually DISTINCT 'visual_prompt'. "
        "Do NOT repeat or closely paraphrase the same visual description across multiple scenes. "
        "Each scene must describe a DIFFERENT camera angle, character action, setting detail, or moment in the story. "
        "If two scenes involve the same characters talking, differentiate them by: camera angle (close-up vs wide), character positioning, expressions, gestures, background elements, or lighting changes. "
        ""
        "If a scene contains on-screen text or titles, use a 'title_card' type with a 'text' field. "
        "IMPORTANT: Every scene object must have this structure: {\"type\": \"...\", \"character\": \"...\", \"dialogue_text\": \"...\", \"visual_prompt\": \"...\"} "
        "The top-level JSON must include: {\"global_visual_description\": \"...\", \"language\": \"...\", \"accent\": \"...\", \"characters\": [...], \"scenes\": [...]} "
        "ENSURE THE FINAL OUTPUT IS VALID, PROPERLY FORMATTED JSON. Each array element is a complete object. "
        "Your entire response must be ONLY the JSON object. Do not include any conversational text, introductions, or explanations. Your response must start with `{` and end with `}`."
        "\n\nExample for dialogue scene (Hinglish — Hindi written in Roman script, COPY AS-IS):"
        """
        {
          "type": "dialogue",
          "character": "ATHER",
          "dialogue_text": "Yaar, kya scene hai? Main toh pagal ho gaya.",
          "visual_prompt": "ATHER looks toward the horizon with an excited expression..."
        }
        """
        "\n\nExample for a Hinglish script with dialogue (KEEP ORIGINAL LANGUAGE):"
        """
        {
          "global_visual_description": "A cinematic live-action sunset beach scene...",
          "language": "Hinglish",
          "accent": "native Hindi desi",
          "characters": [
            {"name": "ATHER", "description": "Indian male, early 40s..."}
          ],
          "scenes": [
            {"type": "visual_segment", "visual_prompt": "Wide establishing shot of ocean at sunset..."},
            {"type": "dialogue", "character": "ATHER", "dialogue_text": "Arre yaar, dekho kitna khoobsurat hai ye jagah.", "visual_prompt": "Medium close-up of ATHER looking toward the horizon with warm smile..."},
            {"type": "visual_segment", "visual_prompt": "Close-up of waves gently hitting the shore with golden reflections..."}
          ]
        }
        """
        "\n\nExample for a tutorial (e.g., baking):"
        """
        {
          "global_visual_description": "A bright, clean, top-down shot of a kitchen counter, style of a Bon Appétit video.",
          "scenes": [
            {"type": "instruction_step", "visual_prompt": "Hands cracking two eggs into a glass bowl of flour."},
            {"type": "title_card", "text": "Step 2: Whisk the dry ingredients"},
            {"type": "instruction_step", "visual_prompt": "A whisk mixes the flour, sugar, and salt in the bowl in a time-lapse."}
          ]
        }
        """
    )

    # Conditionally remove the "EVERY SCENE MUST HAVE DIALOGUE" block if force_dialogue is off
    if not st.session_state.get("force_dialogue_all_scenes", False):
        system_prompt = system_prompt.replace(
            "CRITICAL - EVERY SCENE MUST HAVE DIALOGUE: The user requires that ALL scenes contain dialogue. "
            "Distribute the script's dialogue evenly across ALL scenes. Do NOT leave any scene without dialogue_text. "
            "If the script has limited dialogue, split existing dialogue into smaller parts so every scene has some speech. "
            "Every single scene object MUST have a non-empty 'dialogue_text' field and type='dialogue'. Zero silent scenes allowed. ",
            ""
        )

    try:
        response = ollama.chat(
            model=model,
            format='json', # Enforce JSON output mode. This is more reliable.
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': detailed_script}
            ],
            options={'temperature': 0.5} # Lower temperature for more structured output
        )
        
        content = response['message']['content']
        logging.info(f"Ollama response received (partial): {content[:500]}...")

        # --- Robust JSON Cleaning ---
        # 1. Attempt to find the JSON block, ignoring conversational text or markdown.
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start == -1 or json_end == -1 or json_end <= json_start:
            raise ValueError("No JSON object found in Ollama's response.")
        
        json_str = content[json_start:json_end]
        # Use safe JSON parser that handles control characters
        parsed_data = safely_parse_json_with_control_chars(json_str)
        
        # --- NEW RESILIENCE LOGIC ---
        # Sometimes the LLM returns a dictionary for 'characters' or 'scenes' like {"0": {...}, "1": {...}}
        # instead of a list. Let's try to fix that.
        for key in ["characters", "scenes"]:
            if isinstance(parsed_data.get(key), dict):
                logging.warning(f"LLM returned a dictionary for '{key}', attempting to convert to a list.")
                try:
                    # Sort by key to maintain order and convert values to a list
                    parsed_data[key] = [v for k, v in sorted(parsed_data[key].items())]
                except Exception as e:
                    logging.error(f"Failed to convert '{key}' dictionary to list: {e}")
        
        # Basic validation of the parsed structure
        if "global_visual_description" not in parsed_data or "scenes" not in parsed_data:
            raise ValueError("Parsed JSON is missing required top-level keys (global_visual_description, scenes).")
        
        # Characters are now optional
        if "characters" in parsed_data:
            if not isinstance(parsed_data["characters"], list):
                raise ValueError("Characters list, if present, must be a list.")
            # Normalize: convert string entries to dict entries
            normalized_chars = []
            for c in parsed_data["characters"]:
                if isinstance(c, str):
                    normalized_chars.append({"name": c, "description": "character"})
                elif isinstance(c, dict) and "name" in c:
                    normalized_chars.append(c)
                else:
                    logging.warning(f"Skipping malformed character entry: {c}")
            parsed_data["characters"] = normalized_chars
        
        # Validate scenes - with better error reporting
        if not isinstance(parsed_data["scenes"], list):
            raise ValueError(f"Scenes is not a list. Got type: {type(parsed_data['scenes'])}")
        
        # Check each scene element
        for idx, s in enumerate(parsed_data["scenes"]):
            if not isinstance(s, dict):
                logging.error(f"Scene {idx} is not a dict: {s}")
                raise ValueError(f"Scene {idx} is not a valid object. Got type: {type(s)}")
            if "type" not in s:
                logging.error(f"Scene {idx} missing 'type' field: {s}")
                raise ValueError(f"Scene {idx} missing required 'type' field: {s}")

        status_text.success(f"✅ AI Script parsed successfully! Found {len(parsed_data['scenes'])} events.")
        logging.info(f"Successfully parsed script into {len(parsed_data['scenes'])} events.")
        return parsed_data

    except ollama.ResponseError as e:
        st.error(f"Ollama API Error: {e.error}")
        logging.error(f"Ollama API Error: {e.error}")
        if "model not found" in e.error:
            st.error(f"Please make sure the model '{model}' is available in Ollama.")
        return None
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Failed to parse AI response. The model did not return valid JSON or expected structure. Error: {e}. Raw content: {content}")
        logging.error(f"JSON parsing failed: {e}. Raw content: {content}")
        return None # Ensure return None is always reached
    except Exception as e:
        st.error(f"An unexpected error occurred with Ollama: {e}")
        logging.error(f"Ollama integration error: {e}", exc_info=True)
        return None

def queue_prompt_and_wait(prompt_workflow, client_id, progress_bar, status_text, comfyui_output_dir, node_id_mapping):
    if not check_comfyui_health(status_text):
        return None

    # Validate JSON serialization before sending
    try:
        workflow_json = json.dumps(prompt_workflow, indent=2)
        logging.debug(f"Prompt workflow JSON size: {len(workflow_json)} bytes")
        
        if len(workflow_json) > 10_000_000:  # 10MB limit
            st.error(f"❌ Workflow is too large ({len(workflow_json)} bytes). This may exceed ComfyUI's limits.")
            logging.error(f"Workflow JSON too large: {len(workflow_json)} bytes")
            return None
    except (TypeError, ValueError) as e:
        st.error(f"❌ Failed to serialize workflow to JSON: {e}")
        logging.error(f"JSON serialization failed: {e}", exc_info=True)
        return None
    
    logging.debug(f"Sending prompt to ComfyUI: {workflow_json[:500]}...")  # Log first 500 chars only
    
    # DEBUG: Log the actual prompt text being sent
    try:
        # Try to extract the prompt from json for verification
        workflow_dict = json.loads(workflow_json)
        if "92:3" in workflow_dict and "inputs" in workflow_dict["92:3"]:
            sent_prompt = workflow_dict["92:3"]["inputs"]["text"]
            logging.info(f"ACTUAL PROMPT SENT TO ComfyUI Node 92:3: {sent_prompt[:200]}...")
            if "Dialogue:" in sent_prompt or "lip sync" in sent_prompt:
                logging.info(f"✅ Dialogue confirmed in ComfyUI submission")
            else:
                logging.warning(f"⚠️ Dialogue may NOT be in ComfyUI submission")
    except Exception as e:
        logging.warning(f"Could not extract prompt for debug: {e}")
    
    try:
        req = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": prompt_workflow, "client_id": client_id}, timeout=10)
        req.raise_for_status()
        prompt_id = req.json()['prompt_id']
        logging.info(f"✅ Prompt queued with ID: {prompt_id}")

        ws = websocket.WebSocket()
        ws.connect(COMFYUI_WS_URL + f"?clientId={client_id}")
        logging.info("🔌 WebSocket connected.")
        ws.settimeout(300.0) # 5-minute timeout to handle long generation times

        while True:
            try:
                out = ws.recv()
                if not isinstance(out, str): continue
                message = json.loads(out)
                logging.debug(f"Received WS message: {message['type']}")

                if message['type'] == 'status':
                    queue_remaining = message['data']['status']['exec_info']['queue_remaining']
                    status_text.info(f"⏳ In queue... {queue_remaining} tasks remaining.")
                elif message['type'] == 'progress':
                    data = message['data']
                    progress = data['value'] / data['max']
                    progress_bar.progress(progress, text=f"Executing node... {int(progress * 100)}%")
                elif message['type'] == 'execution_started' and message['data']['prompt_id'] == prompt_id:
                    status_text.info("🚀 Execution has started!")
                elif message['type'] == 'execution_error' and message['data']['prompt_id'] == prompt_id:
                    error_data = message['data']
                    error_msg = error_data.get('exception_message', 'Unknown error')
                    st.error(f"❌ ComfyUI Execution Error: {error_msg}")
                    if "prompt" in error_msg.lower() or "text" in error_msg.lower():
                        st.warning("💡 This might be a prompt-related error. Check that your prompt doesn't contain unsupported characters or is too long.")
                    logging.error(f"ComfyUI Execution Error for prompt {prompt_id}: {json.dumps(error_data, indent=2)}")
                    ws.close()
                    return None
                elif message['type'] == 'executed' and message['data']['prompt_id'] == prompt_id:
                    node_id = message['data']['node']
                    # We only care about the output from the final save node.
                    if node_id == node_id_mapping["save"]:
                        logging.info(f"✅ Save node {node_id} executed. Processing output...")
                        output_data = message['data'].get('output', {})
                        
                        # The output can be in 'videos' or 'images' key
                        files_created = output_data.get('videos', []) + output_data.get('images', [])

                        if not files_created:
                            st.error(f"Save node executed, but ComfyUI did not report any saved files. This points to an internal error in the workflow. Please check ComfyUI logs.")
                            logging.error(f"Save node {node_id} reported no output files.")
                            ws.close()
                            return None

                        # Process the first file reported by the save node
                        file_info = files_created[0]
                        filename = file_info.get('filename')
                        subfolder = file_info.get('subfolder')
                        
                        # Construct the full path to the output file
                        found_path = os.path.join(COMFYUI_REAL_OUTPUT_DIR, subfolder, filename)
                        logging.info(f"ComfyUI reported saved file: {found_path}")
                        ws.close() # We have our file path, we can close the connection

                        if _wait_for_file_to_be_written(found_path, status_text):
                            destination_path = os.path.join(comfyui_output_dir, filename)
                            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                            shutil.copy2(found_path, destination_path)
                            logging.info(f"✅ Successfully copied '{filename}'.")
                            return destination_path
                        else:
                            st.error(f"ComfyUI reported file '{filename}', but it could not be found or read from disk. Check file permissions and ComfyUI logs.")
                            logging.error(f"Failed to find or read reported file: {found_path}")
                            return None
                    else:
                        logging.debug(f"Node {node_id} executed. Waiting for save node {node_id_mapping['save']}...")
                        continue # Wait for the next message
            except websocket.WebSocketTimeoutException:
                logging.warning("WebSocket timeout. Checking ComfyUI health...")
                if not check_comfyui_health(status_text):
                    st.error("Lost connection to ComfyUI during generation.")
                    ws.close()
                    return None
                logging.info("ComfyUI is alive, continuing to wait.")
                continue
    except requests.exceptions.Timeout:
        st.error("❌ Request to ComfyUI timed out. The server may be overloaded or unresponsive.")
        logging.error("ComfyUI request timeout")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to communicate with ComfyUI: {e}")
        logging.error(f"ComfyUI request failed: {e}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.error(f"Error in queue_prompt_and_wait: {e}", exc_info=True)
        return None
    return None



def score_frame_for_character_presence(frame):
    """
    Score a frame based on character presence using multiple heuristics:
    - Edge density (higher edges = more detail/characters)
    - Non-black pixel ratio
    - Contrast and variance
    - Contour complexity (more contours = more objects/characters)
    - Color range (diverse colors suggest characters vs solid background)
    Returns a score between 0 and 1.
    """
    try:
        if frame is None or frame.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Edge detection (Canny) - character edges are important
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size if edges.size > 0 else 0
        
        # 2. Non-black pixel ratio (characters usually aren't pure black)
        non_black_pixels = np.count_nonzero(gray > 30)
        non_black_ratio = non_black_pixels / gray.size if gray.size > 0 else 0
        
        # 3. Variance/contrast in the image (more variance = more content)
        variance = np.var(gray) / 255.0  # Normalize to 0-1
        
        # 4. Contour detection - more contours often means more objects (characters)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours if cv2.contourArea(c) > 50]  # Filter tiny noise
        contour_score = min(1.0, len(significant_contours) / 50.0)  # Normalize (expect ~50 contours)
        
        # 5. Color diversity - characters usually have varied colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        unique_colors = len(np.unique(hsv.reshape(-1, hsv.shape[2]), axis=0))
        color_diversity = min(1.0, unique_colors / 10000.0)  # Normalize to 0-1
        
        # 6. Center focus - characters are often in center (weight center higher)
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        center_region = gray[max(0, center_y-h//3):min(h, center_y+h//3), 
                            max(0, center_x-w//3):min(w, center_x+w//3)]
        center_brightness = np.mean(center_region) / 255.0 if center_region.size > 0 else 0
        
        # Combine scores with weights
        # Edge ratio is most important (character details)
        # Non-black ratio helps avoid blank/faded frames
        # Contours help identify distinct character shapes
        # Color diversity helps avoid monochromatic backgrounds
        # Variance helps avoid uniform color frames
        # Center brightness helps prioritize frames where content is center-focused
        score = (
            edge_ratio * 0.35 +           # 35% edges (most reliable)
            non_black_ratio * 0.15 +      # 15% non-black pixels
            variance * 0.15 +             # 15% contrast
            contour_score * 0.20 +        # 20% contour complexity (new)
            color_diversity * 0.10 +      # 10% color diversity (new)
            center_brightness * 0.05      # 5% center focus (reduced)
        )
        
        logging.debug(f"Frame score breakdown - edges:{edge_ratio:.3f} non_black:{non_black_ratio:.3f} var:{variance:.3f} contours:{contour_score:.3f} colors:{color_diversity:.3f} center:{center_brightness:.3f} → total:{score:.3f}")
        
        return max(0.0, min(1.0, score))  # Clamp to 0-1
    except Exception as e:
        logging.error(f"Error scoring frame for character presence: {e}")
        return 0.0


def extract_best_frame_for_continuity(video_path, output_image_path, num_samples=5):
    """
    Intelligently extracts the best frame from a video for continuity/character consistency.
    Instead of always using the last frame, this analyzes multiple frames and selects
    the one with the strongest character presence (edges, detail, non-black pixels).
    
    Args:
        video_path: Path to the generated video
        output_image_path: Where to save the selected frame
        num_samples: Number of frames to sample throughout the video (default 5)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error: Could not open video {video_path}")
            return False
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            return False
        
        # Sample frames at regular intervals: 0%, 25%, 50%, 75%, 100%
        sample_positions = [int(frame_count * (i / (num_samples - 1))) for i in range(num_samples)]
        
        best_frame = None
        best_score = -1
        best_position = -1
        frames_analyzed = []
        
        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                score = score_frame_for_character_presence(frame)
                frames_analyzed.append((pos, score))
                
                if score > best_score:
                    best_score = score
                    best_frame = frame
                    best_position = pos
                
                logging.debug(f"Frame at position {pos}/{frame_count} scored {score:.3f}")
        
        cap.release()
        
        if best_frame is None:
            logging.error("No suitable frame found in video")
            return False
        
        # Save the best frame
        cv2.imwrite(output_image_path, best_frame)
        
        # Log which frame was selected
        frame_percentage = (best_position / frame_count * 100) if frame_count > 0 else 0
        logging.info(f"Selected best continuity frame: position {best_position}/{frame_count} ({frame_percentage:.1f}%) with score {best_score:.3f}")
        
        # Display detailed frame analysis in logs
        scores_summary = " | ".join([f"{int((pos/frame_count)*100)}%({s:.2f})" for pos, s in frames_analyzed])
        logging.info(f"Frame analysis: {scores_summary} → Best: {best_score:.3f}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error extracting best frame: {e}", exc_info=True)
        return False


def extract_last_frame(video_path, output_image_path):
    """Legacy function for extracting the last frame. Now calls the smart version."""
    return extract_best_frame_for_continuity(video_path, output_image_path, num_samples=5)

def stitch_videos(video_files, output_file):
    # ... (This function remains the same)
    if not video_files: return
    if len(video_files) == 1:
        import shutil
        shutil.copy(video_files[0], output_file)
        return
    list_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for video_file in video_files:
            f.write(f"file '{os.path.abspath(video_file)}'\n")
    try:
        command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_file]
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        logging.info(f"FFmpeg stitch successful. Output:\n{result.stdout}")
        st.success(f"Final video saved to {output_file}")
    except subprocess.CalledProcessError as e:
        # Log the detailed error from ffmpeg, which is crucial for debugging
        error_message = f"FFmpeg Error: Failed to stitch videos. Return code: {e.returncode}\n\nFFmpeg stderr:\n{e.stderr}"
        logging.error(error_message)
        st.error(error_message)
    except Exception as e:
        logging.error(f"An unexpected error occurred during video stitching: {e}", exc_info=True)
        st.error(f"An unexpected error occurred during video stitching: {e}")
    finally:
        if os.path.exists(list_path): os.remove(list_path)


def upscale_video_ffmpeg(input_path, output_path, target_width, target_height, crf=18):
    """
    Upscale a video to target resolution using FFmpeg lanczos filter.
    Preserves audio stream. Uses H.264 encoding with configurable CRF quality.
    
    Args:
        input_path: Path to input video
        output_path: Path to save upscaled video
        target_width: Target width in pixels
        target_height: Target height in pixels
        crf: Constant Rate Factor (lower = better quality, 18 = visually lossless)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        command = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", f"scale={target_width}:{target_height}:flags=lanczos",
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", "slow",
            "-c:a", "copy",
            output_path
        ]
        logging.info(f"Upscaling video: {input_path} -> {target_width}x{target_height}")
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        logging.info(f"FFmpeg upscale successful: {output_path}")
        
        # Verify output exists and has size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            original_size = os.path.getsize(input_path) / (1024 * 1024)
            upscaled_size = os.path.getsize(output_path) / (1024 * 1024)
            logging.info(f"Upscale complete: {original_size:.1f}MB -> {upscaled_size:.1f}MB ({target_width}x{target_height})")
            return True
        else:
            logging.error("Upscaled video file is missing or empty")
            return False
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg upscale failed: {e.stderr}")
        st.error(f"❌ Video upscale failed: {e.stderr[:500]}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during video upscale: {e}", exc_info=True)
        st.error(f"❌ Unexpected error during upscale: {e}")
        return False


def cleanup_scene_videos(scene_video_paths, output_dir):
    """
    Deletes individual scene videos after final video is created to save disk space.
    Returns (success, total_space_freed_mb)
    """
    try:
        total_size = 0
        deleted_count = 0
        
        for video_path in scene_video_paths:
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                os.remove(video_path)
                total_size += file_size
                deleted_count += 1
                logging.info(f"Deleted scene video: {video_path}")
        
        space_freed_mb = total_size / (1024 * 1024)  # Convert to MB
        logging.info(f"Cleanup complete: Deleted {deleted_count} scene videos, freed {space_freed_mb:.2f} MB")
        return True, space_freed_mb
    except Exception as e:
        logging.error(f"Error during cleanup: {e}", exc_info=True)
        return False, 0

def cleanup_empty_directories(root_path):
    """
    Recursively delete empty directories (no files, only empty subdirectories).
    Returns the count of directories deleted.
    """
    try:
        deleted_dirs = 0
        
        # Walk through directories from deepest to shallowest
        for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
            # Skip if directory doesn't exist anymore (might have been deleted by previous iteration)
            if not os.path.exists(dirpath):
                continue
            
            # If there are files in this directory, don't delete it
            if filenames:
                continue
            
            # Check if directory is completely empty (no subdirectories with files)
            try:
                # Try to remove the directory
                os.rmdir(dirpath)
                deleted_dirs += 1
                logging.info(f"Deleted empty directory: {dirpath}")
            except OSError:
                # Directory might not be empty or permission denied
                pass
        
        if deleted_dirs > 0:
            logging.info(f"Cleanup: Deleted {deleted_dirs} empty directories")
        return deleted_dirs
    except Exception as e:
        logging.error(f"Error during empty directory cleanup: {e}", exc_info=True)
        return 0

def purge_comfyui_vram(status_text):
    """Queues a simple workflow that triggers a VRAM purge on the ComfyUI server."""
    purge_workflow_path = "workfllows/purge_vram_workflow.json"
    try:
        with open(purge_workflow_path, 'r', encoding='utf-8') as f:
            purge_workflow = json.load(f)

        client_id = str(uuid.uuid4()) # Use a unique client ID for this task
        logging.info("Queueing VRAM purge workflow...")
        status_text.info("🧹 Sending VRAM Purge command to server...")

        # We can use a simplified version of queue_prompt_and_wait since we don't expect a file output.
        req = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": purge_workflow, "client_id": client_id})
        req.raise_for_status()
        prompt_id = req.json()['prompt_id']

        try:
            ws = websocket.WebSocket()
            ws.connect(COMFYUI_WS_URL + f"?clientId={client_id}")
            ws.settimeout(120.0) # Increased timeout for the purge task

            while True:
                out = ws.recv()
                if not isinstance(out, str): continue
                message = json.loads(out)
                # We only care about the final 'executed' message for our prompt
                if message.get('type') == 'executed' and message.get('data', {}).get('prompt_id') == prompt_id:
                    logging.info("✅ VRAM purge workflow executed successfully.")
                    status_text.info("🧹 VRAM cache purged on server.")
                    ws.close()
                    break
        except websocket.WebSocketException as e:
            logging.error(f"VRAM purge failed due to WebSocket error: {e}")
            status_text.warning("Could not connect to ComfyUI to purge VRAM.")
    except FileNotFoundError:
        logging.warning(f"`{purge_workflow_path}` not found. Skipping VRAM purge.")
    except Exception as e:
        logging.error(f"An error occurred during VRAM purge: {e}", exc_info=True)

def generate_character(prompt, character_name, comfyui_input_dir, comfyui_output_dir, nodes, run_id, workflow_template):
    """
    Generates a character image by running the video workflow for a single frame
    and then extracting that frame as a PNG image.

    This is a workaround to produce a still image using a text-to-video workflow.
    The final character image is saved to the application's 'input' directory
    to be used for subsequent scene generation.
    """
    st.info(f"Generating image for {character_name}...")
    
    # Validate the prompt before using it
    is_valid, validated_prompt, warnings = validate_and_sanitize_prompt(prompt)
    
    if warnings:
        for warning in warnings:
            st.warning(warning)
    
    if not is_valid:
        st.error(f"❌ Cannot generate character '{character_name}': Prompt validation failed.")
        return None
    
    client_id = str(uuid.uuid4())
    
    # Use a deep copy of the template to avoid modifying the original
    workflow = copy.deepcopy(workflow_template)

    # --- Apply Global Negative Prompt ---
    if nodes.get("negative_prompt") and st.session_state.get("negative_prompt_text"):
        neg_prompt_is_valid, neg_prompt_sanitized, neg_warnings = validate_and_sanitize_prompt(st.session_state.negative_prompt_text)
        if neg_prompt_is_valid:
            workflow[nodes["negative_prompt"]]["inputs"]["text"] = neg_prompt_sanitized
            logging.info(f"Applying global negative prompt to character generation for '{character_name}'")
        else:
            st.warning("⚠️ Negative prompt validation failed. Skipping negative prompt for this character.")

    # Sanitize character name for filename
    safe_char_name = "".join(c for c in character_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
    
    # Create a unique prefix for this generation
    filename_prefix = f"{run_id.split('-')[0]}_character_{safe_char_name}"

    # Modify workflow for image generation
    workflow[nodes["prompt"]]["inputs"]["text"] = validated_prompt
    workflow[nodes["seed"]]["inputs"]["seed"] = random.randint(0, 1_000_000_000)
    workflow[nodes["save"]]["inputs"]["filename_prefix"] = filename_prefix
    
    # --- FIX: Start character generation from a blank slate ---
    workflow[nodes["image"]]["inputs"]["image"] = BLANK_IMAGE_FILENAME
    
    logging.info(f"Character generation workflow: Prompt='{validated_prompt[:100]}...', Seed='{workflow[nodes['seed']]['inputs']['seed']}', Prefix='{filename_prefix}'")
    
    # Set the length to 1 to generate a single image, using the mapped node ID
    if nodes.get("frame_count"):
        workflow[nodes["frame_count"]]["inputs"]["value"] = 1
    else:
        st.warning("Frame Count node not mapped. Cannot set character image frame count to 1. The result might be a short video.")

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Queue and wait for the character image
    generated_video_path = queue_prompt_and_wait(workflow, client_id, progress_bar, status_text, comfyui_output_dir, nodes)

    if generated_video_path and os.path.exists(generated_video_path):
        st.success(f"'{character_name}' (single frame video) generated!")
        
        # Extract the frame from the video. The output filename is now prefixed.
        # We can create a cleaner name for the final PNG in our input directory.
        character_image_filename = f"{safe_char_name}.png"
        character_image_path = os.path.join(comfyui_input_dir, character_image_filename)
        
        if extract_last_frame(generated_video_path, character_image_path):
            st.success(f"Image for '{character_name}' saved to {character_image_path}")
            st.image(character_image_path, caption=character_name)
            return character_image_path
        else:
            st.error(f"Failed to extract image for '{character_name}'.")
            return None
    else:
        st.error(f"Generation failed for '{character_name}'.")
        return None

# --- Streamlit UI ---
# (set_page_config already called at the top of file)

# --- Dependency Check (after set_page_config) ---
if not OLLAMA_AVAILABLE:
    st.error("❌ The 'ollama' library is not installed. Please install it by running: pip install ollama")
    st.stop()

# --- Session State Initialization ---
if 'run_id' not in st.session_state:
    st.session_state.run_id = str(uuid.uuid4())
    # This is the app's local directory for this specific run's final outputs
    st.session_state.app_run_output_dir = os.path.join(APP_OUTPUT_DIR, st.session_state.run_id)
    os.makedirs(st.session_state.app_run_output_dir, exist_ok=True)
    # --- New State Management ---
    st.session_state.stage = "script_input"
    st.session_state.parsed_script = None


# Load configuration at the start of the script
config = load_config()

with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Video Generation Settings")
    total_video_duration = st.slider("Total Video Duration (seconds)", min_value=12, max_value=600, value=60, step=12, help="Choose the total duration for the entire video. Will be divided into 12-second scenes.")
    num_scenes = max(1, total_video_duration // 12)
    st.info(f"📹 Video will be divided into {num_scenes} scene(s) of 12 seconds each.")
    
    scene_duration = st.slider("Scene Duration (seconds)", min_value=1, max_value=20, value=12, step=1, help="Set the duration for each generated video segment. Default is 12 seconds. Shorter durations require less VRAM and can prevent 'out of memory' errors.")
    scene_cooldown = st.slider("Cooldown Between Scenes (seconds)", min_value=0, max_value=60, value=10, step=1, help="Pause between generating each scene to allow GPU VRAM to be cleared. Increase this if you get 'out of memory' errors.")
    purge_vram = st.checkbox("Purge VRAM between scenes", value=True, help="Actively clear GPU memory on the ComfyUI server after each scene. Helps prevent 'out of memory' errors on long videos, but may slightly slow down generation.")
    
    st.subheader("Post-Processing Upscale")
    upscale_resolution = st.selectbox(
        "Final Video Resolution:",
        ["720p (1280×720) - No upscale", "1080p Full HD (1920×1080)", "2K (2560×1440)", "4K (3840×2160)"],
        index=1,
        help="Upscale the final stitched video using FFmpeg lanczos filter. Generation stays at 720p for stability, then the final video is upscaled. Higher resolutions = larger file size."
    )
    upscale_quality = st.slider("Upscale Quality (CRF)", min_value=15, max_value=30, value=18, step=1, help="Lower = better quality but larger file. 18 is visually lossless for most content. 23 is FFmpeg default.")
    # ... (Workflow mapping remains the same)
    if os.path.exists(WORKFLOW_FILE):
        with open(WORKFLOW_FILE, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
        node_titles = {node_id: node.get("_meta", {}).get("title", f"Untitled {node_id}") for node_id, node in workflow_data.items()}
        node_titles_list = sorted(list(set(node_titles.values()))) # Use a sorted, unique list of titles

        # This list is needed for optional dropdowns like IPAdapter
        optional_nodes_list = ["None"] + node_titles_list

        # --- Helper to Get Indices for Defaults ---
        def get_index(title, options):
            try:
                return options.index(title)
            except (ValueError, KeyError):
                return 0 # Default to first item if not found

        prompt_index = get_index(DEFAULT_TITLES["prompt"], node_titles_list)
        negative_prompt_index = get_index(DEFAULT_TITLES.get("negative_prompt", "Negative Prompt"), node_titles_list)
        seed_index = get_index(DEFAULT_TITLES["seed"], node_titles_list)
        image_index = get_index(DEFAULT_TITLES["image"], node_titles_list)
        save_index = get_index(DEFAULT_TITLES["save"], node_titles_list)
        frame_count_index = get_index(DEFAULT_TITLES["frame_count"], node_titles_list)
        frame_rate_index = get_index(DEFAULT_TITLES["frame_rate"], node_titles_list)
        ipadapter_index = get_index(DEFAULT_TITLES.get("ipadapter", "IPAdapter"), optional_nodes_list)

        # --- Create Select Boxes with Defaults ---
        prompt_node_title = st.selectbox("Prompt Node:", node_titles_list, index=prompt_index)
        negative_prompt_node_title = st.selectbox("Negative Prompt Node:", node_titles_list, index=negative_prompt_index, help="Select the node that takes the negative prompt text (e.g., a `CLIPTextEncode` node).")
        seed_node_title = st.selectbox("Seed Node:", node_titles_list, index=seed_index)
        image_input_node_title = st.selectbox("Image Input Node:", node_titles_list, index=image_index, help="Select the node that takes an image as input for continuity. Should be a 'LoadImage' node.")
        save_node_title = st.selectbox("Save Node:", node_titles_list, index=save_index)
        frame_count_node_title = st.selectbox("Frame Count/Length Node:", node_titles_list, index=frame_count_index, help="Node that controls the number of frames to generate (e.g., 'Video Length').")
        frame_rate_node_title = st.selectbox("Frame Rate Node:", node_titles_list, index=frame_rate_index, help="Node that controls the video's frame rate.")
        ipadapter_node_title = st.selectbox("IPAdapter Node (Optional):", optional_nodes_list, index=ipadapter_index, help="Select the IPAdapter node for character consistency. Requires a workflow with an IPAdapter.")
        
        # New selections for dialogue and ambient audio, if they exist in the workflow
        # These will need corresponding nodes in your ComfyUI workflow to be functional.
        dialogue_prompt_node_title = st.selectbox("Dialogue Prompt Node (Optional):", optional_nodes_list, help="Select a CLIPTextEncode-like node specifically for dialogue audio. Requires workflow modification.")
        ambient_audio_prompt_node_title = st.selectbox("Ambient Audio Prompt Node (Optional):", optional_nodes_list, help="Select a CLIPTextEncode-like node specifically for ambient audio. Requires workflow modification.")

        # Store node IDs in a dictionary for easier access
        node_id_mapping = {
            "prompt": next((id for id, title in node_titles.items() if title == prompt_node_title), None),
            "negative_prompt": next((id for id, title in node_titles.items() if title == negative_prompt_node_title), None),
            "seed": next((id for id, title in node_titles.items() if title == seed_node_title), None),
            "image": next((id for id, title in node_titles.items() if title == image_input_node_title), None), # Added 'image' key
            "save": next((id for id, title in node_titles.items() if title == save_node_title), None),
            "frame_count": next((id for id, title in node_titles.items() if title == frame_count_node_title), None),
            "frame_rate": next((id for id, title in node_titles.items() if title == frame_rate_node_title), None),
            "dialogue_prompt": next((id for id, title in node_titles.items() if title == dialogue_prompt_node_title and title != "None"), None),
            "ambient_audio_prompt": next((id for id, title in node_titles.items() if title == ambient_audio_prompt_node_title and title != "None"), None),
            "ipadapter": next((id for id, title in node_titles.items() if title == ipadapter_node_title and title != "None"), None)
        }
        
        # DEBUG: Log node ID mapping
        logging.info(f"Node ID Mapping:")
        logging.info(f"  prompt: {node_id_mapping['prompt']} (title: {prompt_node_title})")
        logging.info(f"  negative_prompt: {node_id_mapping['negative_prompt']}")
        logging.info(f"  seed: {node_id_mapping['seed']}")
        logging.info(f"  image: {node_id_mapping['image']}")
        logging.info(f"  save: {node_id_mapping['save']}")

        image_input_node_class_type = None
        if node_id_mapping["image"] and node_id_mapping["image"] in workflow_data:
            image_input_node_class_type = workflow_data[node_id_mapping["image"]].get("class_type")
            if image_input_node_class_type == "EmptyImage":
                st.error(f"❌ CRITICAL ERROR: 'Image Input Node' is set to 'EmptyImage' node. "
                         f"The i2v model REQUIRES a LoadImage node to load reference images. "
                         f"Please select the 'Load Image' node instead of 'EmptyImage' in the sidebar above.")
    else:
        st.error(f"Workflow file not found at {WORKFLOW_FILE}.")
        st.stop()
    
    st.markdown("---")
    st.warning("💡 **i2v Model Notes:**\n\n"
               "**Image Input is CRITICAL**: Unlike T2V models, i2v REQUIRES reference images for each scene.\n\n"
               "**Troubleshooting**:\n"
               "- If you get solid color or noisy video: Check VAE connectivity in your ComfyUI workflow\n"
               "- If characters look distorted or wrong: Try uploading higher quality reference images\n"
               "- If videos seem disconnected: Enable 'Extract continuity frames' to use last frame as reference for next scene")

    st.markdown("---")
    st.header("🤖 AI Script Parsing")
    parser_choice = st.radio("Select Script Parser", ["Ollama (Local)", "Gemini (API)"], help="**Ollama (Local)**: Free, no API key needed, runs on your computer. Recommended for reliability.\n\n**Gemini (API)**: Cloud-based, powerful, requires Google API key. May have model compatibility issues with older Python packages.")
    
    if parser_choice == "Ollama (Local)":
        ollama_model = st.text_input("Ollama Model Name:", value="llama3")
    else: # Gemini
        gemini_model = st.text_input("Gemini Model Name:", value="gemini-2.5-flash", help="Uses 'gemini-2.5-flash' by default (recommended). The app will auto-select a compatible model if unavailable.")
        
        saved_gemini_key = config.get("gemini_api_key", "")
        gemini_api_key = st.text_input(
            "Google AI Studio API Key:",
            type="password",
            value=saved_gemini_key,
            help="Get your key from Google AI Studio. Click 'Save Key' to store it for future sessions."
        )

        if st.button("Save Gemini API Key"):
            config["gemini_api_key"] = gemini_api_key
            save_config(config)
            st.success("✅ Gemini API Key saved to config.json!")


    force_dialogue_all_scenes = st.checkbox(
        "🗣️ Force Dialogue in Every Scene", value=False,
        help="When enabled, ALL scenes will have dialogue. The script's dialogue will be evenly distributed so no scene is silent. "
             "Ideal for 12-15 second scenes where silent segments look unnatural."
    )

    dialogue_language_accent = st.selectbox(
        "🌐 Dialogue Language & Accent",
        ["Auto-detect", "Hinglish — native Hindi desi accent", "Hindi — native Hindi desi accent", "English — neutral American accent", "English — British accent"],
        index=0,
        help="Force the language and accent for ALL dialogue in the video. "
             "'Auto-detect' tries to guess from your script. Choose explicitly if the auto-detection gets it wrong."
    )

    generate_char_images = st.checkbox("Enable Character Setup", value=False, help="Go to a character setup screen to upload or generate images. If unchecked, the video will be generated without specific character images.")


# ===== MAIN UI SECTION - IMAGE-TO-VIDEO INPUT =====
st.markdown("---")
st.header("🎥 Video Generation Setup")
st.info("**i2v Model Info**: This app uses the LTX i2v (Image-to-Video) model. You provide images, and the model creates animated videos based on your text prompts.")

# === Image Input Section ===
st.subheader("📸 Image Input Options")
st.write("For the i2v model, you need reference images. Choose how to provide them:")

image_input_method = st.radio(
    "Select image input method:",
    ["Single image for all scenes", "Upload images per-scene", "Generate from character setup", "Start with blank (will use first generated frame)"],
    horizontal=True
)

uploaded_single_image = None
uploaded_images_dict = {}

if image_input_method == "Single image for all scenes":
    uploaded_single_image = st.file_uploader(
        "Upload a reference image to use for all scenes:",
        type=["png", "jpg", "jpeg"],
        key="single_image_uploader",
        help="This image will be used as the starting point for every video scene"
    )
    if uploaded_single_image:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(uploaded_single_image, caption="Reference image preview")
        with col2:
            st.write(f"**Size**: {uploaded_single_image.size} bytes")
            
elif image_input_method == "Upload images per-scene":
    st.write("You'll be able to upload different images for different scenes later in the character setup.")
    
elif image_input_method == "Generate from character setup":
    st.write("Images will be generated from character descriptions during the character setup phase.")
    

detailed_story_script = st.text_area("Enter your detailed story script here (using the template structure):", height=400, value="""A high-quality cinematic 3D animated fantasy scene set in a glowing crystal valley at golden hour, filled with floating pollen lights, colorful mushrooms, shimmering waterfalls, and warm sunlight reflecting off rainbow-tinted rocks. The animation style is premium 3D with rich vibrant colors, soft global illumination, expressive faces, rounded character design, cinematic depth of field, detailed textures, and slow professional camera movement.

There are exactly two animated characters, with fixed identity throughout the video:

EMBER – baby dragon, small and round, smooth coral-red scales with golden highlights, big sparkling amber eyes, tiny wings, soft glowing chest, curious and shy personality.

LYRA – young girl explorer, age 8–10, warm brown skin, short wavy hair, colorful scarf, light adventure outfit, kind and encouraging nature.

Identities must remain stable. No character merging. No face swapping.

The camera opens with a slow wide establishing shot of the crystal valley as glowing particles drift through the air. Waterfalls sparkle in the distance. The camera gently pushes forward toward EMBER sitting on a mossy rock, nervously tapping his tiny claws. LYRA stands nearby holding a small glowing lantern. Only subtle motion: soft breathing, gentle head tilts, light hand gestures.

The camera settles into a medium-wide shot of both characters.

EMBER looks down at his paws and speaks first in a soft, nervous baby voice:
EMBER: “Lyra… everyone says dragons are brave. But my fire won’t even glow.”

The camera slowly moves closer to LYRA. EMBER becomes silent.

LYRA kneels beside him, smiling warmly, and replies gently:
LYRA: “Maybe you’re trying too hard. Magic doesn’t come from force… it comes from feeling.”

LYRA stops speaking. The camera subtly shifts back to EMBER.

EMBER lifts his head, eyes wide, and asks quietly:
EMBER: “Feeling… like what?”

The camera eases toward LYRA again. EMBER becomes silent.

LYRA places her hand softly on EMBER’s glowing chest and says with kindness:
LYRA: “Like joy. Like courage. Like believing in yourself.”

Cut to a short silent montage:
– LYRA spinning slowly as glowing petals float around her.
– EMBER flapping his tiny wings once, wobbling cutely.
– The valley lights pulsing gently in rhythm.

Return to both characters in the same positions.

The camera centers on EMBER. LYRA is silent. EMBER closes his eyes, takes a small breath, and speaks with growing confidence:
EMBER: “Okay… I’ll try.”

EMBER opens his mouth and releases a tiny stream of rainbow-tinted flame that blossoms into sparkling light butterflies. His chest glows bright gold. His eyes widen in amazement.

The camera slowly pulls back to include LYRA.

LYRA laughs softly and says with pride:
LYRA: “See? You already had it inside you.”

EMBER looks at LYRA, smiling for the first time, and replies happily:
EMBER: “Then I’m ready to shine.”

Lips move only for the active speaker. Facial expressions are soft and emotional. Eyes blink naturally. Subtle breathing is visible. EMBER’s wings flutter gently. LYRA’s scarf moves lightly in the breeze. Lighting shifts dynamically as the valley glows brighter. Camera motion remains slow and cinematic.

The scene ends with a wide magical shot of EMBER and LYRA standing together as glowing butterflies rise into the colorful sky, conveying courage, friendship, and self-belief.""")

negative_prompt_text = st.text_area("Global Negative Prompt:", value="blurry, low quality, deformed, extra limbs, mutated hands, poorly drawn hands, poorly drawn face, mutation, ugly, bad anatomy, bad proportions, watermark, text, signature, extra character, fused characters", height=100, help="Enter terms to avoid during generation. These will be applied to all scenes.")

if st.session_state.stage == "script_input":
    if not detailed_story_script.strip():
        st.warning("Please enter a detailed story script.")
    
    if st.button("📝 Parse Script & Setup Characters"):
        required_nodes = ["prompt", "seed", "image", "save", "frame_count", "frame_rate"]
        st.session_state.negative_prompt_text = negative_prompt_text # Store for use in the generation stage
        st.session_state.image_input_method = image_input_method  # Store image input method
        st.session_state.force_dialogue_all_scenes = force_dialogue_all_scenes  # Store for post-processing
        st.session_state.dialogue_language_accent = dialogue_language_accent  # Store language/accent choice
        
        # Save uploaded image to disk IMMEDIATELY - BytesIO objects don't survive Streamlit reruns
        if image_input_method == "Single image for all scenes" and uploaded_single_image:
            try:
                img_filename = "reference_image_single.png"
                img_path = os.path.join(COMFYUI_REAL_INPUT_DIR, img_filename)
                with open(img_path, 'wb') as img_f:
                    img_f.write(uploaded_single_image.getbuffer())
                st.session_state.saved_single_image_filename = img_filename
                logging.info(f"Saved single reference image to {img_path} at parse time")
            except Exception as img_err:
                logging.error(f"Failed to save reference image at parse time: {img_err}")
                st.session_state.saved_single_image_filename = None
        else:
            st.session_state.saved_single_image_filename = None
        
        if not all(node_id_mapping.get(key) for key in required_nodes):
            st.error(f"One or more required ComfyUI nodes are not mapped. Please check all sidebar selections: {', '.join(required_nodes)}")
        elif not detailed_story_script.strip():
            st.warning("Please enter a detailed story script before parsing.")
        else:
            status_area = st.empty()
            parsed_script = None
            if parser_choice == "Ollama (Local)":
                parsed_script = parse_detailed_script_with_ollama(detailed_story_script, ollama_model, status_area, num_scenes=num_scenes)
            else: # Gemini
                if 'google.generativeai' not in sys.modules:
                    st.error("The 'google-generativeai' library is not installed. Please run: pip install google-generativeai")
                elif not gemini_api_key:
                    st.error("Please enter your Google AI Studio API Key in the sidebar to use the Gemini parser.")
                else:
                    # Validate model name - gemini-1.5-flash-latest may not exist
                    normalized_model = gemini_model.replace("-latest", "")  # Remove -latest suffix if present
                    valid_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-pro"]
                    if normalized_model not in valid_models:
                        if "gemini-1.5" in normalized_model or "gemini-2.0" in normalized_model:
                            # Try to find the closest match
                            for valid in valid_models:
                                if valid.startswith(normalized_model[:10]):
                                    normalized_model = valid
                                    break
                        else:
                            normalized_model = "gemini-1.5-flash"  # Fallback to reliable model
                    parsed_script = parse_detailed_script_with_gemini(detailed_story_script, gemini_api_key, normalized_model, status_area, num_scenes=num_scenes)

            if parsed_script:
                # --- POST-PROCESSING: Ensure we have enough scenes ---
                scenes = parsed_script.get("scenes", [])
                visual_scenes = [s for s in scenes if s.get("type") in ["visual_segment", "dialogue", "montage"]]
                if len(visual_scenes) < num_scenes:
                    logging.warning(f"LLM returned {len(visual_scenes)} visual scenes but {num_scenes} are needed. Padding scenes...")
                    st.warning(f"⚠️ LLM returned {len(visual_scenes)} scenes, but {num_scenes} are needed. Auto-splitting/padding to fill the gap.")
                    
                    # Strategy: cycle through existing scenes to fill the gap
                    if len(visual_scenes) == 0:
                        # No visual scenes at all - create placeholder scenes from the script
                        base_desc = parsed_script.get("global_visual_description", "Cinematic scene")
                        for idx in range(num_scenes):
                            parsed_script["scenes"].append({
                                "type": "visual_segment",
                                "character": "",
                                "dialogue_text": "",
                                "visual_prompt": f"{base_desc} - Scene {idx + 1} continuation with smooth cinematic movement."
                            })
                    else:
                        # We have some scenes but not enough - duplicate/cycle existing ones
                        original_visual = list(visual_scenes)  # copy
                        while len(visual_scenes) < num_scenes:
                            source = original_visual[len(visual_scenes) % len(original_visual)]
                            new_scene = copy.deepcopy(source)
                            new_scene["visual_prompt"] = new_scene.get("visual_prompt", "") + " Continuation with subtle camera movement and natural progression."
                            parsed_script["scenes"].append(new_scene)
                            visual_scenes.append(new_scene)
                        logging.info(f"Padded scenes to {len(parsed_script['scenes'])} total ({num_scenes} visual scenes)")
                
                # --- POST-PROCESSING: Language/Accent from user's explicit choice ---
                _user_lang_choice = st.session_state.get("dialogue_language_accent", "Auto-detect")
                
                if _user_lang_choice != "Auto-detect":
                    # User explicitly chose a language/accent — override everything
                    _lang_map = {
                        "Hinglish — native Hindi desi accent": ("Hinglish", "native Hindi desi"),
                        "Hindi — native Hindi desi accent": ("Hindi", "native Hindi desi"),
                        "English — neutral American accent": ("English", "neutral American"),
                        "English — British accent": ("English", "British"),
                    }
                    _chosen_lang, _chosen_accent = _lang_map.get(_user_lang_choice, ("English", "neutral"))
                    old_lang = parsed_script.get('language', '')
                    old_accent = parsed_script.get('accent', '')
                    parsed_script["language"] = _chosen_lang
                    parsed_script["accent"] = _chosen_accent
                    logging.info(f"User explicitly set language/accent: '{_chosen_lang}' / '{_chosen_accent}' (was: '{old_lang}' / '{old_accent}')")
                    st.info(f"🌐 Language/Accent set to: {_chosen_lang} with {_chosen_accent} accent")
                else:
                    # Auto-detect mode: set defaults then try Hinglish detection
                    if "language" not in parsed_script or not parsed_script.get("language"):
                        parsed_script["language"] = "English"
                        logging.info("Language not detected by LLM, defaulting to 'English'")
                    if "accent" not in parsed_script or not parsed_script.get("accent"):
                        parsed_script["accent"] = "neutral"
                        logging.info("Accent not detected by LLM, defaulting to 'neutral'")
                    
                    # --- Hinglish auto-detection override ---
                    import re as _re_hinglish
                    _hinglish_words = [
                        r'\byaar\b', r'\bbhai\b', r'\barre\b', r'\baccha\b', r'\bkya\b',
                        r'\bhai\b', r'\bmain\b', r'\btum\b', r'\bnahi\b', r'\bnahin\b',
                        r'\baur\b', r'\blekin\b', r'\bpar\b', r'\bse\b', r'\bko\b',
                        r'\bwoh\b', r'\byeh\b', r'\bkaise\b', r'\bkahan\b', r'\bkab\b',
                        r'\bkyon\b', r'\bkyun\b', r'\bkyunki\b', r'\bagar\b', r'\btoh\b',
                        r'\bdekho\b', r'\bsuno\b', r'\bchalo\b', r'\bchal\b', r'\bhaan\b',
                        r'\bpagal\b', r'\bsamajh\b', r'\bbaat\b', r'\blog\b', r'\bdost\b',
                        r'\babhi\b', r'\bphir\b', r'\btheek\b', r'\bbilkul\b', r'\bsach\b',
                        r'\bdikhao\b', r'\bbolo\b', r'\bsuniye\b', r'\bdekhiye\b',
                        r'\bmujhe\b', r'\btujhe\b', r'\bhamara\b', r'\btumhara\b',
                        r'\bkuch\b', r'\bsab\b', r'\bkoi\b', r'\bbahut\b', r'\bzyada\b',
                    ]
                    _script_lower = detailed_story_script.lower()
                    _hinglish_hits = sum(1 for pat in _hinglish_words if _re_hinglish.search(pat, _script_lower))
                    if _hinglish_hits >= 3:
                        old_lang = parsed_script.get('language', 'English')
                        old_accent = parsed_script.get('accent', 'neutral')
                        parsed_script["language"] = "Hinglish"
                        parsed_script["accent"] = "native Hindi desi"
                        logging.info(f"Hinglish auto-detected in original script ({_hinglish_hits} Hinglish words found). "
                                     f"Overriding language '{old_lang}' → 'Hinglish', accent '{old_accent}' → 'native Hindi desi'")
                        st.info(f"🗣️ Hinglish detected in your script! Setting accent to 'native Hindi desi' (was: '{old_accent}')")
                
                logging.info(f"Language: {parsed_script['language']}, Accent: {parsed_script['accent']}")
                
                # --- POST-PROCESSING: Validate dialogue distribution ---
                scenes = parsed_script.get("scenes", [])
                dialogue_scenes = [s for s in scenes if s.get("type") == "dialogue" and s.get("dialogue_text")]
                total_scenes = len(scenes)
                dialogue_ratio = len(dialogue_scenes) / total_scenes if total_scenes > 0 else 0
                
                # Check for long stretches of silent scenes
                max_consecutive_silent = 0
                current_silent_streak = 0
                for s in scenes:
                    if s.get("type") != "dialogue" or not s.get("dialogue_text"):
                        current_silent_streak += 1
                        max_consecutive_silent = max(max_consecutive_silent, current_silent_streak)
                    else:
                        current_silent_streak = 0
                
                if dialogue_ratio > 0.3 and max_consecutive_silent >= 3:
                    st.warning(f"⚠️ Dialogue distribution issue detected: {len(dialogue_scenes)} dialogue scenes but {max_consecutive_silent} consecutive silent scenes. "
                               f"Consider adjusting your script or re-parsing for more natural dialogue flow.")
                    logging.warning(f"Dialogue distribution: {len(dialogue_scenes)}/{total_scenes} dialogue scenes, max silent streak: {max_consecutive_silent}")
                
                # --- POST-PROCESSING: Auto-reclassify visual_segment scenes that contain dialogue ---
                # LLMs sometimes set type=visual_segment even when the scene has dialogue_text.
                # This ensures those scenes are properly treated as dialogue scenes.
                reclassified_count = 0
                for s in scenes:
                    if s.get("type") != "dialogue" and s.get("dialogue_text") and s.get("dialogue_text").strip():
                        old_type = s.get("type")
                        s["type"] = "dialogue"
                        reclassified_count += 1
                        logging.warning(f"Auto-reclassified scene from '{old_type}' to 'dialogue' because it has dialogue_text: {s.get('dialogue_text', '')[:60]}...")
                if reclassified_count > 0:
                    st.info(f"🔄 Auto-reclassified {reclassified_count} scene(s) from visual_segment → dialogue (they had dialogue_text but wrong type).")
                    logging.info(f"Reclassified {reclassified_count} scenes from visual_segment to dialogue")
                
                # --- POST-PROCESSING: Force Dialogue in Every Scene (if checkbox enabled) ---
                if st.session_state.get("force_dialogue_all_scenes", False):
                    scenes = parsed_script.get("scenes", [])
                    script_language = parsed_script.get("language", "English")
                    
                    # Collect all dialogue text from all scenes (preserving original language)
                    all_dialogue_parts = []
                    seen_dialogue = set()
                    default_character = None
                    for s in scenes:
                        if s.get("dialogue_text") and s.get("dialogue_text", "").strip():
                            dt = s["dialogue_text"].strip()
                            # Deduplicate: Ollama often puts the SAME full dialogue in every scene
                            # Only add unique dialogue parts to avoid massive duplication
                            if dt not in seen_dialogue:
                                all_dialogue_parts.append(dt)
                                seen_dialogue.add(dt)
                            if not default_character and s.get("character"):
                                default_character = s["character"]
                    
                    if len(seen_dialogue) < len(scenes) and len(seen_dialogue) > 0:
                        logging.warning(f"Force Dialogue: Detected duplicate dialogue across scenes ({len(seen_dialogue)} unique out of {len(scenes)} scenes). Deduplicating before redistribution.")
                    
                    if all_dialogue_parts:
                        # Combine all dialogue, then split into roughly equal chunks by sentences
                        import re as _re
                        full_dialogue = " ".join(all_dialogue_parts)
                        
                        # Smart sentence splitting: handle Hindi/Urdu (।), standard punctuation, and ellipsis (…/...)
                        # This preserves the original language structure during splits
                        sentences = _re.split(r'(?<=[.!?…।\u0964])\s+|(?<=\.\.\.)\s+', full_dialogue)
                        sentences = [s.strip() for s in sentences if s.strip()]
                        
                        if len(sentences) == 0:
                            sentences = [full_dialogue]
                        
                        # If we have fewer sentences than scenes, try splitting on commas/semicolons as well
                        num_total_scenes = len(scenes)
                        if len(sentences) < num_total_scenes:
                            # Try finer-grained splitting at clause boundaries
                            finer_sentences = []
                            for sent in sentences:
                                clauses = _re.split(r'(?<=[,;])\s+', sent)
                                clauses = [c.strip() for c in clauses if c.strip() and len(c.strip()) > 3]
                                if clauses:
                                    finer_sentences.extend(clauses)
                                else:
                                    finer_sentences.append(sent)
                            sentences = finer_sentences
                            logging.info(f"Force Dialogue: Fine-split to {len(sentences)} dialogue chunks for {num_total_scenes} scenes")
                        
                        # Distribute sentences across all scenes as evenly as possible
                        scenes_needing_dialogue = num_total_scenes
                        
                        # Calculate chunks: at least 1 sentence per scene, extras go to earlier scenes
                        base_per_scene = max(1, len(sentences) // scenes_needing_dialogue)
                        remainder = max(0, len(sentences) - (base_per_scene * scenes_needing_dialogue))
                        
                        sentence_idx = 0
                        redistributed_count = 0
                        for scene_idx, s in enumerate(scenes):
                            # How many sentences for this scene
                            count = base_per_scene + (1 if scene_idx < remainder else 0)
                            chunk = sentences[sentence_idx:sentence_idx + count]
                            sentence_idx = min(sentence_idx + count, len(sentences))
                            
                            if chunk:
                                new_dialogue = " ".join(chunk)
                                old_dialogue = s.get("dialogue_text", "")
                                if not old_dialogue or not old_dialogue.strip():
                                    redistributed_count += 1
                                s["dialogue_text"] = new_dialogue
                                s["type"] = "dialogue"
                                if not s.get("character") or not s["character"].strip():
                                    s["character"] = default_character or "Character"
                            elif not s.get("dialogue_text") or not s["dialogue_text"].strip():
                                # No sentences left but scene has no dialogue — cycle back through original
                                cycle_idx = scene_idx % len(sentences)
                                s["dialogue_text"] = sentences[cycle_idx]
                                s["type"] = "dialogue"
                                if not s.get("character") or not s["character"].strip():
                                    s["character"] = default_character or "Character"
                                redistributed_count += 1
                        
                        # --- VALIDATION: Check all scenes now have dialogue and it's in correct language ---
                        empty_dialogue_scenes = []
                        for idx, s in enumerate(scenes):
                            dt = s.get("dialogue_text", "").strip()
                            if not dt:
                                empty_dialogue_scenes.append(idx + 1)
                        
                        if empty_dialogue_scenes:
                            st.warning(f"⚠️ Force Dialogue: Scenes {empty_dialogue_scenes} still have no dialogue after redistribution. The script may not have enough dialogue text to cover all {num_total_scenes} scenes.")
                            logging.warning(f"Force Dialogue: Scenes still empty after redistribution: {empty_dialogue_scenes}")
                        
                        st.success(
                            f"🗣️ Force Dialogue: Redistributed dialogue across all {num_total_scenes} scenes "
                            f"({redistributed_count} previously silent scenes now have dialogue). "
                            f"Language: {script_language}. All dialogue is from the original script only."
                        )
                        logging.info(f"Force Dialogue: {len(sentences)} sentences distributed across {num_total_scenes} scenes. {redistributed_count} scenes gained new dialogue. Language: {script_language}")
                    else:
                        st.warning("⚠️ Force Dialogue is ON but no dialogue text was found in the parsed script. Cannot redistribute.")
                        logging.warning("Force Dialogue enabled but no dialogue_text found in any scene.")
                
                # --- POST-PROCESSING: Check scene uniqueness ---
                visual_prompts = [s.get("visual_prompt", "") for s in scenes if s.get("visual_prompt")]
                duplicate_warnings = []
                for idx_a in range(len(visual_prompts)):
                    for idx_b in range(idx_a + 1, len(visual_prompts)):
                        # Simple overlap check: if prompts share >70% of words, flag as duplicate
                        words_a = set(visual_prompts[idx_a].lower().split())
                        words_b = set(visual_prompts[idx_b].lower().split())
                        if len(words_a) > 0 and len(words_b) > 0:
                            overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
                            if overlap > 0.70:
                                duplicate_warnings.append((idx_a + 1, idx_b + 1, overlap))
                                logging.warning(f"Scene {idx_a+1} and Scene {idx_b+1} have {overlap:.0%} word overlap - may generate similar visuals")
                
                if duplicate_warnings:
                    dup_msgs = [f"Scene {a} ↔ Scene {b} ({o:.0%} similar)" for a, b, o in duplicate_warnings]
                    st.warning(f"⚠️ Similar visual prompts detected - these scenes may generate repetitive visuals:\n" + "\n".join(dup_msgs))
                
                st.session_state.parsed_script = parsed_script
                
                # Generate video title from the story
                st.info("📝 Generating title for your video...")
                video_title = None
                if parser_choice == "Ollama (Local)":
                    video_title = generate_video_title_with_ollama(detailed_story_script, ollama_model, status_area)
                else:  # Gemini
                    video_title = generate_video_title_with_gemini(detailed_story_script, gemini_api_key, normalized_model, status_area)
                
                if video_title:
                    st.session_state.video_title = video_title
                    st.success(f"✅ Video title generated: **{video_title}**")
                else:
                    # Fallback to default if title generation fails
                    st.warning("Could not generate a title. Using default name.")
                    st.session_state.video_title = "final_story"
                
                if generate_char_images:
                    st.session_state.stage = "character_setup"
                else:
                    st.session_state.character_uploaders = {} # Ensure it exists but is empty
                    st.session_state.stage = "generation" # Skip character setup
                st.rerun()

if st.session_state.stage == "character_setup":
    st.header("🎭 Character Setup")
    parsed_script = st.session_state.parsed_script

    with st.expander("🤖 AI-Parsed Script Details", expanded=True):
        st.json(parsed_script)

    # Use a form to gather all character images at once before proceeding
    with st.form("character_setup_form"):
        all_characters_data = parsed_script.get("characters", [])
        character_uploaders = {}

        if not all_characters_data:
            st.warning("No characters were identified by the AI.")
        else:
            st.info("For each character, you can upload a reference image. If no image is provided, one will be generated from the description.")
            for char_data in all_characters_data:
                if isinstance(char_data, str):
                    char_name = char_data
                    char_description = "character"
                else:
                    char_name = char_data.get("name", "Unknown Character")
                    char_description = get_character_description(char_data)
                with st.container(border=True):
                    st.subheader(f"Character: {char_name}")
                    st.write(f"*{char_description}*")
                    character_uploaders[char_name] = st.file_uploader(f"Upload image for {char_name}", type=["png", "jpg", "jpeg"], key=f"uploader_{char_name}")

        submitted = st.form_submit_button("🎬 Confirm Characters & Generate Video")

        if submitted:
            st.session_state.character_uploaders = character_uploaders
            st.session_state.stage = "generation"
            st.rerun()

if st.session_state.stage == "generation":
    # This is the main generation block
    parsed_script = st.session_state.parsed_script
    APP_RUN_OUTPUT_DIR = st.session_state.app_run_output_dir
    character_image_paths = {}
    image_input_method = st.session_state.get("image_input_method", "Start with blank")
    
    # === RETRIEVE SINGLE UPLOADED IMAGE (already saved to disk during parse stage) ===
    single_image_filename = st.session_state.get("saved_single_image_filename", None)
    if single_image_filename:
        img_full_path = os.path.join(COMFYUI_REAL_INPUT_DIR, single_image_filename)
        if os.path.exists(img_full_path):
            st.header("🖼️ Reference Image")
            st.success(f"✅ Using saved reference image: {single_image_filename}")
            st.image(img_full_path, caption="Using this image for all scenes", width=400)
            logging.info(f"Using previously saved reference image: {img_full_path}")
        else:
            st.error(f"❌ Reference image file not found at {img_full_path}")
            logging.error(f"Reference image not found: {img_full_path}")
            single_image_filename = None

    # --- 1. Process Character Images (Upload or Generate) ---
    if generate_char_images:
        st.header("🎭 Processing Characters...")
        all_characters_data = parsed_script.get("characters", [])
        character_uploaders = st.session_state.get("character_uploaders", {})

        if all_characters_data:
            try:
                with open(WORKFLOW_FILE, 'r', encoding='utf-8') as f:
                    char_gen_workflow_template = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                st.error(f"Failed to load workflow file for character generation: {e}")
                st.stop()

            for char_data in all_characters_data:
                if isinstance(char_data, str):
                    char_name = char_data
                else:
                    char_name = char_data.get("name", "Unknown Character")
                uploaded_image = character_uploaders.get(char_name)

                if uploaded_image:
                    st.subheader(f"Using uploaded image for {char_name}")
                    safe_char_name = "".join(c for c in char_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
                    image_filename = f"{safe_char_name}.png"
                    image_path = os.path.join(COMFYUI_REAL_INPUT_DIR, image_filename)
                    with open(image_path, "wb") as f:
                        f.write(uploaded_image.getbuffer())
                    logging.info(f"Saved uploaded image for '{char_name}' to '{image_path}'")
                    st.image(image_path, width=256)
                    character_image_paths[char_name] = image_filename
                else:
                    st.subheader(f"Generating image for {char_name}")
                    char_description = get_character_description(char_data) if isinstance(char_data, dict) else "character"
                    char_prompt = f"A cinematic, high-quality, full-body shot of {char_name}, {char_description}, in a neutral setting."
                    char_image_path = generate_character(char_prompt, char_name, COMFYUI_REAL_INPUT_DIR, APP_RUN_OUTPUT_DIR, node_id_mapping, st.session_state.run_id, char_gen_workflow_template)
                    if char_image_path:
                        character_image_paths[char_name] = os.path.basename(char_image_path)
                    else:
                        st.error(f"Failed to generate image for {char_name}.")

    # --- 2. Scene Generation ---
    st.header("🎬 Generating Scenes...")
    # ... (The rest of the generation logic is moved here) ...
    generated_videos = []
    last_frame_path = None
    client_id = str(uuid.uuid4())

    with open(WORKFLOW_FILE, 'r', encoding='utf-8') as f:
        base_workflow = json.load(f)

    base_prompt = parsed_script.get("global_visual_description", "")
    character_descriptions = " ".join([
        f"{c.get('name', 'Character')} is {get_character_description(c)}." if isinstance(c, dict)
        else f"{c} is a character."
        for c in parsed_script.get("characters", [])
    ])
    if character_descriptions:
        base_prompt += f" The characters are: {character_descriptions}"

    # --- Extract language/accent for consistent propagation to ALL scenes ---
    script_language = parsed_script.get("language", "English")
    script_accent = parsed_script.get("accent", "neutral")
    language_directive = f"All speech and dialogue must be in {script_language} language with {script_accent} accent. Maintain consistent voice style throughout."
    logging.info(f"Language/accent for all scenes: {script_language} / {script_accent}")

    visual_events = [event for event in parsed_script.get("scenes", []) if event.get("type") in ["visual_segment", "dialogue", "montage"]]
    # LIMIT: Only generate num_scenes, not all visual_events
    scenes_to_generate = visual_events[:num_scenes]
    
    # DEBUG: Log what was parsed
    logging.info(f"Total scenes parsed: {len(parsed_script.get('scenes', []))}")
    logging.info(f"Visual events (visual_segment/dialogue/montage): {len(visual_events)}")
    for idx, evt in enumerate(visual_events):
        evt_type = evt.get("type", "unknown")
        evt_char = evt.get("character", "N/A")
        evt_dialogue = evt.get("dialogue_text", "")[:80] if evt.get("dialogue_text") else "N/A"
        logging.info(f"  Scene {idx}: type={evt_type}, character={evt_char}, dialogue={evt_dialogue}")
    
    st.warning(f"⚠️ Total events: {len(visual_events)} | Generating: {len(scenes_to_generate)} scenes (based on {total_video_duration}s duration)")

    for i, event in enumerate(scenes_to_generate):
        event_type = event.get("type")
        st.subheader(f"Generating Video Segment {i+1}/{len(scenes_to_generate)} (Type: {event_type})")
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_message_written = False # Flag to prevent overwriting status messages

        workflow = copy.deepcopy(base_workflow)

        # --- Apply Global Negative Prompt ---
        if node_id_mapping.get("negative_prompt") and st.session_state.get("negative_prompt_text"):
            neg_prompt_is_valid, neg_prompt_sanitized, neg_warnings = validate_and_sanitize_prompt(st.session_state.negative_prompt_text)
            if neg_prompt_is_valid:
                workflow[node_id_mapping["negative_prompt"]]["inputs"]["text"] = neg_prompt_sanitized
                logging.info(f"Applying global negative prompt to node {node_id_mapping['negative_prompt']}")
            else:
                st.warning("⚠️ Negative prompt validation failed. Skipping negative prompt for this scene.")
                if neg_warnings:
                    for warning in neg_warnings:
                        logging.warning(f"Negative prompt issue: {warning}")

        workflow[node_id_mapping["seed"]]["inputs"]["seed"] = random.randint(0, 1_000_000_000)

        run_id_prefix = st.session_state.run_id.split('-')[0]
        filename_prefix = f"{run_id_prefix}_scene_{i+1}"
        workflow[node_id_mapping["save"]]["inputs"]["filename_prefix"] = filename_prefix

        prompt_parts = [base_prompt]

        # --- LANGUAGE/ACCENT CONSISTENCY: Only add speech directive to dialogue scenes ---
        # For non-dialogue scenes, language directive mentioning "speech" can confuse
        # the LTX audio model into generating gibberish voice audio.
        # Check BOTH type field AND actual dialogue_text presence as a safety net
        has_dialogue_text = bool(event.get("dialogue_text") and event.get("dialogue_text", "").strip())
        is_dialogue_scene = (event_type == "dialogue" and has_dialogue_text) or has_dialogue_text
        if is_dialogue_scene:
            prompt_parts.append(language_directive)

        # --- SCENE UNIQUENESS: Add scene context to help differentiate ---
        prompt_parts.append(f"[Scene {i+1} of {len(scenes_to_generate)}]")

        event_description = ""
        if event_type == "montage":
            montage_descriptions = [s.get("description", "") for s in event.get("segments", [])]
            event_description = "The scene is a montage showing: " + ", ".join(filter(None, montage_descriptions))
        else:
            event_description = event.get("visual_prompt") or event.get("description")

        # --- CRITICAL: Strip silence markers from visual_prompt when scene has dialogue ---
        # The LLM parser/enhancement pass sometimes bakes "No speech. No dialogue..." into the
        # visual_prompt field. When the scene actually has dialogue, these silence instructions
        # OVERRIDE the dialogue and the model produces a silent scene. Strip them out.
        if is_dialogue_scene and event_description:
            import re as _re_clean
            # Remove all variations of silence markers from the visual_prompt
            silence_patterns = [
                r'No speech\.?\s*',
                r'No dialogue\.?\s*',
                r'No voiceover\.?\s*',
                r'No talking\.?\s*',
                r'No human voice\.?\s*',
                r'Silent scene[^.]*\.?\s*',
                r'No speech,?\s*no dialogue,?\s*no voiceover,?\s*no talking[^.]*\.?\s*',
                r'only soft ambient environmental sounds\.?\s*',
            ]
            cleaned_description = event_description
            for pattern in silence_patterns:
                cleaned_description = _re_clean.sub(pattern, '', cleaned_description, flags=_re_clean.IGNORECASE)
            # Clean up extra whitespace left behind
            cleaned_description = _re_clean.sub(r'\s{2,}', ' ', cleaned_description).strip()
            cleaned_description = cleaned_description.rstrip('.')
            if cleaned_description != event_description:
                logging.info(f"Scene {i+1}: Stripped silence markers from visual_prompt (scene has dialogue)")
                logging.info(f"  Before: {event_description[:120]}...")
                logging.info(f"  After:  {cleaned_description[:120]}...")
            event_description = cleaned_description

        if event_description:
            prompt_parts.append(f"The current action is: {event_description}.")

        # --- DIALOGUE INTEGRATED INTO PROMPT ---
        # LTX 2.0 has built-in audio generation from the text prompt.
        # For dialogue scenes: include speech text so audio model generates proper speech.
        # For visual-only scenes: explicitly mark as SILENT so audio model doesn't
        # try to vocalize the visual description (which produces gibberish audio).
        # IMPORTANT: Check actual dialogue_text presence, not just type field,
        # because LLMs sometimes classify dialogue scenes as visual_segment.
        dialogue_expression_parts = []
        if is_dialogue_scene:
            char_name = event.get("character")
            dialogue = event.get("dialogue_text")
            logging.info(f"Scene {i+1}: Event has DIALOGUE (type={event_type}). character={char_name}, dialogue={dialogue[:80] if dialogue else 'EMPTY'}")
            if char_name and dialogue:
                # Format dialogue with proper structure for i2v lip sync
                # Include header and expression markers as shown in working ComfyUI tests
                # Include language/accent directive for consistent voice
                dialogue_part = (
                    f"{char_name} speaks in {script_language} with {script_accent} accent, natural expressive voice with proper lip sync:\n\n"
                    f"Dialogue ({script_language}):\n"
                    f"\"{dialogue}\"\n\n"
                    f"Voice must be consistent {script_language} with {script_accent} accent throughout. "
                    f"Natural facial expressions with mouth movement synchronized to dialogue. "
                    f"Ultra realistic facial animation, subtle body movement, cinematic natural lighting."
                )
                prompt_parts.append(dialogue_part)
                logging.info(f"Scene {i+1}: ✅ Dialogue ADDED to prompt for '{char_name}' with lip sync markers and {script_language}/{script_accent}: {dialogue[:80]}...")
            else:
                # Dialogue scene but missing text — treat as silent
                prompt_parts.append(
                    "No speech. No dialogue. No voiceover. No talking. "
                    "Silent scene with only soft ambient environmental sounds. No human voice."
                )
                logging.warning(f"Scene {i+1}: ⚠️ Dialogue scene missing character or text — marking as SILENT. char_name={char_name}, dialogue={dialogue}")
        else:
            # --- CRITICAL: Non-dialogue scenes must be explicitly marked silent ---
            # Without this, the LTX audio model tries to vocalize the visual description
            # text, producing gibberish/random speech audio.
            # Only mark as silent if there's genuinely no dialogue_text at all
            prompt_parts.append(
                "No speech. No dialogue. No voiceover. No talking. "
                "Silent scene with only soft ambient environmental sounds. No human voice."
            )
            logging.info(f"Scene {i+1}: Event type is {event_type}, no dialogue_text found — marked as SILENT audio")

        final_prompt = " ".join(filter(None, prompt_parts))
        
        # Validate and sanitize the prompt before injecting into workflow
        is_valid, sanitized_prompt, prompt_warnings = validate_and_sanitize_prompt(final_prompt)
        
        logging.info(f"Scene {i+1}: BEFORE sanitization: {final_prompt[:150]}...")
        logging.info(f"Scene {i+1}: AFTER sanitization: {sanitized_prompt[:150]}...")
        
        if prompt_warnings:
            for warning in prompt_warnings:
                status_text.warning(warning)
                logging.warning(f"Prompt validation warning: {warning}")
        
        if not is_valid:
            st.error(f"❌ Prompt validation failed. Cannot proceed with generation.")
            for warning in prompt_warnings:
                st.error(warning)
            break
        
        # DEBUG: Log node mapping for this scene
        prompt_node_id = node_id_mapping.get("prompt")
        logging.info(f"Scene {i+1}: Injecting into node ID: {prompt_node_id}")
        
        workflow[node_id_mapping["prompt"]]["inputs"]["text"] = sanitized_prompt
        
        # DEBUG: Verify injection was successful
        actual_text = workflow[node_id_mapping["prompt"]]["inputs"]["text"]
        logging.info(f"Scene {i+1}: VERIFIED in workflow - text starts with: {actual_text[:100]}...")
        if "Dialogue:" in actual_text or "lip sync" in actual_text:
            logging.info(f"Scene {i+1}: ✅ Dialogue confirmed present in workflow prompt")
        else:
            logging.warning(f"Scene {i+1}: ⚠️ Dialogue may not be in workflow prompt")
        
        with st.expander("Full Prompt for this segment"):
            st.write(sanitized_prompt)
            if len(sanitized_prompt) > 1500:
                st.info(f"📏 Prompt length: {len(sanitized_prompt)} characters (very long prompts may have unpredictable results)")

        frame_rate = 25  # Default for LTX workflow
        try:
            # Frame rate lives as a parameter in the conditioning node (LTXVConditioning)
            fr_node = node_id_mapping.get("frame_rate")
            if fr_node and fr_node in base_workflow:
                fr_val = base_workflow[fr_node]["inputs"].get("frame_rate") or base_workflow[fr_node]["inputs"].get("value")
                if fr_val:
                    frame_rate = int(fr_val)
        except (KeyError, TypeError, ValueError):
            logging.warning(f"Could not read frame rate from workflow. Using default {frame_rate}fps.")
        frame_count = int(scene_duration * frame_rate)
        workflow[node_id_mapping["frame_count"]]["inputs"]["value"] = frame_count
        st.info(f"Segment duration set to {scene_duration} seconds ({frame_count} frames @ {frame_rate}fps). Total video duration: {total_video_duration}s ({num_scenes} scenes).")
        
        # --- IMAGE SELECTION FOR i2v MODEL ---
        # For i2v (Image-to-Video), the image is the PRIMARY input, not optional.
        # Priority: Single ref image > Last frame > Character image > Blank
        image_to_use = None
        
        # 1. PRIORITY 1: Use single uploaded reference image if available (applies to all scenes)
        if single_image_filename:
            image_to_use = single_image_filename
            status_text.info(f"📸 Using reference image: {single_image_filename}")
            status_message_written = True
            logging.info(f"Scene {i+1}: Using single reference image - {single_image_filename}")
        
        # 2. PRIORITY 2: Use last frame for continuity (if no single image)
        elif last_frame_path:
            image_to_use = os.path.basename(last_frame_path)
            status_text.info(f"📸 Using last frame for continuity: {image_to_use}")
            status_message_written = True
            logging.info(f"Scene {i+1}: Using last frame for continuity - {image_to_use}")
        
        # 3. PRIORITY 3: Use character image if mentioned in this scene
        else:
            scene_description = event.get("visual_prompt") or event.get("description", "")
            mentioned_characters = [name for name in character_image_paths.keys() if name.lower() in scene_description.lower()]
            
            if len(mentioned_characters) == 1:
                image_to_use = character_image_paths[mentioned_characters[0]]
                status_text.info(f"📸 Using character image: {image_to_use}")
                status_message_written = True
                logging.info(f"Scene {i+1}: Using character image for '{mentioned_characters[0]}' - {image_to_use}")
            elif is_dialogue_scene or event_type == "dialogue":
                char_name = event.get("character")
                if char_name and char_name in character_image_paths:
                    image_to_use = character_image_paths[char_name]
                    status_text.info(f"📸 Using character image (speaking): {image_to_use}")
                    status_message_written = True
                    logging.info(f"Scene {i+1}: Using character image for speaker '{char_name}' - {image_to_use}")
        
        # 4. PRIORITY 4: Fallback to blank image
        if not image_to_use:
            image_to_use = BLANK_IMAGE_FILENAME
            if not status_message_written:
                status_text.info(f"📸 Starting with blank image")
            logging.info(f"Scene {i+1}: Using blank image (no reference available)")
        
        # === APPLY IMAGE TO WORKFLOW ===
        # For i2v, the image input is CRITICAL - it's the main content input
        if node_id_mapping.get("image"):
            workflow[node_id_mapping["image"]]["inputs"]["image"] = image_to_use
            logging.info(f"i2v input image for scene {i+1}: {image_to_use}")
        else:
            st.warning("⚠️ Image node not mapped! i2v requires an image input.")

        generated_video_path = queue_prompt_and_wait(workflow, client_id, progress_bar, status_text, APP_RUN_OUTPUT_DIR, node_id_mapping)

        if generated_video_path and os.path.exists(generated_video_path):
            st.success(f"Segment {i+1} generated!")
            st.video(generated_video_path)
            generated_videos.append(generated_video_path)
            progress_bar.progress(100)

            if i < len(scenes_to_generate) - 1:
                next_frame_filename = f"scene_{i+1}_last_frame.png"
                next_frame_path = os.path.join(COMFYUI_REAL_INPUT_DIR, next_frame_filename)
                logging.info(f"Extracting continuity frame for next scene: {next_frame_path}")
                if extract_last_frame(generated_video_path, next_frame_path):
                    last_frame_path = next_frame_path
                    st.success(f"✅ Continuity frame extracted for scene {i+2}")
                else:
                    st.warning("Failed to extract continuity frame. Continuity may be affected.")
                    last_frame_path = None
            else:
                st.info(f"Final scene {i+1} - no continuity frame needed for next scene")
                last_frame_path = None
        else:
            st.error(f"Segment {i+1} failed to generate. Stopping.")
            break

        if i < len(scenes_to_generate) - 1:
            if purge_vram:
                purge_comfyui_vram(status_text)
            if scene_cooldown > 0:
                st.info(f"Cooldown: Waiting for {scene_cooldown} seconds...")
                time.sleep(scene_cooldown)

    # --- Cleanup Empty Directories (runs even if generation fails) ---
    st.info("🗑️ Cleaning up empty folders from failed runs...")
    empty_dirs_deleted = cleanup_empty_directories(APP_OUTPUT_DIR)
    if empty_dirs_deleted > 0:
        st.info(f"✅ Removed {empty_dirs_deleted} empty directory folders.")
        logging.info(f"Deleted {empty_dirs_deleted} empty directories from output")

    # --- 3. Final Video Stitching ---
    if generated_videos:
        st.header("Final Touches")
        
        # Get the generated video title or use default
        video_title = st.session_state.get("video_title", "final_story")
        # Sanitize the title for use as filename
        sanitized_title = sanitize_filename(video_title)
        final_video_filename = f"{sanitized_title}.mp4"
        final_video_path = os.path.join(APP_RUN_OUTPUT_DIR, final_video_filename)
        
        with st.spinner("Stitching video..."):
            stitch_videos(generated_videos, final_video_path)
        
        # --- 3b. Post-Processing Upscale ---
        if os.path.exists(final_video_path) and upscale_resolution != "720p (1280×720) - No upscale":
            # Parse target resolution from selection
            resolution_map = {
                "1080p Full HD (1920×1080)": (1920, 1080),
                "2K (2560×1440)": (2560, 1440),
                "4K (3840×2160)": (3840, 2160),
            }
            target_w, target_h = resolution_map.get(upscale_resolution, (1920, 1080))
            
            upscaled_filename = f"{sanitized_title}_upscaled_{target_w}x{target_h}.mp4"
            upscaled_path = os.path.join(APP_RUN_OUTPUT_DIR, upscaled_filename)
            
            with st.spinner(f"Upscaling to {target_w}x{target_h} (this may take a moment)..."):
                upscale_success = upscale_video_ffmpeg(final_video_path, upscaled_path, target_w, target_h, crf=upscale_quality)
            
            if upscale_success and os.path.exists(upscaled_path):
                # Replace the 720p version with the upscaled one
                original_size = os.path.getsize(final_video_path) / (1024 * 1024)
                upscaled_size = os.path.getsize(upscaled_path) / (1024 * 1024)
                
                # Remove the 720p stitched version to save space
                os.remove(final_video_path)
                # Rename upscaled to the final name
                final_video_path_hd = os.path.join(APP_RUN_OUTPUT_DIR, final_video_filename)
                os.rename(upscaled_path, final_video_path_hd)
                final_video_path = final_video_path_hd
                
                st.success(f"✅ Upscaled to {target_w}x{target_h}! ({original_size:.1f}MB → {upscaled_size:.1f}MB)")
            else:
                st.warning("⚠️ Upscale failed. Keeping original 720p video.")
        
        # Clean up scene videos after successful final video creation
        if os.path.exists(final_video_path):
            st.info("🧹 Cleaning up scene videos to save disk space...")
            success, space_freed = cleanup_scene_videos(generated_videos, APP_RUN_OUTPUT_DIR)
            if success and space_freed > 0:
                st.success(f"✅ Cleanup complete! Freed {space_freed:.2f} MB of disk space. Only final video retained.")
                logging.info(f"Successfully freed {space_freed:.2f} MB by removing scene videos")
            elif success:
                st.info("Scene video cleanup completed.")
        
        st.balloons()
        st.subheader("🎉 Final Masterpiece 🎉")
        if os.path.exists(final_video_path):
            st.video(final_video_path)
            st.info(f"✨ Your video is ready!\n\n🎬 **Title:** {video_title}\n\n📁 **Location:** `{final_video_path}`")
        else:
            st.warning("Final video not created.")
    
    # Reset stage to allow for a new run
    st.session_state.stage = "script_input"

st.markdown("---")
st.info("Videos are saved in the `output` directory.")
