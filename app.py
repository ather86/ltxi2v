import streamlit as st
import requests
import json
import time
import os
import uuid
import websocket
import subprocess
import cv2
import shutil
import random
import logging

# --- Dependency Check ---
try:
    import ollama
except ImportError:
    st.error("The 'ollama' library is not installed. Please install it by running: pip install ollama")
    st.stop()

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
WORKFLOW_FILE = "workfllows/video_ltx2_t2v.json"
COMFYUI_REAL_OUTPUT_DIR = "D:\\StabilityMatrix-win-x64\\Data\\Packages\\ComfyUI\\output" # The actual output directory for ComfyUI
BASE_COMFYUI_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
BASE_COMFYUI_INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
FINAL_VIDEO_FILE = "final_story.mp4"

# Create base directories if they don't exist
os.makedirs(BASE_COMFYUI_OUTPUT_DIR, exist_ok=True)
os.makedirs(BASE_COMFYUI_INPUT_DIR, exist_ok=True)


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


def _wait_for_file_to_be_written(file_path, timeout_sec=120):
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
                    print(" " * 70, end="\r") # Clean up waiting line
                    logging.info(f"File '{os.path.basename(file_path)}' is complete.")
                    return True

            elapsed = int(time.time() - start_time)
            print(f"⏳ Waiting for output file... {elapsed}s", end="\r")
            time.sleep(1)
        except FileNotFoundError:
            # This can happen in a race condition, just continue waiting
            time.sleep(1)
            continue
        except Exception as e:
            logging.error(f"An error occurred while waiting for file: {e}", exc_info=True)
            time.sleep(1)

    print(" " * 70, end="\r") # Clean up waiting line
    logging.error(f"Timeout: File '{os.path.basename(file_path)}' not found or not complete after {timeout_sec}s.")
    return None


def parse_detailed_script_with_ollama(detailed_script, model, status_text):
    """
    Uses Ollama to parse a detailed narrative script into a structured JSON format
    for sequential video generation.
    """
    status_text.info("🤖 Contacting AI Script Parser (Ollama)...")
    logging.info(f"Parsing detailed script with Ollama using model: {model}")

    # The system prompt instructs Ollama to output a specific JSON structure
    system_prompt = (
        "You are an expert script parser. Your task is to take a detailed video script "
        "and extract its components into a structured JSON format. The script will contain "
        "a global visual description, character definitions, and a sequence of events "
        "including visual cues, character actions, and dialogue. "
        "Each event in the 'scenes' array should have a 'type' (e.g., 'visual_segment', 'dialogue', 'camera_move', 'montage'). "
        "For 'dialogue' events, include 'character' and 'dialogue_text'. "
        "For 'visual_segment' and 'dialogue' events, provide a 'visual_prompt' key that summarizes the visual aspect of that moment. "
        "For dialogue, 'visual_prompt_modifier' should describe the character's action while speaking. "
        "For 'montage' events, include a 'segments' array, each with a 'description'. "
        "Do not include any other text or explanation outside the JSON."
        "\n\nOutput Format Example:"
        """
        {
          "global_visual_description": "A cinematic live-action sunset beach scene with warm golden light...",
          "characters": [
            {"name": "ATHER", "description": "Indian male, early 40s, short black hair, clean shaved, casual white linen shirt, calm thoughtful presence, medium-light brown skin."},
            {"name": "HARSHIT", "description": "Indian male, early 40s, short hair with trimmed beard, denim shirt, energetic friendly personality, medium-light brown skin."}
          ],
          "scenes": [
            {
              "type": "visual_segment",
              "description": "The camera opens with a slow wide establishing shot of the ocean at sunset, waves rolling gently toward the shore.",
              "visual_prompt": "Wide establishing shot of ocean at sunset, gentle waves."
            },
            {
              "type": "camera_move",
              "description": "The camera slowly pans to reveal ATHER, HARSHIT, SANDY, and SATBIR walking barefoot along the waterline together."
            },
            {
              "type": "dialogue",
              "character": "ATHER",
              "dialogue_text": "You know… days like this remind me how important it is to slow down.",
              "visual_prompt": "ATHER looks toward the horizon and speaks first, calmly and warmly.",
              "visual_prompt_modifier": "ATHER looks toward the horizon and speaks first, calmly and warmly."
            },
            {
              "type": "montage",
              "segments": [
                {"description": "The four friends splashing water playfully."},
                {"description": "ATHER taking a group selfie as everyone laughs."}
              ]
            }
          ]
        }
        """
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': detailed_script}
            ],
            options={'temperature': 0.7} # Lower temperature for more structured output
        )
        
        content = response['message']['content']
        logging.info(f"Ollama response received (partial): {content[:500]}...")

        # Clean up common LLM artifacts like smart quotes and markdown code blocks.
        cleaned_content = content.replace("“", '"').replace("”", '"')
        if cleaned_content.strip().startswith("```json"):
            cleaned_content = cleaned_content.strip()[7:-3].strip()
        elif cleaned_content.strip().startswith("```"):
            cleaned_content = cleaned_content.strip()[3:-3].strip()

        # Attempt to find and parse the JSON block
        json_start = cleaned_content.find('{')
        json_end = cleaned_content.rfind('}') + 1
        if json_start == -1 or json_end == -1:
            raise ValueError("No JSON object found in Ollama's response.")
        
        json_str = cleaned_content[json_start:json_end]
        
        parsed_data = json.loads(json_str)
        
        # Basic validation of the parsed structure
        if not all(k in parsed_data for k in ["global_visual_description", "characters", "scenes"]):
            raise ValueError("Parsed JSON is missing required top-level keys (global_visual_description, characters, scenes).")
        if not isinstance(parsed_data["characters"], list) or not all(isinstance(c, dict) and "name" in c for c in parsed_data["characters"]):
            raise ValueError("Characters list is not correctly formatted.")
        if not isinstance(parsed_data["scenes"], list) or not all(isinstance(s, dict) and "type" in s for s in parsed_data["scenes"]):
            raise ValueError("Scenes list is not correctly formatted.")

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


def find_latest_file_by_prefix(directory, prefix):
    """Finds the most recently modified file in a directory or its subdirectories with a given prefix."""
    logging.info(f"Recursively searching for files with prefix '{prefix}' in '{directory}'")
    found_files = []
    try:
        for root, _, files in os.walk(directory):
            for f in files:
                if f.startswith(prefix):
                    found_files.append(os.path.join(root, f))

        if not found_files:
            logging.warning(f"No files found with prefix: {prefix}")
            return None

        latest_file = max(found_files, key=os.path.getmtime)
        logging.info(f"Found latest file: {os.path.basename(latest_file)} in {os.path.dirname(latest_file)}")
        return latest_file
    except FileNotFoundError:
        logging.error(f"Search directory not found: {directory}")
        return None
    except Exception as e:
        logging.error(f"Error finding file by prefix: {e}", exc_info=True)
        return None


def queue_prompt_and_wait(prompt_workflow, client_id, progress_bar, status_text, comfyui_output_dir, node_id_mapping, filename_prefix):
    if not check_comfyui_health(status_text):
        return None

    try:
        req = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": prompt_workflow, "client_id": client_id})
        req.raise_for_status()
        prompt_id = req.json()['prompt_id']
        logging.info(f"✅ Prompt queued with ID: {prompt_id}")

        ws = websocket.WebSocket()
        ws.connect(COMFYUI_WS_URL + f"?clientId={client_id}")
        logging.info("🔌 WebSocket connected.")
        ws.settimeout(10.0) # Increased timeout

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
                    st.error(f"❌ ComfyUI Execution Error: {error_data.get('exception_message', 'Unknown')}. Full error: {json.dumps(error_data, indent=2)}")
                    logging.error(f"ComfyUI Execution Error for prompt {prompt_id}: {json.dumps(error_data, indent=2)}")
                    ws.close()
                    return None
                elif message['type'] == 'executed' and message['data']['prompt_id'] == prompt_id:
                    logging.info("✅ Execution finished on server. Locating output file...")
                    ws.close()

                    # --- ROBUST FILE FINDING LOGIC ---
                    # Poll for the file to appear, as there can be a delay between ComfyUI
                    # finishing and the file being visible on the filesystem.
                    found_path = None
                    search_start_time = time.time()
                    # Wait up to 60 seconds for the file to appear in the directory listing.
                    search_timeout = 60
                    logging.info(f"Polling for output file with prefix '{filename_prefix}' for up to {search_timeout}s...")
                    while time.time() - search_start_time < search_timeout:
                        found_path = find_latest_file_by_prefix(COMFYUI_REAL_OUTPUT_DIR, filename_prefix)
                        if found_path:
                            logging.info(f"Found candidate file: {os.path.basename(found_path)}")
                            break
                        time.sleep(2) # Poll every 2 seconds
                        status_text.info(f"Searching for output file... ({int(time.time() - search_start_time)}s)")


                    if found_path and _wait_for_file_to_be_written(found_path):
                        filename = os.path.basename(found_path)
                        destination_path = os.path.join(comfyui_output_dir, filename)
                        logging.info(f"Copying '{found_path}' to '{destination_path}'")
                        shutil.copy2(found_path, destination_path)
                        logging.info(f"✅ Successfully copied '{filename}'.")
                        return destination_path
                    else:
                        st.error(f"Execution finished, but could not find the output file with prefix '{filename_prefix}' in '{COMFYUI_REAL_OUTPUT_DIR}'. Please check ComfyUI logs.")
                        logging.error(f"Failed to find output file with prefix '{filename_prefix}'.")
                        return None
            except websocket.WebSocketTimeoutException:
                logging.warning("WebSocket timeout. Checking ComfyUI health...")
                if not check_comfyui_health(status_text):
                    st.error("Lost connection to ComfyUI during generation.")
                    ws.close()
                    return None
                logging.info("ComfyUI is alive, continuing to wait.")
                continue
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.error(f"Error in queue_prompt_and_wait: {e}", exc_info=True)
        return None
    return None



def extract_last_frame(video_path, output_image_path):
    # ... (This function remains the same)
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error: Could not open video {video_path}")
            return False
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0: return False
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        if not ret: return False
        cv2.imwrite(output_image_path, frame)
        cap.release()
        st.image(output_image_path, caption=f"Last frame of {os.path.basename(video_path)}")
        return True
    except Exception as e:
        st.error(f"OpenCV Error extracting frame: {e}")
        return False

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
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        st.success(f"Final video saved to {output_file}")
    except Exception as e:
        st.error(f"FFmpeg Error: {e}")
    finally:
        if os.path.exists(list_path): os.remove(list_path)

def generate_character(prompt, character_name, comfyui_input_dir, comfyui_output_dir, nodes, run_id):
    """Generates a character image using a modified T2V workflow."""
    st.info(f"Generating image for {character_name}...")
    client_id = str(uuid.uuid4())
    
    with open(WORKFLOW_FILE, 'r') as f:
        workflow = json.load(f)

    # Sanitize character name for filename
    safe_char_name = "".join(c for c in character_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
    
    # Create a unique prefix for this generation
    filename_prefix = f"{run_id.split('-')[0]}_character_{safe_char_name}"

    # Modify workflow for image generation
    workflow[nodes["prompt"]]["inputs"]["text"] = prompt
    workflow[nodes["seed"]]["inputs"]["seed"] = random.randint(0, 1_000_000_000)
    workflow[nodes["save"]]["inputs"]["filename_prefix"] = filename_prefix
    
    logging.info(f"Character generation workflow: Prompt='{prompt}', Seed='{workflow[nodes['seed']]['inputs']['seed']}', Prefix='{filename_prefix}'")
    # This is the key change: set the length to 1 to generate a single image
    workflow["92:62"]["inputs"]["value"] = 1

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Queue and wait for the character image
    generated_video_path = queue_prompt_and_wait(workflow, client_id, progress_bar, status_text, comfyui_output_dir, nodes, filename_prefix)

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
st.set_page_config(layout="wide")
st.title("🎬 Agentic Video Studio")

# --- Session State Initialization ---
if 'run_id' not in st.session_state:
    st.session_state.run_id = str(uuid.uuid4())
    st.session_state.comfyui_output_dir = os.path.join(BASE_COMFYUI_OUTPUT_DIR, st.session_state.run_id)
    st.session_state.comfyui_input_dir = os.path.join(BASE_COMFYUI_INPUT_DIR, st.session_state.run_id)
    os.makedirs(st.session_state.comfyui_output_dir, exist_ok=True)
    os.makedirs(st.session_state.comfyui_input_dir, exist_ok=True)


with st.sidebar:
    st.header("⚙️ Configuration")
    scene_duration = st.slider("Scene Duration (seconds)", min_value=1, max_value=20, value=4, step=1, help="Set the duration for each generated video segment. Shorter durations require less VRAM and can prevent 'out of memory' errors. 4-5 seconds is a safe start.")
    scene_cooldown = st.slider("Cooldown Between Scenes (seconds)", min_value=0, max_value=60, value=10, step=1, help="Pause between generating each scene to allow GPU VRAM to be cleared. Increase this if you get 'out of memory' errors.")
    # ... (Workflow mapping remains the same)
    if os.path.exists(WORKFLOW_FILE):
        with open(WORKFLOW_FILE, 'r') as f:
            workflow_data = json.load(f)
        node_titles = {node_id: node.get("_meta", {}).get("title", f"Untitled {node_id}") for node_id, node in workflow_data.items()}
        node_titles_list = sorted(list(set(node_titles.values()))) # Use a sorted, unique list of titles

        # --- Define Default Titles ---
        DEFAULT_TITLES = {
            "prompt": "Positive Prompt",
            "seed": "RandomNoise",
            "image": "EmptyImage",
            "save": "Save Video"
        }

        # --- Get Indices for Defaults ---
        def get_index(title, options):
            try:
                return options.index(title)
            except (ValueError, KeyError):
                return 0 # Default to first item if not found

        prompt_index = get_index(DEFAULT_TITLES["prompt"], node_titles_list)
        seed_index = get_index(DEFAULT_TITLES["seed"], node_titles_list)
        image_index = get_index(DEFAULT_TITLES["image"], node_titles_list)
        save_index = get_index(DEFAULT_TITLES["save"], node_titles_list)

        # --- Create Select Boxes with Defaults ---
        prompt_node_title = st.selectbox("Prompt Node:", node_titles_list, index=prompt_index)
        seed_node_title = st.selectbox("Seed Node:", node_titles_list, index=seed_index)
        image_input_node_title = st.selectbox("Image Input Node:", node_titles_list, index=image_index, help="Select the node that takes an image as input for continuity. If your workflow uses EmptyImage, select that.")
        save_node_title = st.selectbox("Save Node:", node_titles_list, index=save_index)
        
        # New selections for dialogue and ambient audio, if they exist in the workflow
        # These will need corresponding nodes in your ComfyUI workflow to be functional.
        optional_nodes_list = ["None"] + node_titles_list
        dialogue_prompt_node_title = st.selectbox("Dialogue Prompt Node (Optional):", optional_nodes_list, help="Select a CLIPTextEncode-like node specifically for dialogue audio. Requires workflow modification.")
        ambient_audio_prompt_node_title = st.selectbox("Ambient Audio Prompt Node (Optional):", optional_nodes_list, help="Select a CLIPTextEncode-like node specifically for ambient audio. Requires workflow modification.")

        # Store node IDs in a dictionary for easier access
        node_id_mapping = {
            "prompt": next((id for id, title in node_titles.items() if title == prompt_node_title), None),
            "seed": next((id for id, title in node_titles.items() if title == seed_node_title), None),
            "image": next((id for id, title in node_titles.items() if title == image_input_node_title), None), # Added 'image' key
            "save": next((id for id, title in node_titles.items() if title == save_node_title), None),
            "dialogue_prompt": next((id for id, title in node_titles.items() if title == dialogue_prompt_node_title and title != "None"), None),
            "ambient_audio_prompt": next((id for id, title in node_titles.items() if title == ambient_audio_prompt_node_title and title != "None"), None)
        }

        image_input_node_class_type = None
        if node_id_mapping["image"] and node_id_mapping["image"] in workflow_data:
            image_input_node_class_type = workflow_data[node_id_mapping["image"]].get("class_type")
            if image_input_node_class_type == "EmptyImage":
                st.warning(f"Warning: 'Image Input Node' is set to an 'EmptyImage' node ({node_id_mapping['image']}). "
                           "For true Image-to-Video (I2V) continuity, please ensure your ComfyUI workflow includes a 'LoadImage' node "
                           "and select it here. Image continuity features will be skipped with 'EmptyImage'.")
    else:
        st.error(f"Workflow file not found at {WORKFLOW_FILE}.")
        st.stop()
    
    st.markdown("---")
    st.header("🤖 AI Script Parsing")
    ollama_model = st.text_input("Ollama Model Name for Script Parsing:", value="llama3")
    generate_char_images = st.checkbox("Generate Character Images", value=True, help="Generate a still image for each character. Useful for I2V workflows or visual reference. Can be disabled for pure T2V workflows to save time.")

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


if st.button("✨ Generate Video from Script ✨"):
    if not all(node_id_mapping[key] for key in ["prompt", "seed", "image", "save"]):
        st.error("Required ComfyUI nodes (Prompt, Seed, Image Input, Save) are not mapped. Check sidebar selections.")
        st.stop()
    if not detailed_story_script.strip():
        st.warning("Please enter a detailed story script.")
        st.stop()

    # Use the session state directories
    COMFYUI_OUTPUT_DIR = st.session_state.comfyui_output_dir
    COMFYUI_INPUT_DIR = st.session_state.comfyui_input_dir

    status_area = st.empty()
    parsed_script = parse_detailed_script_with_ollama(detailed_story_script, ollama_model, status_area)

    if not parsed_script:
        st.error("Could not parse the detailed script. Halting.")
        st.stop()
        
    with st.expander("🤖 AI-Parsed Script Details", expanded=True):
        st.json(parsed_script)

    character_image_paths = {}
    if generate_char_images:
        # --- Character Generation ---
        st.header("🎭 Generating Characters...")
        all_characters_data = parsed_script.get("characters", [])

        if not all_characters_data:
            st.warning("No characters were identified by the AI. Proceeding with scene-to-scene continuity only.")
        else:
            for char_data in all_characters_data:
                char_name = char_data["name"]
                char_description = char_data["description"]
                st.subheader(f"Creating: {char_name}")
                char_prompt = f"A cinematic, high-quality, full-body shot of {char_name}, {char_description}, in a neutral setting."
                
                # Generate the character image
                char_image_path = generate_character(char_prompt, char_name, COMFYUI_INPUT_DIR, COMFYUI_OUTPUT_DIR, node_id_mapping, st.session_state.run_id)
                
                if char_image_path:
                    st.success(f"Successfully created image for {char_name}")
                    # The generate_character function now handles saving and displaying the image.
                    # We just need to store the resulting filename.
                    character_image_paths[char_name] = os.path.basename(char_image_path)
                else:
                    st.error(f"Failed to generate image for {char_name}.")

    # --- Scene Generation ---
    st.header("🎬 Generating Scenes...")
    generated_videos = []
    last_frame_path = None
    client_id = str(uuid.uuid4())

    with open(WORKFLOW_FILE, 'r') as f:
        base_workflow = json.load(f)

    # --- Build a more robust base prompt ---
    base_prompt = parsed_script.get("global_visual_description", "")
    character_descriptions = " ".join([f"{c['name']} is {c['description']}." for c in parsed_script.get("characters", [])])
    if character_descriptions:
        base_prompt += f" The characters are: {character_descriptions}"

    # --- Filter for visually generatable events and iterate ---
    visual_events = [event for event in parsed_script.get("scenes", []) if event.get("type") in ["visual_segment", "dialogue", "montage"]]

    for i, event in enumerate(visual_events):
        event_type = event.get("type")
        st.subheader(f"Generating Video Segment {i+1}/{len(visual_events)} (Type: {event_type})")
        progress_bar = st.progress(0)
        status_text = st.empty()

        workflow = base_workflow.copy()
        workflow[node_id_mapping["seed"]]["inputs"]["seed"] = random.randint(0, 1_000_000_000)

        run_id_prefix = st.session_state.run_id.split('-')[0]
        filename_prefix = f"{run_id_prefix}_scene_{i+1}"
        workflow[node_id_mapping["save"]]["inputs"]["filename_prefix"] = filename_prefix

        # --- CONSTRUCT THE FINAL PROMPT FOR THIS SCENE ---
        # This logic is updated to better follow the LTX-2 prompt guidelines
        # by creating a more cohesive, single-paragraph prompt for each scene.
        prompt_parts = [base_prompt]

        event_description = ""
        # For montages, combine segment descriptions
        if event_type == "montage":
            montage_descriptions = [s.get("description", "") for s in event.get("segments", [])]
            event_description = "The scene is a montage showing: " + ", ".join(filter(None, montage_descriptions))
        else: # For dialogue and visual_segment
            event_description = event.get("visual_prompt") or event.get("description")

        if event_description:
            prompt_parts.append(f"The current action is: {event_description}.")

        # If it's a dialogue event, add the spoken line as per LTX-2 guidelines.
        if event_type == "dialogue":
            char_name = event.get("character")
            dialogue = event.get("dialogue_text")
            if char_name and dialogue:
                prompt_parts.append(f'{char_name} says: "{dialogue}".')

        final_prompt = " ".join(filter(None, prompt_parts))

        # Set the final prompt in the workflow
        workflow[node_id_mapping["prompt"]]["inputs"]["text"] = final_prompt
        with st.expander("Full Prompt for this segment"):
            st.write(final_prompt)

        # --- Set dynamic scene length to manage VRAM ---
        frame_rate = 24 # Fallback default
        try:
            # Try to read frame rate from the workflow file for accuracy
            frame_rate = base_workflow.get("92:102", {}).get("inputs", {}).get("value", 24)
        except Exception:
            pass # Use fallback
        frame_count = int(scene_duration * frame_rate)
        workflow["92:62"]["inputs"]["value"] = frame_count
        st.info(f"Segment duration set to {scene_duration} seconds ({frame_count} frames @ {frame_rate}fps).")

        # --- INTELLIGENT CONTINUITY (IMAGE INPUT) ---
        if image_input_node_class_type != "EmptyImage":
            used_specific_image = False
            # Prioritize using the speaking character's image for dialogue
            if event_type == "dialogue":
                char_name = event.get("character")
                if char_name and char_name in character_image_paths:
                    workflow[node_id_mapping["image"]]["inputs"]["image"] = character_image_paths[char_name]
                    status_text.info(f"Using character image for '{char_name}'.")
                    used_specific_image = True

            # If no character image was used, fall back to the last frame for continuity
            if not used_specific_image and last_frame_path:
                workflow[node_id_mapping["image"]]["inputs"]["image"] = os.path.basename(last_frame_path)
                status_text.info("Using last frame for continuity.")
            elif not used_specific_image:
                status_text.info("No specific input image for this segment. Using default from workflow.")
        else:
            status_text.info("Image continuity skipped (EmptyImage node selected).")

        generated_video_path = queue_prompt_and_wait(workflow, client_id, progress_bar, status_text, COMFYUI_OUTPUT_DIR, node_id_mapping, filename_prefix)

        if generated_video_path and os.path.exists(generated_video_path):
            st.success(f"Segment {i+1} generated!")
            st.video(generated_video_path)
            generated_videos.append(generated_video_path)
            progress_bar.progress(100)

            # Extract last frame for the next segment's continuity
            if i < len(visual_events) - 1:
                next_frame_filename = f"scene_{i+1}_last_frame.png"
                next_frame_path = os.path.join(COMFYUI_INPUT_DIR, next_frame_filename)
                if extract_last_frame(generated_video_path, next_frame_path):
                    last_frame_path = next_frame_path
                else:
                    st.warning("Failed to extract last frame. Continuity may be affected.")
                    last_frame_path = None
        else:
            st.error(f"Segment {i+1} failed to generate. Stopping.")
            break

        # --- Add Cooldown ---
        if i < len(visual_events) - 1: # Don't wait after the last scene
            st.info(f"Cooldown: Waiting for {scene_cooldown} seconds to free up VRAM...")
            time.sleep(scene_cooldown)

    if generated_videos:
        st.header("Final Touches")
        final_video_path = os.path.join(COMFYUI_OUTPUT_DIR, FINAL_VIDEO_FILE)
        with st.spinner("Stitching video..."):
            stitch_videos(generated_videos, final_video_path)
        
        st.balloons()
        st.subheader("🎉 Final Masterpiece 🎉")
        if os.path.exists(final_video_path):
            st.video(final_video_path)
        else:
            st.warning("Final video not created.")

st.markdown("---")
st.info("Videos are saved in the `output` directory.")