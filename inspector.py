import json
import sys

def inspect_workflow(workflow_file):
    """
    Loads a ComfyUI workflow JSON file and prints a formatted list of all
    nodes, their IDs, titles, and class types.
    """
    try:
        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{workflow_file}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file '{workflow_file}' is not a valid JSON file.")
        sys.exit(1)

    print("-" * 80)
    print(f"Inspecting Workflow: {workflow_file}")
    print("-" * 80)
    print(f"{ 'Node ID':<10} | { 'Node Title':<40} | { 'Class Type':<30}")
    print("-" * 80)

    # Sort nodes by their ID, handling composite IDs like "group:node"
    def sort_key(item):
        parts = item[0].split(':')
        return tuple(map(int, parts))
    sorted_nodes = sorted(workflow.items(), key=sort_key)

    for node_id, node_data in sorted_nodes:
        # The title is stored in the _meta object for nodes added via the UI
        title = node_data.get("_meta", {}).get("title", "N/A")
        class_type = node_data.get("class_type", "N/A")

        print(f"{node_id:<10} | {title:<40} | {class_type:<30}")

    print("-" * 80)
    print("\nHow to use this:")
    print("1. Run the `app.py` Streamlit application.")
    print("2. In the sidebar, use the dropdown menus to select the nodes from your workflow.")
    print("   - **Positive Prompt Node:** Find the node where you type your main prompt (often a `CLIPTextEncode` or similar).")
    print("   - **Seed/Noise Node:** Find the node that controls the random seed (often a `KSampler` or a dedicated `Seed` node).")
    print("   - **Image Input Node:** Find the `LoadImage` or `EmptyImage` node that will be used for the initial scene or continuity.")
    print("   - **Save Video/Image Node:** Find the `SaveAnimatedPNG` or `VHS_VideoCombine` node that saves the final output.")
    print("\nThis script helps you match the titles in the dropdowns with the actual function of each node in your ComfyUI setup.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        workflow_path = sys.argv[1]
    else:
        # Default to the path used in the app if no argument is given
        workflow_path = "workfllows/video_ltx2_t2v.json"

    if not workflow_path.endswith('.json'):
        print("Error: Please provide a path to a .json file.")
    else:
        inspect_workflow(workflow_path)