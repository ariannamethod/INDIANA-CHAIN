#!/usr/bin/env python3
"""
Script to download, process, and upload the aguvis-stage2 dataset.
Downloads from huggingface.co/datasets/xlangai/aguvis-stage2 and uploads to smolagents/aguvis-stage-2
"""

import gc
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Generator, Callable

from tqdm import tqdm
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, login, snapshot_download
from PIL import Image
from huggingface_hub import upload_large_folder

from smolagents.models import get_clean_message_list
from datasets import load_dataset
from dotenv import load_dotenv
import os

api = HfApi()

SYSTEM_PROMPT= """You are a helpful GUI agent. You will be given a task and a screenshot of the screen. You need to perform a series of function calls in code to complete the task.

When you send a message containing Python code between '<code>' and '</code>' tags, it will be executed in a stateful Jupyter notebook environment, and you will then be given the output to continued reasoning in an agentic loop.

The following functions are exposed to the Python interpreter:
<code>
def final_answer(answer: any) -> any:
    \"\"\"
    Provides a final answer to the given problem.
    Args:
        answer: The final answer to the problem
    \"\"\"

def click(x: int, y: int) -> str:
    \"\"\"
    Performs a left-click at the specified coordinates
    Args:
        x: The x coordinate (horizontal position)
        y: The y coordinate (vertical position)
    \"\"\"

def right_click(x: int, y: int) -> str:
    \"\"\"
    Performs a right-click at the specified coordinates
    Args:
        x: The x coordinate (horizontal position)
        y: The y coordinate (vertical position)
    \"\"\"

def double_click(x: int, y: int) -> str:
    \"\"\"
    Performs a double-click at the specified coordinates
    Args:
        x: The x coordinate (horizontal position)
        y: The y coordinate (vertical position)
    \"\"\"

def type_text(text: str) -> str:
    \"\"\"
    Types the specified text at the current cursor position.
    Args:
        text: The text to type
    \"\"\"

def press_key(key: str) -> str:
    \"\"\"
    Presses a keyboard key
    Args:
        key: The key to press (e.g. "enter", "space", "backspace", etc.).
    \"\"\"

def go_back() -> str:
    \"\"\"
    Goes back to the previous page in the browser. If using this tool doesn't work, just click the button directly.
    Args:
    \"\"\"

def drag_and_drop(x1: int, y1: int, x2: int, y2: int) -> str:
    \"\"\"
    Clicks [x1, y1], drags mouse to [x2, y2], then release click.
    Args:
        x1: origin x coordinate
        y1: origin y coordinate
        x2: end x coordinate
        y2: end y coordinate
    \"\"\"

def scroll(x: int, y: int, direction: Literal["up", "down"] = "down", amount: int = 2) -> str:
    \"\"\"
    Moves the mouse to selected coordinates, then uses the scroll button: this could scroll the page or zoom, depending on the app. DO NOT use scroll to move through linux desktop menus.
    Args:
        x: The x coordinate (horizontal position) of the element to scroll/zoom
        y: The y coordinate (vertical position) of the element to scroll/zoom
        direction: The direction to scroll ("up" or "down"), defaults to "down". For zoom, "up" zooms in, "down" zooms out.
        amount: The amount to scroll. A good amount is 1 or 2.
    \"\"\"

def wait(seconds: float) -> str:
    \"\"\"
    Waits for the specified number of seconds. Very useful in case the prior order is still executing (for example starting very heavy applications like browsers or office apps)
    Args:
        seconds: Number of seconds to wait, generally 3 is enough.
    \"\"\"
</code>

The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
"""



config_dict = [{
    "json_path": "mind2web-l1.json",
    "images_folder": "mind2web/",
    "sampling_strategy": "all"
}, {
    "json_path": "mind2web-l2.json",
    "images_folder": "mind2web/",
    "sampling_strategy": "all"}, {
        "json_path": "mind2web-l2.json",
    "images_folder": "mind2web/",
    "sampling_strategy": "all"}, {
        "json_path": "guiact-web-single.json",
    "images_folder": "guiact-web-single/images/",
    "sampling_strategy": "all"}, {
        "json_path": "guiact-web-multi-l1.json",
    "images_folder": "guiact-web-multi/images/",
    "sampling_strategy": "all"}, {
        "json_path": "guiact-web-multi-l2.json",
    "images_folder": "guiact-web-multi/images/",
    "sampling_strategy": "all"}, {
        "json_path": "miniwob-l1.json",
    "images_folder": "miniwob/images",
    "sampling_strategy": "all"}, {
        "json_path": "miniwob-l2.json",
    "images_folder": "miniwob/images/",
    "sampling_strategy": "all"},
    {
    "json_path": "coat.json",
    "images_folder": "coat/images/",
    "sampling_strategy": "all"},
    {
        "json_path": "android_control.json",
    "images_folder": "android_control/images/",
    "sampling_strategy": "all"},
    {
        "json_path": "gui-odyssey-l1.json",
    "images_folder": "gui-odyssey/images/",
    "sampling_strategy": "random:33%"}, {
        "json_path": "gui-odyssey-l2.json",
    "images_folder": "gui-odyssey/images/",
    "sampling_strategy": "random:33%"}, {
        "json_path": "gui-odyssey-l2.json",
    "images_folder": "gui-odyssey/images/",
    "sampling_strategy": "random:33%"}, {
        "json_path": "amex-l1.json",
    "images_folder": "amex/images/",
    "sampling_strategy": "random:33%"}, {
        "json_path": "amex-l2.json",
    "images_folder": "amex/images/",
    "sampling_strategy": "random:33%"}, {
        "json_path": "amex-l2.json",
    "images_folder": "amex/images/",
    "sampling_strategy": "random:33%"}, {
        "json_path": "aitw-l1.json",
    "images_folder": "aitw/images",
    "sampling_strategy": "all"},
    {
        "json_path": "aitw-l2.json",
        "images_folder": "aitw/images/",
        "sampling_strategy": "all"
    },
]


def discover_dataset_config(dataset_path: str) -> List[Dict[str, Any]]:
    """Discover dataset configuration by scanning the data directory."""
    dataset_dir = Path(dataset_path)
    train_dir = dataset_dir

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    configs = []
    processed_splits = set()

    # Find all JSON files in the train directory
    for config in config_dict:
        subset_name = config["json_path"].replace(".json", "").replace("-l1", "").replace("-l2", "")
        
        # Skip if we already processed this split
        if subset_name in processed_splits:
            continue
            
        config["subset_name"] = subset_name
        configs.append(config)
        processed_splits.add(subset_name)
        print(f"Discovered config: {config['subset_name']} -> {config['images_folder']}")

    return configs


def download_dataset(
    repo_id: str = "xlangai/aguvis-stage2", local_dir: str = "./aguvis_raw"
) -> str:
    """Download the dataset using snapshot_download."""
    print(f"Downloading dataset from {repo_id}...")
    local_path = snapshot_download(
        repo_id=repo_id, local_dir=local_dir, repo_type="dataset"
    )
    print(f"Dataset downloaded to: {local_path}")
    return local_path


def extract_zip_files(dataset_path: str):
    """Extract all zip files found in the dataset directory, but only if not already extracted."""
    print("Extracting zip files...")
    dataset_dir = Path(dataset_path)

    for zip_file in dataset_dir.rglob("*.zip"):
        extract_dir = zip_file.parent / zip_file.stem
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"Skipping extraction for {zip_file} (already extracted at {extract_dir})")
            continue

        print(f"Extracting: {zip_file}")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted to: {extract_dir}")


def check_subset_exists(repo_id: str, subset_name: str) -> bool:
    """Check if a subset already exists in the remote dataset."""
    try:
        # Try to get dataset info with specific subset
        from datasets import get_dataset_config_names
        config_names = get_dataset_config_names(repo_id)
        return subset_name in config_names
    except Exception as e:
        print(f"Could not check if subset exists: {e}")
        return False


def load_images_from_folder(
    images_folder: Path, image_paths: List[str]
) -> List[Image.Image]:
    """Load images from the specified folder."""
    images = []
    for img_path in image_paths:
        full_path = images_folder / img_path
        img = Image.open(full_path)
        images.append(img.copy())
        img.close()
    return images


def convert_to_smolagents(messages: list[dict[str, Any]]):
    output_messages = [{
        "content": SYSTEM_PROMPT,
        "role": "system"
    }]
    previous_role = None
    for i in range(1, len(messages)):
        content = messages[i]["content"]

        # Convert the format for content
        content = content.replace("answer(", "final_answer(")

        if messages[i]["role"] == "assistant":
            if content.startswith("Action: "):
                content = content.replace("Action: ", "<think>\n").strip()
                content += "\n</think>\n"
            else:
                content = "<code>\n" + content.replace("pyautogui.", "").strip() + "\n</code>"

        messages[i]["content"] = content

        # Fuse subsequent messages if they are both assistants
        if messages[i]["role"] == "assistant" and messages[i-1]["role"] == "assistant":
            # Need to fuse both messages
            output_messages[-1]["content"] += messages[i]["content"]
        else:
            output_messages.append(messages[i])
    return output_messages

def test_conversion():
    origin = [
    {
        "content": "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of actions to complete the task.\n\nYou have access to the following functions:\n- {\"name\": \"answer\", \"description\": \"Answer a question\", \"parameters\": {\"type\": \"object\", \"properties\": {\"answer\": {\"type\": \"string\", \"description\": \"The answer to the question\"}}, \"required\": [\"answer\"]}}\n",
        "role": "assistant"
    },
    {
        "content": "<image>\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: What information does the site provide about Judith Lauand's career, works and exhibitions?\n\nPrevious actions:\nStep 1: Click on the link labeled 'Judith Lauand: Brazilian 1922-2022' to explore more about her career and exhibitions.\nStep 2: Click on the 'more' link below the overview text to access additional information about Judith Lauand's career and exhibitions.\nStep 3: Scroll down slightly to view additional information about Judith Lauand's career and exhibitions.",
        "role": "user"
    },
    {
        "content": "Action: The answer is 'Judith Lauand was a Brazilian painter who was born in 1922.\\nNumerous key galleries and museums such as MASP, Museu de Arte de São Paulo have featured Judith Lauand's work in the past.\\nJudith Lauand's work has been offered at auction multiple times, with realized prices ranging from 515 USD to 87,500 USD, depending on the size and medium of the artwork. Since 2011 the record price for this artist at auction is 87,500 USD for Composition on Red Background, sold at Christie's New York in 2015.'\n",
        "role": "assistant"
    },
    {
        "content": "answer('Judith Lauand was a Brazilian painter who was born in 1922.\nNumerous key galleries and museums such as MASP, Museu de Arte de São Paulo have featured Judith Lauand's work in the past.\nJudith Lauand's work has been offered at auction multiple times, with realized prices ranging from 515 USD to 87,500 USD, depending on the size and medium of the artwork. Since 2011 the record price for this artist at auction is 87,500 USD for Composition on Red Background, sold at Christie's New York in 2015.')",
        "role": "assistant"
    }
    ]
    converted = convert_to_smolagents(origin)
    print("CONVERTED:\n", converted)
    expected_messages = [
    {
        "content": SYSTEM_PROMPT,
        "role": "system"
    },
    {
        "content": "<image>\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: What information does the site provide about Judith Lauand's career, works and exhibitions?\n\nPrevious actions:\nStep 1: Click on the link labeled 'Judith Lauand: Brazilian 1922-2022' to explore more about her career and exhibitions.\nStep 2: Click on the 'more' link below the overview text to access additional information about Judith Lauand's career and exhibitions.\nStep 3: Scroll down slightly to view additional information about Judith Lauand's career and exhibitions.",
        "role": "user"
    },
    {
        "content": "<think>The answer is 'Judith Lauand was a Brazilian painter who was born in 1922.\\nNumerous key galleries and museums such as MASP, Museu de Arte de São Paulo have featured Judith Lauand's work in the past.\\nJudith Lauand's work has been offered at auction multiple times, with realized prices ranging from 515 USD to 87,500 USD, depending on the size and medium of the artwork. Since 2011 the record price for this artist at auction is 87,500 USD for Composition on Red Background, sold at Christie's New York in 2015.'\n</think>\n<code>\nfinal_answer(\"The answer is 'Judith Lauand was a Brazilian painter who was born in 1922.\\nNumerous key galleries and museums such as MASP, Museu de Arte de São Paulo have featured Judith Lauand's work in the past.\\nJudith Lauand's work has been offered at auction multiple times, with realized prices ranging from 515 USD to 87,500 USD, depending on the size and medium of the artwork. Since 2011 the record price for this artist at auction is 87,500 USD for Composition on Red Background, sold at Christie's New York in 2015.'\n\")\n</code>",
        "role": "assistant"
    },
    ]
    for i, message in enumerate(converted):
        if not message == expected_messages[i]:
            print(f"Message {i} is not equal to expected message")
            print(f"Expected: {expected_messages[i]}")
            print(f"Actual: {message}")
            return False
    return True

def convert_to_chat_format(data_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert data item to chat template format."""
    # This is a placeholder - you'll need to adapt this based on the actual data structure
    # The exact conversion depends on how the original data is structured
    chat_messages = []

    # Example conversion - adapt based on actual data structure
    if "conversations" in data_item:
        for conv in data_item["conversations"]:
            if "from" in conv and "value" in conv:
                role = "user" if conv["from"] == "human" else "assistant"
                message = {"role": role, "content": conv["value"]}
                chat_messages.append(message)
    elif "instruction" in data_item and "response" in data_item:
        chat_messages = [
            {"role": "user", "content": data_item["instruction"]},
            {"role": "assistant", "content": data_item["response"]},
        ]

    chat_messages = convert_to_smolagents(chat_messages)
    return chat_messages


def process_subset(config: Dict[str, Any], dataset_path: str, destination_path: str, override_existing: bool = False) -> Callable:
    """Process a single dataset subset."""
    subset_name = config['subset_name']
    repo_id = "smolagents/aguvis-stage-2"

    # Check if the subset already exists in the remote dataset
    if check_subset_exists(repo_id, subset_name) and not override_existing:
        print(f"Subset '{subset_name}' already exists in {repo_id}, skipping processing.")
        return None

    print(f"Processing split: {subset_name}")

    dataset_dir = Path(dataset_path)
    images_folder = dataset_dir / config["subset_name"] / config["images_folder"]

    # Find all JSON files that match this split (e.g., mind2web-l1.json, mind2web-l2.json)
    json_files = []
    for cfg in config_dict:
        cfg_split = cfg["json_path"].replace(".json", "").replace("-l1", "").replace("-l2", "")
        if cfg_split == subset_name:
            json_path = dataset_dir / cfg["json_path"]
            if json_path.exists():
                json_files.append(json_path)

    # Load and merge JSON data from all matching files
    data = []
    for json_file in json_files:
        print(f"Loading data from: {json_file}")
        with open(json_file, "r") as f:
            file_data = json.load(f)
            data.extend(file_data)
            print(f"  Added {len(file_data)} items")

    def get_images_total_weight(images_folder: Path, image_paths: list) -> int:
        try:
            return sum(os.path.getsize(images_folder / img_path) for img_path in image_paths)
        except Exception as e:
            print(f"Error getting image weight: {e}", images_folder, image_paths)
            return 0


    def process_items() -> Generator[Dict[str, Any], None, None]:
        pbar = tqdm(data)
        for item in pbar:
            # Extract image paths from the data item
            try:
                image_paths = []
                if "images" in item:
                    image_paths = (
                        item["images"]
                        if isinstance(item["images"], list)
                        else [item["images"]]
                    )
                elif "image" in item:
                    image_paths = [item["image"]]

                # Load images
                images = load_images_from_folder(images_folder, image_paths)

                texts = convert_to_chat_format(item)

                entry = {"images": images, "texts": texts}
                yield entry
            except Exception as e:
                print(f"Error processing item: {e}", item)
                continue
    return process_items


def authenticate_huggingface():
    """Authenticate with HuggingFace Hub using token."""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Authenticating with HuggingFace Hub using token...")
        login(token=hf_token)
    else:
        raise ValueError("HF_TOKEN environment variable not set.")


def main():
    """Main function to orchestrate the entire process."""
    load_dotenv(override=True)

    print("Starting aguvis-stage2 dataset processing...")

    # Step 0: Authenticate with HuggingFace Hub
    authenticate_huggingface()

    data_folder = Path("./aguvis_raw")

    dataset_path = download_dataset("xlangai/aguvis-stage2", data_folder)

    extract_zip_files(dataset_path)

    dataset_configs = discover_dataset_config(dataset_path)
    converted_folder = "./aguvis_converted"

    for config in dataset_configs:
        print(f"\n{'=' * 50}")
        print(config)
        process_items = process_subset(config, dataset_path, f"{config['subset_name']}", override_existing=True)
        
        # Skip if process_subset returned None (subset already exists)
        if process_items is None:
            continue
            
        print("Creating dataset...")
        data = Dataset.from_generator(process_items)
        print("Pushing to hub...")
        # Fix: Use config_name for subset name and split="train"
        data.push_to_hub(
            "smolagents/aguvis-stage-2", 
            config_name=config['subset_name'],  # This sets the subset name
            split="train",  # This should be "train" not the subset name
        )

        print(f"Processed and uploaded subset: {config['subset_name']}")

        # Force garbage collection to manage memory
        gc.collect()

    print(f"Subsets uploaded!")

    # Cleanup
    print("\nCleaning up temporary files...")
    # shutil.rmtree(dataset_path, ignore_errors=True)

    # api.upload_large_folder(folder_path=converted_folder, repo_id="smolagents/aguvis-stage-2", repo_type="dataset")

    shutil.rmtree(converted_folder, ignore_errors=True)

    print("All done!")


if __name__ == "__main__":
    main()