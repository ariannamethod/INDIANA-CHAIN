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

load_dotenv(override=True)

api = HfApi()


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

    return chat_messages


def process_split(config: Dict[str, Any], dataset_path: str, destination_path: str) -> Callable:
    """Process a single dataset split."""
    subset_name = config['subset_name']
    repo_id = "smolagents/aguvis-stage-2"

    # Check if the subset already exists in the remote dataset
    if check_subset_exists(repo_id, subset_name):
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

    processed_data = []
    current_weight = 0
    shard_number = 0
    MAX_WEIGHT = 1000 * 1024 * 1024

    def process_items() -> Generator[Dict[str, Any], None, None]:
        pbar = tqdm(data)
        for item in pbar:
            # Extract image paths from the data item
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
        process_items = process_split(config, dataset_path, f"{config['subset_name']}")
        
        # Skip if process_split returned None (subset already exists)
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