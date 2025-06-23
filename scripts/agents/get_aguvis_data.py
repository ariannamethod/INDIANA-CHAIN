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
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi, login, snapshot_download
from PIL import Image

load_dotenv(override=True)


def discover_dataset_config(dataset_path: str) -> List[Dict[str, Any]]:
    """Discover dataset configuration by scanning the data/aguvis/train directory."""
    dataset_dir = Path(dataset_path)
    train_dir = dataset_dir / "data" / "aguvis" / "train"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    configs = []

    # Find all JSON files in the train directory
    for json_file in train_dir.glob("*.json"):
        base_name = json_file.stem.replace("-l1", "").replace("-l2", "")

        # Determine the images folder based on the base name
        images_folder = None

        # Generate potential folder names by trying common patterns
        potential_folders = [base_name, f"{base_name}/images"]

        # Find the first existing folder
        for folder in potential_folders:
            full_folder_path = dataset_dir / folder
            if full_folder_path.exists():
                images_folder = folder
                break

        if images_folder is None:
            print(
                f"Warning: No images folder found for {base_name}, trying default pattern"
            )
            images_folder = f"{base_name}/images"

        config = {
            "json_path": str(json_file.relative_to(dataset_dir)),
            "images_folder": images_folder,
            "sampling_strategy": "all",  # Default to all for now
            "split_name": base_name,
        }

        configs.append(config)
        print(f"Discovered config: {base_name} -> {images_folder}")

    return configs


def download_dataset(
    repo_id: str = "xlangai/aguvis-stage2", local_dir: str = "./aguvis_raw"
) -> str:
    """Download the dataset using snapshot_download."""
    print(f"Downloading dataset from {repo_id}...")
    try:
        local_path = snapshot_download(
            repo_id=repo_id, local_dir=local_dir, repo_type="dataset"
        )
        print(f"Dataset downloaded to: {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("This might be due to authentication issues or network problems.")
        raise


def extract_zip_files(dataset_path: str):
    """Extract all zip files found in the dataset directory."""
    print("Extracting zip files...")
    dataset_dir = Path(dataset_path)

    for zip_file in dataset_dir.rglob("*.zip"):
        print(f"Extracting: {zip_file}")
        extract_dir = zip_file.parent / zip_file.stem

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        print(f"Extracted to: {extract_dir}")


def load_images_from_folder(
    images_folder: Path, image_paths: List[str]
) -> List[Image.Image]:
    """Load images from the specified folder."""
    images = []
    for img_path in image_paths:
        full_path = images_folder / img_path
        if full_path.exists():
            try:
                img = Image.open(full_path)
                images.append(img.copy())
                img.close()
            except Exception as e:
                print(f"Warning: Could not load image {full_path}: {e}")
        else:
            print(f"Warning: Image not found: {full_path}")
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


def process_split(config: Dict[str, Any], dataset_path: str) -> Dataset:
    """Process a single dataset split."""
    print(f"Processing split: {config['split_name']}")

    dataset_dir = Path(dataset_path)
    json_path = dataset_dir / config["json_path"]
    images_folder = dataset_dir / config["images_folder"]

    if not json_path.exists():
        print(f"Warning: JSON file not found: {json_path}")
        return None

    if not images_folder.exists():
        print(f"Warning: Images folder not found: {images_folder}")
        return None

    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    processed_data = []

    for item in data:
        try:
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

            # Convert to chat format
            texts = convert_to_chat_format(item)

            processed_data.append({"images": images, "texts": texts})

        except Exception as e:
            print(f"Warning: Error processing item: {e}")
            continue

    print(f"Processed {len(processed_data)} items for split {config['split_name']}")

    # Create dataset
    dataset = Dataset.from_list(processed_data)
    return dataset


def upload_dataset(
    dataset_dict: DatasetDict, repo_id: str = "smolagents/aguvis-stage-2"
):
    """Upload the processed dataset to HuggingFace Hub."""
    print(f"Uploading dataset to {repo_id}...")

    # Create the repository if it doesn't exist
    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Repository creation info: {e}")

    # Push to hub
    try:
        dataset_dict.push_to_hub(repo_id)
        print(f"Dataset uploaded successfully to {repo_id}")
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        print("This might be due to authentication issues or insufficient permissions.")
        raise


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

    # Step 1: Download dataset
    dataset_path = download_dataset()

    # Step 2: Extract zip files
    extract_zip_files(dataset_path)

    # Step 3: Discover dataset configuration
    dataset_configs = discover_dataset_config(dataset_path)

    # Step 4: Process each split
    dataset_dict = {}

    for config in dataset_configs:
        print(f"\n{'=' * 50}")
        dataset = process_split(config, dataset_path)

        if dataset is not None:
            dataset_dict[config["split_name"]] = dataset

        # Force garbage collection to manage memory
        gc.collect()

    # Step 5: Create DatasetDict and upload
    if dataset_dict:
        final_dataset = DatasetDict(dataset_dict)
        upload_dataset(final_dataset)
    else:
        print("No datasets were successfully processed.")

    # Cleanup
    print("\nCleaning up temporary files...")
    shutil.rmtree(dataset_path, ignore_errors=True)

    print("Process completed!")


if __name__ == "__main__":
    main()
