# -*- coding: utf-8 -*-
"""
clean_important_heads_outputs.py
================================
This script recursively traverses a result directory and processes each
`important_heads.json` file by trimming the "assistant" part from the
"output" field inside each response, keeping only the actual model output.
"""

import os
import json
from tqdm import tqdm

def extract_assistant_only(output_text):
    """Keep only the part after 'assistant' if present."""
    if "assistant" in output_text:
        return output_text.split("assistant", 1)[-1].strip()
    else:
        return output_text.strip()

def process_json_file(file_path):
    """Process a single important_heads.json file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "responses" not in data:
        print(f"⚠️ Skipping file {file_path}: missing 'responses' field.")
        return

    for resp in data["responses"]:
        if "output" in resp:
            resp["output"] = extract_assistant_only(resp["output"])

    # Overwrite the original file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def traverse_and_process(root_dir):
    """Recursively traverse the directory and process all important_heads.json files."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "important_heads.json":
                file_path = os.path.join(dirpath, filename)
                process_json_file(file_path)

if __name__ == "__main__":
    # Replace with your actual root directory, e.g., "results"
    root_directory = "results"
    traverse_and_process(root_directory)
    print("✅ All files have been processed.")
