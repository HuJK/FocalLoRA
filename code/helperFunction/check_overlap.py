import os
import json
from collections import Counter

def collect_important_heads(root_dir):
    head_counter = Counter()

    # Traverse all subdirectories ending with "_outputs" under the results directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not dirpath.endswith("_outputs"):
            continue

        for filename in filenames:
            if filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "important_heads" in data:
                            for head_info in data["important_heads"]:
                                if isinstance(head_info, list) and len(head_info) >= 1:
                                    head_counter[head_info[0]] += 1
                except Exception as e:
                    print(f"Error reading file: {file_path}, Error: {e}")

    return head_counter

def main():
    results_path = "results"
    head_counts = collect_important_heads(results_path)

    print("Important head frequency (sorted by descending count):")
    for head, count in head_counts.most_common():
        print(f"{head}: {count} times")

if __name__ == "__main__":
    main()
