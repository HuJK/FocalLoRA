"""
FocalLoRA Dataset Generation - Full Implementation
====================================================

Generates datasets following the paper's specification with three splits plus a
combined view:
- dev: fixed 8 tasks, saves only dev_eval.json (no per-constraint json files)
- test: 5% of remaining tasks, original format per constraint/format
- train: remaining tasks, original format per constraint/format
- combined: all samples from every split in one json
"""

import json
import os
import random
from collections import Counter
from typing import List, Dict

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = "data"
TASKS_FILE = "tasks.json"
OUTPUT_DIRS = {
    "train": os.path.join(BASE_PATH, "focal_lora_dataset_train"),
    "dev": os.path.join(BASE_PATH, "focal_lora_dataset_dev"),
    "test": os.path.join(BASE_PATH, "focal_lora_dataset_test")
}

for path in OUTPUT_DIRS.values():
    os.makedirs(path, exist_ok=True)

# ============================================================================
# CONSTRAINT DEFINITIONS (Table 3 from Paper)
# ============================================================================

CONSTRAINT_CONFIGS = {
    "language": {
        "abbr": "LAN",
        "difficulty": {"constraint_1": "hard", "constraint_2": "easy"},
        "simple": {
            "constraint_1": "Your entire response should be in English, no other language is allowed.",
            "constraint_2": "Your entire response should be in French, no other language is allowed."
        },
        "rich": {
            "constraint_1": (
                "As a professional assistant, you must maintain consistency in language usage. "
                "Your entire response should be in English, no other language is allowed. "
                "This is a strict requirement for all communications."
            ),
            "constraint_2": (
                "Pour assurer la cohérence de notre communication, veuillez répondre exclusivement "
                "en français. Your entire response should be in French, no other language is allowed. "
                "C'est une exigence importante."
            )
        }
    },

    "digit_presence": {
        "abbr": "DIG",
        "difficulty": {"constraint_1": "hard", "constraint_2": "easy"},
        "simple": {
            "constraint_1": "Do not include any digits (0-9) in the response.",
            "constraint_2": "Include at least three digits (0-9) somewhere in the response."
        },
        "rich": {
            "constraint_1": (
                "For clarity and readability, please avoid using any numeric digits (0-9) in your response. "
                "Write out numbers in word form instead. This helps maintain a consistent narrative style."
            ),
            "constraint_2": (
                "To support your answer with concrete examples, please include at least three digits (0-9) "
                "somewhere in the response. Specific numbers help illustrate your points more effectively."
            )
        }
    },

    "output_format": {
        "abbr": "FMT",
        "difficulty": {"constraint_1": "hard", "constraint_2": "easy"},
        "simple": {
            "constraint_1": "Respond strictly in JSON with keys 'answer' and 'reasoning'; output nothing else.",
            "constraint_2": "Respond in plain-text prose; JSON formatting is not allowed."
        },
        "rich": {
            "constraint_1": (
                "Your output must be machine-readable by another service. Please format your entire response "
                "strictly in JSON, using only the fields 'answer' and 'reasoning'. Do not include any "
                "explanation, commentary, or markdown formatting."
            ),
            "constraint_2": (
                "I'm a middle school student doing a science project. Please write your answer in plain English, "
                "as a single paragraph. Avoid using any JSON format or code blocks—I just want a normal, "
                "easy-to-read explanation."
            )
        }
    },

    "list_structure": {
        "abbr": "LST",
        "difficulty": {"constraint_1": "hard", "constraint_2": "easy"},
        "simple": {
            "constraint_1": "Provide exactly five numbered items (1.-5.).",
            "constraint_2": "Provide one single paragraph with no list formatting or line breaks."
        },
        "rich": {
            "constraint_1": (
                "To make your response easy to scan, please organize your answer as exactly five numbered items "
                "(1.-5.). Use clear list formatting with each point on a separate line."
            ),
            "constraint_2": (
                "I prefer reading continuous text rather than bullet points. Please provide one single paragraph "
                "with no list formatting or line breaks. Make it flow naturally as prose."
            )
        }
    },

    "quotation_marks": {
        "abbr": "QUO",
        "difficulty": {"constraint_1": "easy", "constraint_2": "hard"},
        "simple": {
            "constraint_1": 'Include at least one phrase enclosed in double quotation marks (" ").',
            "constraint_2": "Do not use any quotation marks in your response."
        },
        "rich": {
            "constraint_1": (
                'To emphasize key concepts or phrases, please include at least one phrase enclosed in double '
                'quotation marks (" "). This helps highlight important terminology or direct citations.'
            ),
            "constraint_2": (
                "For a clean, streamlined appearance, please do not use any quotation marks in your response. "
                "Paraphrase any concepts without using direct quotes."
            )
        }
    },

    "sentence_count": {
        "abbr": "SNT",
        "difficulty": {"constraint_1": "hard", "constraint_2": "easy"},
        "simple": {
            "constraint_1": "Write exactly ten sentences.",
            "constraint_2": "Write fewer than five sentences."
        },
        "rich": {
            "constraint_1": (
                "To ensure comprehensive coverage, please write exactly ten sentences in your response. "
                "This length allows for thorough explanation while maintaining focus."
            ),
            "constraint_2": (
                "I need a brief summary due to time constraints. Please write fewer than five sentences. "
                "Keep it concise and to the point."
            )
        }
    },

    "word_count": {
        "abbr": "WRD",
        "difficulty": {"constraint_1": "hard", "constraint_2": "easy"},
        "simple": {
            "constraint_1": "Write at least 300 words.",
            "constraint_2": "Write fewer than 50 words."
        },
        "rich": {
            "constraint_1": (
                "For a detailed and comprehensive explanation, please write at least 300 words. "
                "This length ensures you can cover all important aspects with sufficient depth and examples."
            ),
            "constraint_2": (
                "I'm looking for a quick answer that I can read in seconds. Please write fewer than 50 words. "
                "Be extremely concise and focus only on the essential information."
            )
        }
    },

    "case": {
        "abbr": "CAS",
        "difficulty": {"constraint_1": "hard", "constraint_2": "easy"},
        "simple": {
            "constraint_1": "Write the whole response in English using ALL CAPITAL LETTERS.",
            "constraint_2": "Write the whole response in English using all lowercase letters."
        },
        "rich": {
            "constraint_1": (
                "For emphasis and visibility, write the whole response in English using ALL CAPITAL LETTERS. "
                "This formatting requirement must be applied to every word in your answer."
            ),
            "constraint_2": (
                "For a casual, informal tone, write the whole response in English using all lowercase letters. "
                "Do not capitalize anything, including the first letter of sentences."
            )
        }
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_tasks(filepath: str) -> List[str]:
    """Load base tasks from JSON file."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Using default tasks.")
        return get_default_tasks()

    with open(filepath, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    return tasks

def get_default_tasks() -> List[str]:
    """Fallback tasks if tasks.json doesn't exist (from Table 4 in paper)."""
    return [
        "Describe the greenhouse effect and explain how human activities, such as fossil-fuel combustion, intensify this natural process.",
        "Explain quantum entanglement in accessible terms, then cite one landmark experiment that confirmed its non-classical correlations.",
        "Summarize the main political, economic, and social causes that led to World War I in a concise, chronological narrative.",
        "Provide a beginner-friendly introduction to machine learning and briefly contrast supervised with unsupervised learning.",
        "Explain how blockchain technology maintains a tamper-evident ledger and mention one real-world application beyond cryptocurrencies.",
        "Outline the three stages of cellular respiration, stating where each occurs in the cell and their approximate ATP yield.",
        "Describe the concept of supply and demand, and illustrate market equilibrium with a short numerical example.",
        "State Newton's first law of motion and give one everyday scenario that clearly demonstrates inertia.",
        "Give a step-by-step recipe for classic pancakes, including batter preparation and proper griddle temperature.",
        "Discuss two major ways the Renaissance reshaped European culture, touching on art and scientific inquiry.",
        "Explain the historical significance of the Magna Carta and cite one modern democratic principle it helped inspire.",
        "Restate the law of conservation of energy and illustrate it with the operation of a simple pendulum.",
        "Describe the basic structure of the Internet and outline how data packets travel from sender to receiver.",
        "Provide five practical safety tips to follow during and immediately after an earthquake.",
        "Describe the eight principal phases of the Moon and explain why they appear in a 29-day cycle.",
        "Explain plate tectonics theory and relate it to the formation of earthquakes and mountain ranges.",
        "Write clear, numbered instructions for changing a bicycle tire on the roadside without specialized tools.",
        "Provide a brief history of jazz music, mentioning its roots in New Orleans and its evolution through bebop.",
        "Describe the main functions of the United Nations and reference a recent humanitarian or peacekeeping mission.",
        "Explain the basic principles of quantum computing and note one challenge that hinders large-scale deployment."
    ]


def split_tasks(
    tasks: List[str],
    dev_count: int = 8,
    test_ratio: float = 0.05,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Shuffle and split tasks into train/dev/test.

    Dev set uses a fixed count (default 8). Test set uses a ratio of the
    remaining tasks. At least one task is allocated to each non-empty split
    when possible.
    """
    if test_ratio >= 1:
        raise ValueError("test_ratio must be less than 1.")

    rng = random.Random(seed)
    shuffled = tasks.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)
    dev_count = min(dev_count, total)
    test_count = max(1, int((total - dev_count) * test_ratio)) if total else 0

    # Ensure we do not exceed total tasks
    if dev_count + test_count > total:
        excess = dev_count + test_count - total
        # Reduce dev_count first, then test_count if needed
        reduce_dev = min(excess, dev_count)
        dev_count -= reduce_dev
        excess -= reduce_dev
        test_count = max(0, test_count - excess)

    dev_tasks = shuffled[:dev_count]
    test_tasks = shuffled[dev_count:dev_count + test_count]
    train_tasks = shuffled[dev_count + test_count:]

    return {
        "train": train_tasks,
        "dev": dev_tasks,
        "test": test_tasks
    }

def generate_samples(
    constraint_type: str,
    config: Dict,
    tasks: List[str],
    format_type: str  # "simple" or "rich"
) -> List[Dict]:
    """
    Generate samples for one constraint type and format.

    Creates both normal and conflict samples with role swapping as described in paper.
    """
    samples = []
    abbr = config["abbr"]
    constraints = config[format_type]

    for idx, task in enumerate(tasks, start=1):
        task_id = f"{idx:03d}"

        # ====================================================================
        # SCENARIO 1: Normal (Non-swapped)
        # System has constraint_1, user instruction is empty/compatible
        # ====================================================================
        samples.append({
            "id": f"{abbr}_{task_id}_normal_{format_type}",
            "system_message": constraints["constraint_1"],
            "user_message": "",  # No conflicting instruction
            "task": task,
            "label": "normal",
            "constraint_type": constraint_type,
            "format": format_type,
            "swapped": False
        })

        # ====================================================================
        # SCENARIO 2: Conflict (Non-swapped)
        # System has constraint_1, user has conflicting constraint_2
        # ====================================================================
        samples.append({
            "id": f"{abbr}_{task_id}_conflict_{format_type}",
            "system_message": constraints["constraint_1"],
            "user_message": constraints["constraint_2"],
            "task": task,
            "label": "conflict",
            "constraint_type": constraint_type,
            "format": format_type,
            "swapped": False
        })

        # ====================================================================
        # SCENARIO 3: Normal (Swapped)
        # System has constraint_2, user instruction is empty/compatible
        # This tests if the model can follow constraint_2 when it's in system
        # ====================================================================
        samples.append({
            "id": f"{abbr}_{task_id}_normal_{format_type}_swap",
            "system_message": constraints["constraint_2"],
            "user_message": "",
            "task": task,
            "label": "normal",
            "constraint_type": constraint_type,
            "format": format_type,
            "swapped": True
        })

        # ====================================================================
        # SCENARIO 4: Conflict (Swapped)
        # System has constraint_2, user has conflicting constraint_1
        # This avoids bias from always having same constraint in system
        # ====================================================================
        samples.append({
            "id": f"{abbr}_{task_id}_conflict_{format_type}_swap",
            "system_message": constraints["constraint_2"],
            "user_message": constraints["constraint_1"],
            "task": task,
            "label": "conflict",
            "constraint_type": constraint_type,
            "format": format_type,
            "swapped": True
        })

    return samples

# ============================================================================
# MAIN GENERATION LOGIC
# ============================================================================

def main():
    """Generate complete FocalLoRA dataset following paper specifications."""

    print("=" * 80)
    print("FocalLoRA Dataset Generation")
    print("=" * 80)

    # Load tasks
    tasks = load_tasks(TASKS_FILE)
    print(f"\nLoaded {len(tasks)} base tasks")

    # Split tasks into train/dev/test
    task_splits = split_tasks(tasks, dev_count=8, test_ratio=0.05, seed=42)
    print("Task split (train/dev/test): "
          f"{len(task_splits['train'])}/"
          f"{len(task_splits['dev'])}/"
          f"{len(task_splits['test'])}")

    combined_samples: List[Dict] = []
    split_stats = {}
    example_sample = None
    dev_eval_path = os.path.join(BASE_PATH, "focal_lora_dataset_dev", "dev_eval.json")
    global_combined_path = os.path.join(BASE_PATH, "focal_lora_dataset_all_combined.json")

    # Generate per split
    for split_name, split_task_list in task_splits.items():
        output_dir = OUTPUT_DIRS[split_name]
        print(f"\n{'=' * 80}")
        print(f"Generating split: {split_name.upper()} ({len(split_task_list)} tasks)")
        print(f"{'=' * 80}")

        split_samples: List[Dict] = []
        split_simple = 0
        split_rich = 0

        # Save fixed dev eval set (tasks + constraint configs) for 8 tasks
        if split_name == "dev":
            dev_payload = {
                "tasks": split_task_list,
                "constraint_configs": CONSTRAINT_CONFIGS
            }
            with open(dev_eval_path, 'w', encoding='utf-8') as f:
                json.dump(dev_payload, f, indent=2, ensure_ascii=False)
            print(f"  [DEV] Saved tasks + configs → {dev_eval_path}")

        # Generate for each constraint type
        for constraint_name, constraint_config in CONSTRAINT_CONFIGS.items():
            print(f"\n{'─' * 80}")
            print(f"[{split_name}] Constraint: {constraint_name.upper()}")
            print(f"{'─' * 80}")

            for format_type in ["simple", "rich"]:
                samples = generate_samples(
                    constraint_type=constraint_name,
                    config=constraint_config,
                    tasks=split_task_list,
                    format_type=format_type
                )

                split_samples.extend(samples)
                combined_samples.extend(samples)

                if example_sample is None and samples:
                    example_sample = samples[0]

                normal_count = sum(1 for s in samples if s['label'] == 'normal')
                conflict_count = sum(1 for s in samples if s['label'] == 'conflict')

                if format_type == "simple":
                    split_simple += len(samples)
                else:
                    split_rich += len(samples)

                if split_name in {"train", "test"}:
                    output_file = os.path.join(
                        output_dir,
                        f"{constraint_name}_{format_type}.json"
                    )
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(samples, f, indent=2, ensure_ascii=False)
                    print(f"  [{format_type:6}] {len(samples):4} samples "
                          f"(normal: {normal_count}, conflict: {conflict_count}) → {output_file}")
                else:
                    print(f"  [{format_type:6}] {len(samples):4} samples "
                          f"(normal: {normal_count}, conflict: {conflict_count}) added to combined only")

        split_total = len(split_samples)
        split_stats[split_name] = {
            "tasks": len(split_task_list),
            "samples": split_total,
            "simple": split_simple,
            "rich": split_rich,
            "output_dir": output_dir
        }

    # Combined view across all splits (outside split folders)
    with open(global_combined_path, 'w', encoding='utf-8') as f:
        json.dump(combined_samples, f, indent=2, ensure_ascii=False)

    # Summary
    label_counts = Counter(s["label"] for s in combined_samples)
    if combined_samples and label_counts.get("conflict", 0) == 0:
        raise ValueError("Combined dataset missing conflict samples; generation aborted.")

    print(f"\n{'=' * 80}")
    print("DATASET GENERATION COMPLETE")
    print(f"{'=' * 80}")
    total_samples = len(combined_samples)
    print(f"\nTotal samples generated across splits (combined): {total_samples}")
    print(f"Label counts (combined): {dict(label_counts)}")
    max_scenarios = len(CONSTRAINT_CONFIGS) * 2 * 4  # constraints × formats × scenarios per task
    print(f"Expected per split (constraints×formats×scenarios_per_task): {max_scenarios} × tasks_in_split")
    for split_name, stats in split_stats.items():
        expected = max_scenarios * stats["tasks"]
        print(f"  • {split_name}: {stats['samples']} samples "
              f"(expected {expected}) from {stats['tasks']} tasks "
              f"→ {stats['output_dir']}")
    print(f"\nCombined all splits → {global_combined_path} ({len(combined_samples)} samples)")
    print(f"Dev eval set → {dev_eval_path}")

    # Data structure example
    if example_sample:
        print(f"\n{'=' * 80}")
        print("Sample Data Structure (compatible with code/_tuning.py)")
        print(f"{'=' * 80}")
        print(json.dumps(example_sample, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 80}")
    print("Usage Instructions")
    print(f"{'=' * 80}")
    print("""
For head detection (CSHI phase), use the global combined file:
    python code/_tuning.py \\
        --json_path data/focal_lora_dataset_all_combined.json \\
        --model_path <your_model_path> \\
        --tune_path data/focal_lora_dataset_train \\
        --output_dir outputs_lora \\
        --topk 10

The code will automatically combine 'task' and 'user_message' fields:
    usr = f"{s['task']} {s['user_message']}".strip() or s["task"]

For conflict samples (if your dataloader gathers them directly):
    - system_message: high-priority constraint
    - user_message: conflicting constraint
    - task: the actual task to perform
    """)

if __name__ == "__main__":
    main()
