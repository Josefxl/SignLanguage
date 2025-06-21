import json
import os
from collections import defaultdict, Counter
from datetime import datetime


class ReinforcementMemory:
    def __init__(self, log_dir="logs", memory_file="reinforcement_memory.json"):
        self.log_dir = log_dir
        self.memory_path = os.path.join(log_dir, memory_file)
        self.memory = defaultdict(list)
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as f:
                self.memory = defaultdict(list, json.load(f))

    def save_memory(self):
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.memory_path, "w") as f:
            json.dump(self.memory, f, indent=2)

    def prune_memory(self, threshold=0.9, min_successes=2):
        # Remove prompts with consistently high success
        strong = self.get_strong_glosses(threshold, min_successes)
        for prompt in strong:
            if prompt in self.memory:
                del self.memory[prompt]

    def update(self, session_log):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for entry in session_log:
            key = entry['prompt']
            self.memory[key].append({
                "gloss": entry['gloss'],
                "caption": entry['caption'],
                "similarity": entry['similarity'],
                "result": entry['result'],
                "timestamp": timestamp
            })
        self.prune_memory()
        self.save_memory()

    def get_weak_glosses(self, threshold=0.6, min_failures=2):
        failures = Counter()
        for prompt, records in self.memory.items():
            for r in records:
                if r["similarity"] < threshold:
                    failures[prompt] += 1
        return [g for g, count in failures.items() if count >= min_failures]

    def get_strong_glosses(self, threshold=0.9, min_successes=2):
        successes = Counter()
        for prompt, records in self.memory.items():
            for r in records:
                if r["similarity"] >= threshold:
                    successes[prompt] += 1
        return [g for g, count in successes.items() if count >= min_successes]

    def summarize_history(self):
        summary = {
            "total_prompts": len(self.memory),
            "most_common_failures": self.get_weak_glosses(),
            "most_common_successes": self.get_strong_glosses(),
        }
        return summary

    def clear_memory(self):
        self.memory = defaultdict(list)
        self.save_memory()
        print("🧽 Reinforcement memory cleared.")

