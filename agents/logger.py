import json
import os
from datetime import datetime


class SessionLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_session(self, prompts, session_log, score):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_data = {
            "timestamp": timestamp,
            "score": score,
            "prompts": prompts,
            "results": session_log
        }

        filename = f"session_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"\n💾 Session log saved to: {filepath}")
