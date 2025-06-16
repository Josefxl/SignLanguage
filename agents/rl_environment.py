
class TeachingEnvironment:
    def __init__(self):
        self.state = {
            "accuracy": 0.0,
            "difficulty_level": "beginner",
            "consecutive_successes": 0,
            "consecutive_failures": 0
        }
        self.difficulty_levels = ["beginner", "intermediate", "advanced"]
        self.history = []

    def get_state(self):
        return self.state

    def get_available_actions(self):
        return ["increase_difficulty", "decrease_difficulty", "keep_difficulty"]

    def apply_action(self, action):
        current_index = self.difficulty_levels.index(self.state["difficulty_level"])
        if action == "increase_difficulty" and current_index < len(self.difficulty_levels) - 1:
            self.state["difficulty_level"] = self.difficulty_levels[current_index + 1]
        elif action == "decrease_difficulty" and current_index > 0:
            self.state["difficulty_level"] = self.difficulty_levels[current_index - 1]
        # "keep_difficulty" does nothing
        return self.state["difficulty_level"]

    def compute_reward(self, correct, total):
        if total == 0:
            return 0
        accuracy = correct / total
        self.state["accuracy"] = accuracy

        if accuracy >= 0.85:
            self.state["consecutive_successes"] += 1
            self.state["consecutive_failures"] = 0
        else:
            self.state["consecutive_failures"] += 1
            self.state["consecutive_successes"] = 0

        reward = 0
        if self.state["consecutive_successes"] >= 2:
            reward = 1
        elif self.state["consecutive_failures"] >= 2:
            reward = -1

        return reward

    def log_session(self, prompt, result):
        self.history.append((prompt, result))
