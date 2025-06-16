import random

class AdaptiveTutorAgent:
    def __init__(self):
        self.performance_level = "beginner"
        self.history = []
        self.prompt_bank = {
            "beginner": {
                "greetings": ["Hello", "Good morning", "How are you?"],
                "basic needs": ["I am hungry.", "I need water.", "Please help me."]
            },
            "intermediate": {
                "questions": ["Where do you live?", "What is your name?"],
                "statements": ["I want to go to the store.", "I am learning sign language."]
            },
            "advanced": {
                "abstract": ["What are your goals in life?", "How do you feel about change?"],
                "opinions": ["I think learning is important.", "I disagree with that idea."]
            }
        }

    def update_performance(self, correct, total):
        accuracy = correct / total
        if accuracy >= 0.85:
            self.performance_level = "advanced"
        elif accuracy >= 0.60:
            self.performance_level = "intermediate"
        else:
            self.performance_level = "beginner"

    def get_prompt(self):
        categories = list(self.prompt_bank[self.performance_level].values())
        flat_prompts = [prompt for group in categories for prompt in group]
        unused = [p for p in flat_prompts if p not in self.history]
        if not unused:
            self.history = []  # reset if all are used
            unused = flat_prompts
        chosen = random.choice(unused)
        self.history.append(chosen)
        return chosen