import random
from agents.reinforce_memory import ReinforcementMemory
from agents.prompt_generator import PromptLLMGenerator

class AdaptiveTutorAgent:
    def __init__(self):
        self.performance_level = "beginner"
        self.memory = ReinforcementMemory()
        self.prompt_generator = PromptLLMGenerator()

    def update_performance(self, correct, total):
        accuracy = correct / total if total else 0
        if accuracy >= 0.85:
            self.performance_level = "advanced"
        elif accuracy >= 0.60:
            self.performance_level = "intermediate"
        else:
            self.performance_level = "beginner"

    def get_prompt_batch(self, batch_size=5):
        prompts = []

        # 🧠 Always prioritize weak prompts first
        weak = self.memory.get_weak_glosses()
        weak_prompts_to_use = weak[:batch_size]

        if weak_prompts_to_use:
            print(f"🔁 Reinforcing {len(weak_prompts_to_use)} weak prompt(s):")
            for prompt in weak_prompts_to_use:
                print(f"   - {prompt}")
            prompts.extend(weak_prompts_to_use)

        # 🎯 Fill remaining slots with new prompts
        remaining = batch_size - len(prompts)
        if remaining > 0:
            new_prompts = self.prompt_generator.generate_prompts(
                difficulty=self.performance_level,
                count=remaining
            )
            prompts.extend(new_prompts)

        return prompts
