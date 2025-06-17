import random
from agents.reinforce_memory import ReinforcementMemory
from agents.prompt_generator import PromptLLMGenerator

class AdaptiveTutorAgent:
    def __init__(self, llm=None):
        self.performance_level = "beginner"
        self.memory = ReinforcementMemory()
        self.prompt_generator = PromptLLMGenerator(llm=llm)

    def update_performance(self, correct, total):
        accuracy = correct / total if total else 0
        if accuracy >= 0.85:
            self.performance_level = "advanced"
        elif accuracy >= 0.60:
            self.performance_level = "intermediate"
        else:
            self.performance_level = "beginner"

    def get_prompt_batch(self, batch_size=5):
        # Prioritize weak glosses for reinforcement
        weak = self.memory.get_weak_glosses()
        prompts = []

        if weak:
            prompts.extend(random.sample(weak, min(len(weak), batch_size)))

        if len(prompts) < batch_size:
            new_prompts = self.prompt_generator.generate_prompts(
                difficulty=self.performance_level,
                count=batch_size - len(prompts)
            )
            prompts.extend(new_prompts)

        return prompts
