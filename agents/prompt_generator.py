import random

class PromptLLMGenerator:
    def __init__(self, llm=None, use_llm=False):
        self.use_llm = use_llm  # Not used for now, but left for compatibility
        self.hardcoded_prompts = {
            "beginner": [
                "Hello",
                "I am hungry",
                "Help me please",
                "Thank you",
                "Where are you?"
            ],
            "intermediate": [
                "Can I borrow a pencil?",
                "Excuse me, where is the bus stop?",
                "I need help with homework.",
                "How much does it cost?",
                "What time is lunch?"
            ],
            "advanced": [
                "I believe education changes lives.",
                "What do you think about friendship?",
                "Sometimes I feel overwhelmed by my emotions.",
                "I'm always learning new things.",
                "My heart aches for those experiencing hardships."
            ]
        }

    def get_hardcoded_prompts(self, difficulty, count=5):
        prompts = self.hardcoded_prompts.get(difficulty, [])
        return random.sample(prompts, k=min(count, len(prompts)))

    def generate_prompts(self, difficulty="beginner", count=5):
        print(f"\n📚 Generating {count} '{difficulty}' prompts from hardcoded list...")
        prompts = self.get_hardcoded_prompts(difficulty, count)
        print("✅ Prompts:", prompts)
        return prompts
