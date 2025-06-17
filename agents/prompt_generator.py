from transformers import pipeline

class PromptLLMGenerator:
    def __init__(self, llm=None):
        self.generator = llm or pipeline("text-generation", model="google/flan-t5-large")

    def generate_prompts(self, difficulty="beginner", count=5):
        system_instruction = {
            "beginner": "Generate a list of very simple sign language practice sentences (1-5 words) useful for daily communication.",
            "intermediate": "Generate a list of moderately complex sign language practice sentences (5-10 words) with some variety in structure.",
            "advanced": "Generate a list of complex or abstract sign language sentences (10-15 words) involving opinions, reasoning, or emotions."
        }[difficulty]

        prompt = f"{system_instruction} Provide exactly {count} unique and practical examples."
        result = self.generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]['generated_text']

        # Very basic parsing assuming numbered or bulleted output
        lines = result.splitlines()
        cleaned = [l.strip("-•1234567890. ") for l in lines if len(l.strip()) > 5]
        return cleaned[:count]
