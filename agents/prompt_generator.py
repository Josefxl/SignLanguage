from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import re

class PromptLLMGenerator:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.9)

    def generate_prompts(self, difficulty="beginner", count=5):
        instruction_map = {
            "beginner": "You are a tutor agent, part of a personalised system, you are teaching students how to use sign language, your job is to generate sentences that are around 4 to 5 words, these should be easy day to day conversational sentences for the user to practice with, be as free and random with your sentences as you want and they should have variety, but they should make logical sense, and make sure to avoid duplicates",
            "intermediate": "You are a tutor agent, part of a personalised system, you are teaching students how to use sign language, your job is to generate sentences that are around 6 to 8 words, these should be sentences you would find in normal social conversations for the user to practice with, be as free and random with your sentences as you want and they should have variety, but they should make logical sense, and make sure to avoid duplicates",
            "advanced": "You are a tutor agent, part of a personalised system, you are teaching students how to use sign language, your job is to generate sentences that are around 9 to 11 words, these should be slightly advanced things like conversations in a classroom for the user to practice with, be as free and random with your sentences as you want and they should have variety, but they should make logical sense, and make sure to avoid duplicates.'"
        }

        system_prompt = instruction_map[difficulty] + f"\nProvide exactly {count} unique sentences in the format:\n1. Sentence\n2. Sentence\n..."

        try:
            response = self.llm([HumanMessage(content=system_prompt)])
            text = response.content.strip()
            lines = re.findall(r"\d+\.\s*(.+)", text)

            unique = []
            seen = set()
            word_count_ranges = {
                "beginner": (4, 5),
                "intermediate": (6, 8),
                "advanced": (9, 11)
            }
            min_words, max_words = word_count_ranges[difficulty]

            for line in lines:
                sentence = line.strip()
                word_count = len(sentence.split())
                if sentence.lower() not in seen and min_words <= word_count <= max_words:
                    seen.add(sentence.lower())
                    unique.append(sentence)
                if len(unique) >= count:
                    break

            # Fallback
            while len(unique) < count:
                unique.append(f"{difficulty.capitalize()} placeholder sentence.")

            return unique
        except Exception as e:
            print(f"Prompt generation failed: {e}")
            return [f"{difficulty.capitalize()} placeholder sentence." for _ in range(count)]