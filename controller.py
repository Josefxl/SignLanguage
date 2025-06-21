import os
import json
import datetime
from agents.captioner import Captioner
from agents.tutor import AdaptiveTutorAgent
from agents.monitor import BehaviorMonitorAgent
from sentence_transformers import SentenceTransformer, util
from agents.reinforce_memory import ReinforcementMemory
from langchain.memory import ConversationBufferMemory

# Semantic similarity model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def is_semantically_similar(expected: str, actual: str, threshold: float = 0.75):
    embeddings = semantic_model.encode([expected, actual], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return score >= threshold, score

class ControllerAgent:
    def __init__(self):
        self.captioner = Captioner()
        self.tutor = AdaptiveTutorAgent()
        self.monitor = BehaviorMonitorAgent()
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

        # Always clear LLM memory to prevent semantic repetition
        self.memory.clear()
        print("🧠 Chat memory cleared.")

        # Ask about reinforcement memory only
        clear_reinforcement = input("🧪 Also clear reinforcement memory (weak/strong prompts)? (y/n): ").lower()
        if clear_reinforcement == "y":
            self.tutor.memory.clear_memory()
            print("🧽 Reinforcement memory cleared.\n")

    def run_lesson(self):
        batch = self.tutor.get_prompt_batch()
        results = []

        print("\n📘 Starting full sign language lesson...\n")

        for prompt in batch:
            print(f"Tutor says: Please sign – '{prompt}'")
            gloss = input("✋ Please enter your gloss: ")

            caption = self.captioner.caption(gloss)
            print(f"📝 Captioner translated: {caption}")

            similar, similarity_score = is_semantically_similar(prompt, caption)
            correct = similar

            results.append({
                "prompt": prompt,
                "gloss": gloss,
                "caption": caption,
                "similarity": round(similarity_score, 2),
                "result": "Correct" if correct else "Incorrect"
            })

        correct_count = sum(1 for r in results if r["result"] == "Correct")
        skipped_count = sum(1 for r in results if r["gloss"].strip() == "")
        total = len(results)
        score = round(100 * correct_count / total, 1) if total else 0
        incorrect_count = total - correct_count - skipped_count  # ✅ Added this

        # Save feedback
        self.tutor.memory.update(results)
        self.tutor.update_performance(correct_count, total)

        difficulty = self.tutor.performance_level
        state, behavior_action = self.monitor.analyze_behavior(correct_count, incorrect_count, skipped_count)  # ✅ Fixed variable use

        print(f"📊 Behavior State: {state}")
        print(f"{behavior_action}")  # ✅ Correct reference

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/session_{timestamp}.json"

        with open(log_path, "w") as f:
            json.dump({
                "results": results,
                "score": score,
                "difficulty": difficulty,
                "state": state,
                "behavior_action": behavior_action
            }, f, indent=2)

        print("\n🎓 Lesson Summary:")
        print(f"Correct: {correct_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Total: {total}")
        print(f"Score: {score}%")

        print(f"\n🧠 RL Agent selected new difficulty level: {difficulty}")
        print(f"🧠 Based on score: {correct_count}/{total}\n")

        print("📘 Prompt-by-Prompt Feedback:")
        for r in results:
            print(f"- Prompt: {r['prompt']}")
            print(f"  Gloss: {r['gloss']}")
            print(f"  Caption: {r['caption']}")
            print(f"  Similarity: {r['similarity']}")
            print(f"  Result: {r['result']}\n")

        print(f"📊 Behavior Monitor Action:\n{behavior_action}")
        print(f"\n💾 Session log saved to: {log_path}")

        cont = input("\n🔁 Would you like to continue with a new set of prompts? (y/n): ")
        if cont.lower() == "y":
            self.run_lesson()
        else:
            print("\n👋 Ending session. Great job!")
