from agents.captioner import Captioner
from agents.tutor import AdaptiveTutorAgent as TutorAgent
from agents.monitor import BehaviorMonitorAgent
from agents.rl_agent import RLAgent
from agents.rl_environment import TeachingEnvironment
from agents.logger import SessionLogger
from sentence_transformers import SentenceTransformer, util
from langchain.memory import ConversationBufferMemory
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Semantic similarity model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def is_semantically_similar(expected: str, actual: str, threshold: float = 0.75):
    embeddings = semantic_model.encode([expected, actual], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return score >= threshold, score


class ControllerAgent:
    def __init__(self):
        self.captioner = Captioner()
        self.tutor = TutorAgent()
        self.monitor = BehaviorMonitorAgent()
        self.env = TeachingEnvironment()
        self.rl_agent = RLAgent(environment=self.env)
        self.logger = SessionLogger()
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    def run_lesson(self, prompts=None):
        if prompts is None:
            prompts = self.tutor.get_prompt_batch()

        correct = 0
        skipped = 0
        session_log = []

        print("\n📘 Starting full sign language lesson...\n")

        for expected in prompts:
            print(f"Tutor says: Please sign – '{expected}'")
            gloss = input("✋ Please enter your gloss: ").strip()

            if not gloss:
                skipped += 1
                print("⏳ Skipped – no sign detected.\n")
                self.memory.chat_memory.add_user_message("(no gloss)")
                self.memory.chat_memory.add_ai_message("(skipped)")
                session_log.append({
                    "prompt": expected,
                    "gloss": "(none)",
                    "caption": "(skipped)",
                    "similarity": 0.0,
                    "result": "Skipped"
                })
                continue

            caption = self.captioner.caption(gloss)
            print(f"📝 Captioner translated: {caption}")
            self.memory.chat_memory.add_user_message(gloss)
            self.memory.chat_memory.add_ai_message(caption)

            similar, score = is_semantically_similar(expected, caption)

            if similar:
                correct += 1
                print(f"✅ Great job! (Similarity: {score:.2f})\n")
                session_log.append({
                    "prompt": expected,
                    "gloss": gloss,
                    "caption": caption,
                    "similarity": score,
                    "result": "Correct"
                })
            else:
                print(f"❌ Incorrect. Expected: '{expected}' (Similarity: {score:.2f})")
                retry = input("❓ Would you like to try that again? (y/n): ").strip().lower()

                if retry == "y":
                    gloss = input("✋ Try again – enter your gloss: ").strip()
                    caption = self.captioner.caption(gloss)
                    print(f"📝 Captioner translated: {caption}")
                    self.memory.chat_memory.add_user_message(gloss)
                    self.memory.chat_memory.add_ai_message(caption)
                    similar, score = is_semantically_similar(expected, caption)
                    if similar:
                        correct += 1
                        print(f"✅ Great job! (Similarity: {score:.2f})\n")
                        session_log.append({
                            "prompt": expected,
                            "gloss": gloss,
                            "caption": caption,
                            "similarity": score,
                            "result": "Correct (Retry)"
                        })
                        continue
                    else:
                        print("❌ Still incorrect. Let's move on.\n")

                session_log.append({
                    "prompt": expected,
                    "gloss": gloss,
                    "caption": caption,
                    "similarity": score,
                    "result": "Incorrect"
                })

        total = len(prompts)
        score_percent = (correct / total) * 100 if total else 0.0

        # 🎓 Summary
        print("\n🎓 Lesson Summary:")
        print(f"Correct: {correct}")
        print(f"Skipped: {skipped}")
        print(f"Total: {total}")
        print(f"Score: {score_percent:.1f}%")

        # 🧠 RL Agent adapts difficulty
        new_level = self.rl_agent.act_and_learn(correct, total)
        self.tutor.performance_level = new_level
        print(f"\n🧠 RL Agent selected new difficulty level: {new_level}")

        # 📘 Prompt Feedback
        print("\n📘 Prompt-by-Prompt Feedback:")
        for entry in session_log:
            print(f"- Prompt: {entry['prompt']}")
            print(f"  Gloss: {entry['gloss']}")
            print(f"  Caption: {entry['caption']}")
            print(f"  Similarity: {entry['similarity']:.2f}")
            print(f"  Result: {entry['result']}\n")

        # 📊 Behavior Feedback
        state, advice = self.monitor.analyze_behavior(
            correct, total - correct - skipped, skipped
        )
        print("\n📊 Behavior Monitor Feedback:")
        print(f"State: {state}")
        print(f"Advice: {advice}")

        # 💾 Save session log
        self.logger.log_session(prompts, session_log, score_percent)

        # 🧠 Update memory
        self.tutor.memory.update(session_log)

        # 🔁 Ask if user wants to continue
        user_choice = input("\n🔁 Would you like to continue with a new set of prompts? (y/n): ").strip().lower()
        if user_choice == "y":
            new_prompts = self.tutor.get_prompt_batch()
            self.run_lesson(prompts=new_prompts)
        else:
            print("\n👋 Ending session. Great job!\n")
