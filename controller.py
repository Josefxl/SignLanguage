from agents.captioner import Captioner
from agents.tutor import AdaptiveTutorAgent as TutorAgent
from agents.monitor import BehaviorMonitorAgent
from agents.rl_environment import TeachingEnvironment
from agents.rl_agent import RLAgent

from sentence_transformers import SentenceTransformer, util
from langchain.memory import ConversationBufferMemory

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load semantic similarity model
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
        self.rl_agent = RLAgent(self.env)
        self.prompts = [self.tutor.get_prompt() for _ in range(5)]
        self.correct = 0
        self.skipped = 0
        self.session_log = []
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    def run_lesson(self):
        print("📘 Starting full sign language lesson...\n")

        for expected in self.prompts:
            print(f"Tutor says: Please sign – '{expected}'")
            gloss = input("✋ Please enter your gloss: ").strip()

            if not gloss:
                self.skipped += 1
                print("⏳ Skipped – no sign detected.\n")
                self.memory.chat_memory.add_user_message("(no gloss)")
                self.memory.chat_memory.add_ai_message("(skipped)")
                self.session_log.append({
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
                self.correct += 1
                print(f"✅ Great job! (Semantic Similarity: {score:.2f})\n")
                self.session_log.append({
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
                        self.correct += 1
                        print(f"✅ Great job! (Semantic Similarity: {score:.2f})\n")
                        self.session_log.append({
                            "prompt": expected,
                            "gloss": gloss,
                            "caption": caption,
                            "similarity": score,
                            "result": "Correct (Retry)"
                        })
                        continue
                    else:
                        print("❌ Still incorrect. Let's move on.\n")

                self.session_log.append({
                    "prompt": expected,
                    "gloss": gloss,
                    "caption": caption,
                    "similarity": score,
                    "result": "Incorrect"
                })

        total = len(self.prompts)
        print("\n🎓 Lesson Summary:")
        print(f"Correct: {self.correct}")
        print(f"Skipped: {self.skipped}")
        print(f"Total: {total}")
        print(f"Score: {(self.correct / total) * 100:.1f}%")

        # RL-driven performance adaptation
        new_level = self.rl_agent.act_and_learn(self.correct, total)
        print(f"\n🧠 RL Agent selected new difficulty level: {new_level}")

        print("\n📘 Prompt-by-Prompt Feedback:")
        for entry in self.session_log:
            print(f"- Prompt: {entry['prompt']}")
            print(f"  Gloss: {entry['gloss']}")
            print(f"  Caption: {entry['caption']}")
            print(f"  Similarity: {entry['similarity']:.2f}")
            print(f"  Result: {entry['result']}\n")

        state, advice = self.monitor.analyze_behavior(
            self.correct,
            total - self.correct - self.skipped,
            self.skipped
        )
        print("\n📊 Behavior Monitor Feedback:")
        print(f"State: {state}")
        print(f"Advice: {advice}")

        print("\n🧠 Memory Trace:")
        for m in self.memory.chat_memory.messages:
            print(f"{m.type.capitalize()}: {m.content}")
