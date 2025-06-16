import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BehaviorMonitorAgent:
    def __init__(self):
        logger.info("Behavior Monitor Agent initialized.")

    def analyze_behavior(self, correct, incorrect, skipped):
        total = correct + incorrect + skipped
        if total == 0:
            return "No activity detected.", "⚠️ Idle session. Please try a practice round."

        engagement_rate = (correct + incorrect) / total
        accuracy_rate = correct / total if total > 0 else 0
        skip_rate = skipped / total

        # Determine behavior state
        if engagement_rate < 0.4:
            state = "Disengaged"
            advice = "👋 Seems like the student is not engaging. Consider prompting them or suggesting a break."
        elif accuracy_rate < 0.4:
            state = "Confused"
            advice = "❓ The student seems confused. Try simpler signs or a quick review."
        elif skip_rate > 0.4:
            state = "Distracted"
            advice = "⏳ The student is skipping a lot. Ask if they need help or time."
        elif accuracy_rate > 0.8:
            state = "Excelling"
            advice = "🌟 Great progress! You can increase the difficulty or give praise."
        else:
            state = "Engaged"
            advice = "✅ Student is doing okay. Keep the session going."

        return state, advice

# Simulate example use
if __name__ == "__main__":
    monitor = BehaviorMonitorAgent()

    # Simulated session result from TutorAgent
    correct = int(input("Enter number of correct signs: "))
    incorrect = int(input("Enter number of incorrect signs: "))
    skipped = int(input("Enter number of skipped responses: "))

    state, recommendation = monitor.analyze_behavior(correct, incorrect, skipped)

    print("\n📊 Behavior Summary:")
    print(f"Behavior State: {state}")
    print(f"Recommendation: {recommendation}")
