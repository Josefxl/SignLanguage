import random

class RLAgent:
    def __init__(self, environment):
        self.env = environment
        self.q_table = {}  # In-memory Q-table for now
        self.epsilon = 0.2  # Exploration rate
        self.alpha = 0.5    # Learning rate
        self.gamma = 0.9    # Discount factor

    def get_state_key(self, state):
        return f"{state['difficulty_level']}_{state['consecutive_successes']}_{state['consecutive_failures']}"

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon or state_key not in self.q_table:
            return random.choice(self.env.get_available_actions())
        return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def update_q_table(self, state, action, reward, new_state):
        state_key = self.get_state_key(state)
        new_key = self.get_state_key(new_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.env.get_available_actions()}
        if new_key not in self.q_table:
            self.q_table[new_key] = {a: 0.0 for a in self.env.get_available_actions()}

        old_value = self.q_table[state_key][action]
        future_reward = max(self.q_table[new_key].values())
        new_value = old_value + self.alpha * (reward + self.gamma * future_reward - old_value)
        self.q_table[state_key][action] = new_value

    def act_and_learn(self, correct, total):
        state = self.env.get_state()
        action = self.choose_action(state)
        new_difficulty = self.env.apply_action(action)
        reward = self.env.compute_reward(correct, total)
        new_state = self.env.get_state()
        self.update_q_table(state, action, reward, new_state)
        return new_difficulty
