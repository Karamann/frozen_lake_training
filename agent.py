import numpy as np


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, alpha, gamma):
        current_q = self.q_table[state, action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + alpha * (reward + gamma * next_max_q - current_q)
        self.q_table[state, action] = new_q
