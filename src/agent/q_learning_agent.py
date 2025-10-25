import numpy as np
import random
from src.agent.q_table import QTable

class QLearningAgent:
    """
    Bộ não của Agent, thực thi logic Q-Learning.
    """
    def __init__(self, action_space_size, lr, gamma, epsilon, epsilon_decay, epsilon_min):
        self.action_space_size = action_space_size
        self.lr = lr             # Learning Rate (Alpha)
        self.gamma = gamma       # Discount Factor
        self.epsilon = epsilon   # Exploration Rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Agent sẽ quản lý Q-Table của chính nó
        self.q_table = QTable(action_space_size)

    def choose_action(self, state):
        """
        Chọn hành động bằng chiến lược Epsilon-Greedy.
        """
        # KHÁM PHÁ (Explore): Chọn 1 hành động ngẫu nhiên
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        # KHAI THÁC (Exploit): Chọn hành động tốt nhất (Q-value cao nhất)
        else:
            return np.argmax(self.q_table.get(state))

    def learn(self, state, action, reward, next_state):
        """
        Cập nhật Q-Table bằng công thức Q-Learning (Bellman).
        Q(s,a) = Q(s,a) + lr * [r + gamma * max(Q(s',a')) - Q(s,a)]
        """
        old_value = self.q_table.get(state)[action]
        next_max = np.max(self.q_table.get(next_state))
        
        # Công thức cập nhật Q-value
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        
        self.q_table.set(state, action, new_value)

    def update_epsilon(self):
        """Giảm tỷ lệ khám phá (epsilon) theo thời gian."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        self.q_table.save(filepath)

    def load_model(self, filepath):
        self.q_table.load(filepath)