import numpy as np
import random
from src.agent.q_table import QTable
import pickle

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
        # Epsilon-Greedy Strategy
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: Chọn bừa để tìm đường mới
            return random.randint(0, self.action_space_size - 1)
        else:
            # Exploitation: Chọn hành động ngon nhất hiện tại
            # Lấy dòng Q-values của state hiện tại
            state_key = self.q_table._get_state_key(state)
            if state_key not in self.q_table.q_table:
                return random.randint(0, self.action_space_size - 1)
            
            # Lấy index của action có điểm cao nhất
            qs = self.q_table.q_table[state_key]
            # Nếu nhiều hành động điểm bằng nhau, chọn random trong số đó (để đỡ bị kẹt vòng lặp)
            max_val = max(qs)
            best_actions = [i for i, val in enumerate(qs) if val == max_val]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state):
        # 1. Lấy Q-value hiện tại
        current_q = self.q_table.get_q_value(state, action)
        
        # 2. Lấy Max Q-value của trạng thái tiếp theo
        max_next_q = self.q_table.get_max_q(next_state)
        
        # 3. Công thức cập nhật Bellman
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        
        # 4. Lưu lại
        self.q_table.update_q_value(state, action, new_q)

    def update_epsilon(self):
        """Giảm tỷ lệ khám phá (epsilon) theo thời gian."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Lưu Q-table xuống file dùng pickle"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.q_table, f)
            print(f"[Agent] Đã lưu model vào: {filepath}")
        except Exception as e:
            print(f"[Agent] Lỗi khi lưu model: {e}")

    def load_model(self, filepath):
        """Load Q-table từ file dùng pickle"""
        try:
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"[Agent] Đã tải model từ: {filepath}")
        except Exception as e:
            print(f"[Agent] Lỗi khi tải model: {e}")

