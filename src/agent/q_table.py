import pickle
import numpy as np

class QTable:
    def __init__(self, action_size):
        self.q_table = {} # Key: State Tuple, Value: List of Q-values
        self.action_size = action_size

    def _get_state_key(self, state):
        """
        Chuyển đổi State thành Key an toàn cho Dictionary.
        Làm tròn số thực đến 2 chữ số thập phân để gộp các trạng thái tương tự nhau.
        """
        if isinstance(state, (list, np.ndarray)):
            # Làm tròn từng phần tử trong vector state
            return tuple(round(x, 2) if isinstance(x, (float, np.floating)) else x for x in state)
        return state

    def get_q_value(self, state, action):
        state_key = self._get_state_key(state)
        # Nếu chưa gặp state này bao giờ, trả về 0.0 (hoặc số dương nhỏ để khuyến khích tò mò)
        if state_key not in self.q_table:
            return 0.0 
        return self.q_table[state_key][action]

    def get_max_q(self, state):
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            return 0.0
        return max(self.q_table[state_key])

    def update_q_value(self, state, action, value):
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            # Khởi tạo row mới với toàn số 0
            self.q_table[state_key] = [0.0] * self.action_size
        
        self.q_table[state_key][action] = value

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Loaded Q-table from {filename}")
        except FileNotFoundError:
            print("No existing Q-table found. Starting fresh.")