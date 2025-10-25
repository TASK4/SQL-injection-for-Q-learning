import numpy as np
from collections import defaultdict
import json

class QTable:
    """
    Quản lý cấu trúc dữ liệu Q-Table (dạng dictionary).
    Q[state] = array[q_value_for_action_0, q_value_for_action_1, ...]
    """
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size
        # Dùng defaultdict để tự động thêm state mới với giá trị 0
        self.table = defaultdict(lambda: np.zeros(self.action_space_size))

    def get(self, state):
        """Lấy mảng Q-values cho một state. Tự động tạo nếu chưa có."""
        return self.table[state]

    def set(self, state, action_index, value):
        """Cập nhật Q-value cho một cặp (state, action)."""
        self.table[state][action_index] = value

    def save(self, filepath):
        """Lưu Q-Table ra file JSON."""
        # Chuyển defaultdict thành dict thường để lưu
        saveable_table = {k: list(v) for k, v in self.table.items()}
        with open(filepath, 'w') as f:
            json.dump(saveable_table, f)
        print(f"\n[QTable] Đã lưu model vào: {filepath}")

    def load(self, filepath):
        """Tải Q-Table từ file JSON."""
        with open(filepath, 'r') as f:
            loaded_table = json.load(f)
        
        # Nạp lại vào defaultdict
        self.table = defaultdict(lambda: np.zeros(self.action_space_size))
        for state, values in loaded_table.items():
            self.table[state] = np.array(values)
        print(f"\n[QTable] Đã tải model từ: {filepath}")