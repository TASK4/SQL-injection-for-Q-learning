import numpy as np

class StateManager:
    def __init__(self):
        self.current_state = ""
        self.last_action_type = 0 # 0: None, 1: Prefix, 2: Keyword, 3: Payload...

    def reset_state(self):
        self.current_state = ""
        self.last_action_type = 0
        return self.get_feature_vector()

    def update_state(self, action_string, action_index):
        """
        Cập nhật chuỗi và lưu lại loại hành động vừa thực hiện.
        """
        # Logic nối chuỗi giữ nguyên như cũ của bạn
        if action_string.startswith(",") or action_string.startswith("--") or action_string.startswith(")"):
            self.current_state += action_string
        elif action_string.strip() in ["UNION", "SELECT", "FROM", "Users", "WHERE", "NULL"]:
            if not self.current_state.endswith(" "):
                self.current_state += " " + action_string
            else:
                self.current_state += action_string
        else:
            self.current_state += action_string
            
        # Phân loại action dựa trên index (Mapping theo ActionSpace của bạn)
        # Giả sử: 0-1: Prefix, 2-6: Keywords, 7-8: Counting, 9-11: Data, 12: Suffix
        if action_index <= 1: self.last_action_type = 1 # Prefix
        elif action_index <= 6: self.last_action_type = 2 # Keywords (UNION, SELECT...)
        elif action_index <= 8: self.last_action_type = 3 # Counting (, NULL)
        elif action_index <= 11: self.last_action_type = 4 # Data payload
        else: self.last_action_type = 5 # Suffix

        return self.get_feature_vector()

    def get_feature_vector(self):
        s = self.current_state.upper()
        return tuple([
            # ... Các feature cũ giữ nguyên ...
            1.0 if "UNION" in s else 0.0,
            1.0 if "SELECT" in s else 0.0,
            
            # --- FEATURE MỚI QUAN TRỌNG ---
            self.last_action_type / 5.0, # Context: Vừa làm cái gì?
            1.0 if s.strip().endswith("UNION") else 0.0, # Để tránh Union Union
            1.0 if s.strip().endswith(",") else 0.0,     # Để biết cần điền tiếp biến
        ])