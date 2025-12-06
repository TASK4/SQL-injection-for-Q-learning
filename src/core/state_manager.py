import numpy as np

class StateManager:
    def __init__(self):
        self.current_state = ""
        self.step_count = 0 
        self.last_action_type = 0 

    def reset_state(self):
        self.current_state = ""
        self.step_count = 0
        self.last_action_type = 0
        return self.get_feature_vector()

    def update_state(self, action_string, action_index):
        s_action = action_string.upper().strip()
        
        # Nối chuỗi
        if action_string.startswith(",") or action_string.startswith("--") or action_string.startswith(")"):
            self.current_state += action_string
        else:
            if self.current_state == "" or self.current_state.endswith(" "):
                self.current_state += action_string
            else:
                self.current_state += " " + action_string

        # Phân loại action
        if "A'))" in s_action: self.last_action_type = 1
        elif any(k in s_action for k in ["UNION", "SELECT", "FROM"]): self.last_action_type = 2
        elif s_action in ["ID", "EMAIL", "PASSWORD"]: self.last_action_type = 3
        elif "NULL" in s_action: self.last_action_type = 4
        elif "," in s_action or "--" in s_action: self.last_action_type = 5
        else: self.last_action_type = 0
        
        self.step_count += 1
        return self.get_feature_vector()

    def get_feature_vector(self):
        s = self.current_state.upper()
        
        has_entry = 1.0 if s.startswith("A'))") else 0.0
        has_structure = 1.0 if "UNION" in s and "SELECT" in s else 0.0
        has_from = 1.0 if "FROM" in s else 0.0
        
        # Đếm NULL
        try:
            last_select_idx = s.rfind("SELECT")
            if last_select_idx != -1:
                relevant_part = s[last_select_idx:]
                consecutive_nulls = relevant_part.count("NULL")
            else:
                consecutive_nulls = 0
        except:
            consecutive_nulls = 0
        null_feature = min(consecutive_nulls / 10.0, 1.0)
        
        # Action One-hot
        last_action_vec = [0.0] * 6
        if 0 <= self.last_action_type <= 5:
            last_action_vec[self.last_action_type] = 1.0

        # MỚI: Tiến độ thời gian (Quan trọng)
        step_feature = min(self.step_count / 20.0, 1.0)

        return tuple([
            has_entry,
            has_structure,
            has_from,
            null_feature,
            step_feature # <-- Đừng quên cái này
        ] + last_action_vec)