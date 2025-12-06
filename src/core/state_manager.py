import numpy as np

class StateManager:
    def __init__(self):
        self.current_state = ""
        # 0: None, 1: Start/End, 2: Logic, 3: Col, 4: NULL, 5: Symbol
        self.last_action_type = 0 

    def reset_state(self):
        self.current_state = ""
        self.last_action_type = 0
        return self.get_feature_vector()

    def update_state(self, action_string, action_index):
        s_action = action_string.upper().strip()
        
        # 1. Logic nối chuỗi thông minh
        if action_string.startswith(",") or action_string.startswith("--") or action_string.startswith(")"):
            self.current_state += action_string
        else:
            if self.current_state == "" or self.current_state.endswith(" "):
                self.current_state += action_string
            else:
                self.current_state += " " + action_string

        # 2. Phân loại mảnh ghép (Puzzle Piece Type)
        if "A'))" in s_action: self.last_action_type = 1
        elif any(k in s_action for k in ["UNION", "SELECT", "FROM"]): self.last_action_type = 2
        elif s_action in ["ID", "EMAIL", "PASSWORD"]: self.last_action_type = 3
        elif "NULL" in s_action: self.last_action_type = 4
        elif "," in s_action or "--" in s_action: self.last_action_type = 5
        else: self.last_action_type = 0

        return self.get_feature_vector()

    def get_feature_vector(self):
        s = self.current_state.upper()
        
        # --- FEATURE SET ĐƯỢC TỐI ƯU CHO TRANSFER LEARNING ---
        
        # 1. Trạng thái NULL: Quan trọng để biết có đang "dò cột" hay không
        # Đếm số lượng NULL trong cụm cuối cùng (sau chữ SELECT gần nhất)
        try:
            last_select_idx = s.rfind("SELECT")
            if last_select_idx != -1:
                relevant_part = s[last_select_idx:]
                consecutive_nulls = relevant_part.count("NULL")
            else:
                consecutive_nulls = 0
        except:
            consecutive_nulls = 0
            
        # Normalize nhẹ (chia 10) để mạng dễ học
        null_feature = min(consecutive_nulls / 10.0, 1.0)
        
        # 2. Các cờ hiệu ngữ pháp (Syntax Flags)
        has_union = 1.0 if "UNION" in s else 0.0
        has_select = 1.0 if "SELECT" in s else 0.0
        has_from = 1.0 if "FROM" in s else 0.0
        
        # 3. Last Action (One-hot): Mảnh ghép vừa đặt xuống là gì?
        # Đây là feature quan trọng nhất để nó biết mảnh tiếp theo nên là gì
        last_action_vec = [0.0] * 6
        if 0 <= self.last_action_type <= 5:
            last_action_vec[self.last_action_type] = 1.0
            
        # 4. Kiểm tra đóng mở (Structure integrity)
        has_closed_prefix = 1.0 if "A'))" in s else 0.0
        has_comment_suffix = 1.0 if "--" in s else 0.0

        # Note: Đã BỎ feature 'length' vì độ dài payload giữa Mock và Target khác nhau.
        # Transfer Learning sẽ fail nếu dựa vào độ dài.

        return tuple([
            null_feature,
            has_union,
            has_select,
            has_from,           # Thêm cái này
            has_closed_prefix,
            has_comment_suffix
        ] + last_action_vec)