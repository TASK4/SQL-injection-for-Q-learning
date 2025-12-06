import numpy as np

class StateManager:
    def __init__(self):
        self.current_state = ""
        self.step_count = 0 
        self.last_action_type = 0 
        self.last_feedback = 0 # 0: Normal, 1: Syntax Error, 2: Column Mismatch

    def reset_state(self):
        self.current_state = ""
        self.step_count = 0
        self.last_action_type = 0
        self.last_feedback = 0
        return self.get_feature_vector()

    def update_feedback(self, error_text):
        """Cập nhật trạng thái dựa trên phản hồi của DB"""
        if "COLUMN_MISMATCH" in error_text:
            self.last_feedback = 2
        elif "SYNTAX_ERROR" in error_text:
            self.last_feedback = 1
        else:
            self.last_feedback = 0

    def update_state(self, action_string, action_index):
        s_action = action_string.upper().strip()
        
        # Logic nối chuỗi cũ của bạn giữ nguyên, chỉ chỉnh nhẹ để tránh spam khoảng trắng
        if action_string.startswith(",") or action_string.startswith("--") or action_string.startswith(")"):
            self.current_state += action_string
        else:
            if self.current_state == "" or self.current_state.endswith(" "):
                self.current_state += action_string
            else:
                self.current_state += " " + action_string

        # Phân loại action (giữ nguyên logic tốt của bạn)
        if "A'))" in s_action: self.last_action_type = 1
        elif any(k in s_action for k in ["UNION", "SELECT", "FROM"]): self.last_action_type = 2
        elif s_action in ["ID", "EMAIL", "PASSWORD"]: self.last_action_type = 3
        elif "NULL" in s_action: self.last_action_type = 4
        elif "," in s_action or "--" in s_action: self.last_action_type = 5
        else: self.last_action_type = 0
        
        self.step_count += 1
        # Lưu ý: Không trả về feature vector ngay ở đây nữa, vì chưa có feedback từ DB
        return None 

    def get_feature_vector(self):
        s = self.current_state.upper()
        
        # 1. Các Checkpoint quan trọng (như xếp chai nước)
        has_entry = 1 if s.startswith("A'))") else 0
        has_union_select = 1 if "UNION" in s and "SELECT" in s else 0
        has_from = 1 if "FROM" in s else 0
        
        # 2. Đếm số lượng NULL (quan trọng cho Transfer Learning dò cột)
        # Chúng ta bucket nó lại: 0, 1, 2, 3, >3 để giảm state space
        null_count = s.count("NULL")
        if null_count >= 3: null_feat = 3
        else: null_feat = null_count
        
        # 3. Last Action (One-hot encoding)
        last_action_vec = [0] * 6
        if 0 <= self.last_action_type <= 5:
            last_action_vec[self.last_action_type] = 1

        # 4. Feedback từ môi trường (CỰC QUAN TRỌNG)
        # Giúp AI biết: "À, trạng thái này là đang bị lệch cột, mình phải thêm NULL"
        feedback_vec = [0] * 3
        feedback_vec[self.last_feedback] = 1
        
        # Loại bỏ step_feature float, thay bằng bucket đơn giản
        # 0: Mới vào, 1: Giữa game, 2: Sắp hết lượt
        if self.step_count < 5: step_phase = 0
        elif self.step_count < 15: step_phase = 1
        else: step_phase = 2
        step_vec = [0]*3
        step_vec[step_phase] = 1

        # Tổng hợp lại thành tuple (hashable cho Q-Table)
        return tuple(
            [has_entry, has_union_select, has_from, null_feat] + 
            last_action_vec + 
            feedback_vec + 
            step_vec
        )