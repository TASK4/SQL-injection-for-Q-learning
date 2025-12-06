# src/core/state_manager.py
import numpy as np

class StateManager:
    def __init__(self):
        self.current_state = ""       # Chuỗi SQL hiện tại
        self.last_action = "START"    # Hành động vừa thực hiện
        self.feedback = "NORMAL"      # Phản hồi từ server (Lỗi/Thành công)
        
        # Không dùng SentenceTransformer nữa để tránh trạng thái bị unique hóa

    def reset_state(self):
        self.current_state = ""
        self.last_action = "START"
        self.feedback = "NORMAL"
        return self.get_feature_vector()

    def update_state(self, action_string, action_index):
        # 1. Cập nhật câu lệnh SQL
        # Xử lý logic nối chuỗi cho đẹp (tùy chọn)
        if self.current_state == "":
            self.current_state = action_string
        else:
            self.current_state += " " + action_string
            
        # 2. Lưu hành động vừa làm để làm "manh mối" cho State
        self.last_action = action_string.strip()

    def update_feedback(self, feedback_text):
        # Đơn giản hóa Feedback thành các Tag ngắn gọn
        self.feedback = "NORMAL"
        
        feedback_lower = feedback_text.lower()
        
        if "column_mismatch" in feedback_text: # Ưu tiên lỗi này
            self.feedback = "MISMATCH"
        elif "syntax_error" in feedback_text or "syntax" in feedback_lower:
            self.feedback = "SYNTAX"
        elif "internal_error" in feedback_text:
            self.feedback = "ERROR"
        # Nếu status 200 (không lỗi) và không có mismatch -> Tạm coi là OK
        elif feedback_text.startswith('[') or "admin" in feedback_lower: # JSON hoặc admin
            self.feedback = "SUCCESS_CANDIDATE"

    def get_feature_vector(self):
        """
        Thay vì trả về Vector số thực (khiến Q-Table bị loạn),
        ta trả về một CHUỖI ĐẠI DIỆN (Discrete State Key).
        
        State = "Hành động vừa xong" + "Tình trạng hiện tại"
        Ví dụ: "NULL|MISMATCH" -> Agent sẽ học là cần thêm dấu phẩy.
               "NULL|SUCCESS_CANDIDATE" -> Agent sẽ học là cần thêm 'FROM Users'.
        """
        s = self.current_state.upper()
        null_count = s.count("NULL")
        
        if null_count >= 10: 
            null_feat = 10
        else: 
            null_feat = null_count
        
        state_key = f"{self.last_action}|{self.feedback}"
        
        # Q-Table cần key là string hoặc tuple, ta trả về string cho dễ
        return state_key