import numpy as np

class StateManager:
    def __init__(self):
        self.current_state = ""
        # Định nghĩa các loại hành động để AI hiểu ngữ cảnh
        # 0: None, 1: Start/End, 2: Logic (UNION/SELECT), 3: Data (Cột), 4: Value (NULL), 5: Symbol (Phẩy/Comment)
        self.last_action_type = 0 

    def reset_state(self):
        self.current_state = ""
        self.last_action_type = 0
        return self.get_feature_vector()

    def update_state(self, action_string, action_index):
        """
        Cập nhật chuỗi và tự động phân loại hành động dựa trên nội dung text.
        """
        s_action = action_string.upper().strip()
        
        # 1. CẬP NHẬT CHUỖI QUERY (Logic nối chuỗi thông minh)
        # Nếu là dấu phẩy hoặc comment thì nối liền, còn lại thêm dấu cách
        if action_string.startswith(",") or action_string.startswith("--") or action_string.startswith(")"):
            self.current_state += action_string
        else:
            # Nếu state chưa có gì hoặc kết thúc bằng khoảng trắng thì không thêm cách
            if self.current_state == "" or self.current_state.endswith(" "):
                self.current_state += action_string
            else:
                self.current_state += " " + action_string

        # 2. PHÂN LOẠI HÀNH ĐỘNG (Dựa trên nội dung thay vì index)
        # Loại 1: Start (a')) )
        if "A'))" in s_action:
            self.last_action_type = 1
            
        # Loại 2: Logic SQL (UNION, SELECT, FROM)
        elif "UNION" in s_action or "SELECT" in s_action or "FROM" in s_action:
            self.last_action_type = 2
            
        # Loại 3: Columns (id, email, password)
        elif s_action in ["ID", "EMAIL", "PASSWORD"]:
            self.last_action_type = 3
            
        # Loại 4: Values (NULL)
        elif "NULL" in s_action:
            self.last_action_type = 4
            
        # Loại 5: Symbols (Dấu phẩy, Comment --)
        elif "," in s_action or "--" in s_action:
            self.last_action_type = 5
            
        else:
            self.last_action_type = 0

        return self.get_feature_vector()

    def get_feature_vector(self):
        """
        Biến đổi trạng thái hiện tại thành vector số để Neural Network hiểu.
        """
        s = self.current_state.upper()
        
        # Feature 1: Đếm số NULL (Quan trọng cho Phase 2)
        null_count = s.count("NULL")
        
        # Feature 2: Có UNION / SELECT chưa?
        has_union = 1.0 if "UNION" in s else 0.0
        has_select = 1.0 if "SELECT" in s else 0.0
        
        # Feature 3: Độ dài chuỗi (Chuẩn hóa nhẹ)
        length_norm = min(len(s) / 100.0, 1.0)
        
        # Feature 4: Loại hành động vừa đi (Quan trọng cho Phase 3 - Sắp xếp)
        # One-hot encoding cho last_action_type (0-5)
        last_action_vec = [0.0] * 6
        if 0 <= self.last_action_type <= 5:
            last_action_vec[self.last_action_type] = 1.0
            
        # Feature 5: Đã đóng ngoặc đúng chưa? (Check a'))
        has_closed = 1.0 if "A'))" in s else 0.0
        
        # Feature 6: Đã comment chưa?
        has_comment = 1.0 if "--" in s else 0.0

        return tuple([
            float(null_count) / 10.0, # Chuẩn hóa số NULL
            has_union,
            has_select,
            length_norm,
            has_closed,
            has_comment
        ] + last_action_vec)