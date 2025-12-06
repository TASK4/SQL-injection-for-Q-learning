import numpy as np

class RewardSystem:
    def __init__(self, normal_count, success_marker, error_marker, env_type='training'):
        self.env_type = env_type
        # KHUNG SƯỜN tham khảo
        self.skeleton_flow = ["A'))", "UNION", "SELECT", "FROM", "USERS", "--"]
        
    def calculate_reward(self, response, payload):
        # Xóa khoảng trắng để so sánh
        s_payload = payload.upper().replace(" ", "") 
        
        current_reward = -0.1 # Phạt nhẹ chi phí mỗi bước đi
        done = False
        
        # --- GIAI ĐOẠN 1: CHECKPOINT CỬA VÀO ---
        if s_payload.startswith("A'))"):
            current_reward += 1.0 
            
            # --- GIAI ĐOẠN 2: KHUNG SƯỜN ---
            if "UNION" in s_payload and "SELECT" in s_payload:
                current_reward += 2.0 
                
                # --- GIAI ĐOẠN 3: DÒ CỘT (Transfer Learning) ---
                if "COLUMN_MISMATCH" in response.text:
                    # Nếu bị lệch cột: Phạt RẤT NHẸ để nhắc nhở
                    current_reward -= 0.1 
                    # Thưởng ĐỘNG VIÊN nếu đang thêm NULL hoặc dấu phẩy
                    if payload.strip().upper().endswith("NULL") or payload.strip().endswith(","):
                        current_reward += 0.5
                        
                elif "SYNTAX_ERROR" not in response.text and response.status_code == 200:
                    # Đã SELECT thành công (đúng số cột)
                    current_reward += 5.0 
                    
                    # --- GIAI ĐOẠN 4: KHAI THÁC ---
                    if "FROM" in s_payload and "USERS" in s_payload:
                        current_reward += 20.0 # WIN
                        done = True
        else:
            # Chưa có cửa vào mà spam dài -> Phạt
            if len(payload) > 10:
                current_reward -= 1.0

        # Phạt lỗi cú pháp
        if "SYNTAX_ERROR" in response.text:
             current_reward -= 0.5

        return current_reward, done

    def reset(self):
        pass