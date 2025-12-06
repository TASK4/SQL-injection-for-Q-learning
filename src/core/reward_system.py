import numpy as np

class RewardSystem:
    def __init__(self, normal_count, success_marker, error_marker, env_type='training'):
        self.env_type = env_type
        # Đảm bảo các biến này được gán vào self
        self.success_marker = success_marker
        self.error_marker = error_marker
        
    def calculate_reward(self, response, payload):
        # Xóa khoảng trắng để so sánh logic
        s_payload = payload.upper().replace(" ", "") 
        
        current_reward = -0.05 # Phạt rất nhẹ chi phí bước đi
        done = False
        
        # --- [QUAN TRỌNG] KIỂM TRA CHIẾN THẮNG ĐẦU TIÊN ---
        # Nếu tìm thấy success_marker (admin email) -> THẮNG NGAY
        if self.success_marker in response.text:
            current_reward += 100.0  # Thưởng lớn
            done = True              # Kết thúc game
            return current_reward, done

        # --- GIAI ĐOẠN 1: CHECKPOINT CỬA VÀO ---
        if s_payload.startswith("A'))"):
            current_reward += 1.0 
            
            # --- GIAI ĐOẠN 2: KHUNG SƯỜN ---
            if "UNION" in s_payload and "SELECT" in s_payload:
                current_reward += 2.0 
                
                # Thưởng cho các cột quan trọng
                p_upper = payload.upper()
                if "ID" in p_upper: current_reward += 1.0
                if "EMAIL" in p_upper: current_reward += 1.0
                if "PASSWORD" in p_upper: current_reward += 1.0

                # --- GIAI ĐOẠN 3: DÒ CỘT ---
                if "COLUMN_MISMATCH" in response.text:
                    # Nếu lệch cột: Phạt nhẹ
                    current_reward -= 0.1 
                    # Thưởng động viên nếu đang thêm NULL hoặc dấu phẩy
                    if payload.strip().upper().endswith("NULL") or payload.strip().endswith(","):
                        current_reward += 0.5
                        
                elif "SYNTAX_ERROR" not in response.text and response.status_code == 200:
                    # Đã SELECT thành công (đúng số cột)
                    current_reward += 5.0 
                    
                    # --- GIAI ĐOẠN 4: KHAI THÁC ---
                    if "FROM" in s_payload and "USERS" in s_payload:
                        current_reward += 20.0 
                        
        else:
            # Chưa có cửa vào đúng
            pass

        # Phạt lỗi cú pháp
        if "SYNTAX_ERROR" in response.text:
             current_reward -= 0.5

        return current_reward, done

    def reset(self):
        pass