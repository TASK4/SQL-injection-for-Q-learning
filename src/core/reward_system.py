import numpy as np

class RewardSystem:
    def __init__(self, normal_count, success_marker, error_marker, env_type='training'):
        self.env_type = env_type
        
    def calculate_reward(self, response, payload):
        # Xóa khoảng trắng để so sánh logic
        s_payload = payload.upper().replace(" ", "") 
        
        current_reward = -0.05 # Phạt rất nhẹ chi phí bước đi (giảm từ -0.1)
        done = False
        
        # --- GIAI ĐOẠN 1: CHECKPOINT CỬA VÀO ---
        if s_payload.startswith("A'))"):
            current_reward += 1.0 
            
            # --- GIAI ĐOẠN 2: KHUNG SƯỜN ---
            if "UNION" in s_payload and "SELECT" in s_payload:
                current_reward += 2.0 
                
                # [NEW] Thưởng cho các cột quan trọng (Hướng dẫn Agent lấy dữ liệu)
                # Check trong payload gốc (có case sensitive hoặc không tùy ý, ở đây check upper)
                p_upper = payload.upper()
                if "ID" in p_upper: current_reward += 1.0
                if "EMAIL" in p_upper: current_reward += 1.0
                if "PASSWORD" in p_upper: current_reward += 1.0

                # --- GIAI ĐOẠN 3: DÒ CỘT (Transfer Learning) ---
                if "COLUMN_MISMATCH" in response.text:
                    # Nếu lệch cột: Phạt RẤT NHẸ
                    current_reward -= 0.1 
                    # Thưởng ĐỘNG VIÊN nếu đang thêm NULL hoặc dấu phẩy
                    if payload.strip().upper().endswith("NULL") or payload.strip().endswith(","):
                        current_reward += 0.5
                        
                elif "SYNTAX_ERROR" not in response.text and response.status_code == 200:
                    # Đã SELECT thành công (đúng số cột)
                    current_reward += 5.0 
                    
                    # --- GIAI ĐOẠN 4: KHAI THÁC ---
                    # Logic cũ: chỉ check FROM USERS.
                    # Logic mới: Nếu đã đúng cột và có FROM USERS -> Win to
                    if "FROM" in s_payload and "USERS" in s_payload:
                        current_reward += 20.0 
                        # Nếu juice shop trả về admin thì done ở bên environment, 
                        # nhưng ở đây ta cứ khuyến khích mạnh.
                        
        else:
            # Chưa có cửa vào đúng: Phạt nhẹ nếu spam linh tinh
            # ĐÃ XÓA PHẠT ĐỘ DÀI > 10 Ở ĐÂY
            pass

        # Phạt lỗi cú pháp
        if "SYNTAX_ERROR" in response.text:
             current_reward -= 0.5

        return current_reward, done

    def reset(self):
        pass