import numpy as np

class RewardSystem:
    def __init__(self, normal_count, success_marker, error_marker, env_type='training'):
        self.env_type = env_type
        # KHUNG SƯỜN: Agent phải đi theo thứ tự này mới được điểm
        self.skeleton_flow = ["A'))", "UNION", "SELECT", "FROM", "USERS", "--"]
        
    def calculate_reward(self, response, payload):
        # Xóa khoảng trắng để so sánh thứ tự cho dễ
        s_payload = payload.upper().replace(" ", "") 
        
        current_reward = 0.0
        done = False
        
        # --- GIAI ĐOẠN 1: XẾP KHUNG (Structure) ---
        
        # 1. Mở đầu: a')) -> THƯỞNG NGAY LẬP TỨC (+0.5) thay vì bị phạt
        if s_payload.startswith("A'))"):
            current_reward += 0.5 
            
            # 2. Nối tiếp: UNION
            if "A'))UNION" in s_payload: 
                current_reward += 1.0 
                
                # 3. Nối tiếp: SELECT
                if "UNIONSELECT" in s_payload:
                    current_reward += 1.5 
                    
                    # --- GIAI ĐOẠN 2: DÒ CỘT (Transfer Learning) ---
                    if "FROM" in s_payload:
                        # Đã đóng khung -> Kiểm tra kết quả
                        current_reward += 2.0 
                        
                        # Logic phản hồi thực tế
                        if response.status_code == 200 and "SQLITE_ERROR" not in response.text:
                             current_reward += 20.0 # WIN
                             done = True
                        elif "COLUMN_MISMATCH" in response.text:
                             # Sai số lượng cột -> Phạt RẤT NHẸ (-0.2) để nó chỉnh lại
                             current_reward -= 0.2 
                        else:
                             current_reward -= 1.0
                    else:
                        # Chưa có FROM -> Đang ở giai đoạn thêm NULL
                        # Cứ thêm 1 cái NULL hoặc dấu phẩy là được thưởng thêm
                        if payload.strip().upper().endswith("NULL") or payload.strip().endswith(","):
                            current_reward += 0.2 
                            
                        # Phạt nếu chèn từ khóa lạ vào giữa lúc đang dò NULL
                        if "UNION" in payload[payload.upper().rfind("SELECT"):]:
                            current_reward -= 0.5

        # --- GIAI ĐOẠN 3: PHẠT ---
        if len(payload) > 80:
            current_reward -= 1.0
        if "SYNTAX_ERROR" in response.text:
             current_reward -= 0.5

        return current_reward, done

    def reset(self):
        pass