import numpy as np

class RewardSystem:
    def __init__(self, normal_count, success_marker, error_marker, env_type='training'):
        self.env_type = env_type
        # Khung sườn chuẩn (Skeleton) - Đây là thứ tự các "chai nước" cần xếp
        self.skeleton_flow = ["A'))", "UNION", "SELECT", "FROM", "USERS", "--"]
        
    def calculate_reward(self, response, payload):
        """
        Chiến thuật: Incremental Reward (Thưởng tăng tiến)
        """
        s_payload = payload.upper().replace(" ", "") # Xóa space để check thứ tự cho dễ
        
        current_reward = 0.0
        done = False
        
        # --- GIAI ĐOẠN 1: XẾP KHUNG (Structure) ---
        # Kiểm tra từng chặng. Nếu chặng trước chưa xong mà nhảy cóc -> Không có điểm.
        
        # 1. Mở đầu: a'))
        if s_payload.startswith("A'))"):
            current_reward += 0.5 # [Chai 1] Đúng mở đầu -> Có thưởng ngay
            
            # 2. Nối tiếp: UNION
            # (Phải có A')) đứng trước mới tính)
            if "A'))UNION" in s_payload: 
                current_reward += 1.0 # [Chai 2]
                
                # 3. Nối tiếp: SELECT
                if "UNIONSELECT" in s_payload:
                    current_reward += 1.5 # [Chai 3] - Đây là cột mốc quan trọng
                    
                    # --- GIAI ĐOẠN 2: DÒ CỘT (Transfer Learning) ---
                    # Từ sau SELECT đến trước FROM là vùng "Đất thánh" của NULL
                    # Agent cần học cách nhét NULL vào đây.
                    
                    # Tách phần giữa SELECT và FROM (nếu có FROM)
                    if "FROM" in s_payload:
                        # Đã đóng khung -> Kiểm tra kết quả
                        current_reward += 2.0 # [Chai 4] Đã ráp xong câu lệnh
                        
                        # Logic phản hồi thực tế (QUAN TRỌNG CHO TARGET)
                        if response.status_code == 200 and "SQLITE_ERROR" not in response.text:
                             # JACKPOT!
                             current_reward += 20.0
                             done = True
                        elif "COLUMN_MISMATCH" in response.text:
                             # Sai số lượng cột -> Phạt RẤT NHẸ để nó chỉnh lại số lượng NULL
                             # Agent sẽ hiểu: "Cấu trúc ngon rồi, chỉ cần chỉnh số NULL thôi"
                             current_reward -= 0.2 
                        else:
                             # Lỗi cú pháp khác
                             current_reward -= 1.0
                    else:
                        # Chưa có FROM -> Đang ở giai đoạn thêm NULL
                        # Khuyến khích thêm NULL hoặc dấu phẩy
                        # Logic: Cứ thêm 1 cặp ", NULL" là được thưởng nhẹ
                        if payload.strip().upper().endswith("NULL") or payload.strip().endswith(","):
                            current_reward += 0.2 
                            
                        # Nếu đi lạc đề (thêm từ khóa lạ vào giữa lúc đang dò NULL) -> Phạt
                        if "UNION" in payload[payload.upper().rfind("SELECT"):]:
                            current_reward -= 0.5

        # --- GIAI ĐOẠN 3: PHẠT THÔNG MINH (Smart Penalty) ---
        
        # Phạt vòng lặp vô nghĩa (Tránh spam)
        if len(payload) > 80:
            current_reward -= 1.0
            
        # Phạt cú pháp gãy (Syntax Error) từ DB trả về
        if "SYNTAX_ERROR" in response.text:
             current_reward -= 0.5 # Phạt nhẹ để biết đường tránh

        return current_reward, done

    def reset(self):
        pass