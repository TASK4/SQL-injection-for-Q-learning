def calculate_reward(self, response, payload):
        s_payload = payload.upper().replace(" ", "") 
        current_reward = -0.1 # Phạt nhẹ mỗi bước đi để khuyến khích đi ngắn gọn
        done = False
        
        # --- CHECKPOINT 1: CỬA VÀO ---
        if s_payload.startswith("A'))"):
            current_reward += 1.0 # Thưởng to để neo checkpoint này
            
            # --- CHECKPOINT 2: KHUNG SƯỜN ---
            if "UNION" in s_payload and "SELECT" in s_payload:
                current_reward += 2.0
                
                # --- CHECKPOINT 3: DÒ CỘT (Transfer Learning) ---
                # Logic "Chai nước": Nếu bị lệch cột, đừng phạt nặng, hãy thưởng nếu nó đang cố thêm NULL
                if "COLUMN_MISMATCH" in response.text:
                    current_reward -= 0.1 # Chỉ nhắc nhở nhẹ
                    # Nếu hành động vừa rồi là thêm NULL hoặc phẩy, thưởng động viên
                    if payload.strip().upper().endswith("NULL") or payload.strip().endswith(","):
                         current_reward += 0.5
                
                elif "SYNTAX_ERROR" not in response.text and response.status_code == 200:
                    # Đã SELECT thành công (đúng số cột)
                    current_reward += 5.0
                    
                    # --- CHECKPOINT 4: LẤY DATA ---
                    if "FROM" in s_payload and "USERS" in s_payload:
                        current_reward += 20.0 # WIN
                        done = True
        else:
            # Chưa có cửa vào mà spam linh tinh -> Phạt nặng
            if len(payload) > 10: 
                current_reward -= 1.0

        # Phạt Syntax Error (nhưng đừng quá gắt để agent dám thử)
        if "SYNTAX_ERROR" in response.text:
             current_reward -= 0.5
             
        return current_reward, done