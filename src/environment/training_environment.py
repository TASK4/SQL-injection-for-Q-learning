def step(self, action_index):
        action_string = self.action_space.get_action_string(action_index)
        
        # 1. Cập nhật Payload text
        self.state_manager.update_state(action_string, action_index)
        payload = self.state_manager.current_state
        
        # 2. CHẠY SQL GIẢ LẬP (Để lấy Feedback)
        full_query = f"SELECT * FROM Products WHERE ((c1 = '{payload}'))"
        
        response_text = ""
        status_code = 500
        
        try:
            self.cursor.execute(full_query)
            rows = self.cursor.fetchall()
            status_code = 200
            response_text = json.dumps(rows)
        except sqlite3.OperationalError as e:
            err_msg = str(e).lower()
            response_text = self.error_marker
            
            # Gắn tag lỗi rõ ràng cho State Manager đọc
            if "selects to the left and right of union" in err_msg:
                response_text += "_COLUMN_MISMATCH"
            elif "syntax error" in err_msg or "incomplete input" in err_msg:
                response_text += "_SYNTAX_ERROR"
        except Exception:
            response_text = "INTERNAL_ERROR"

        # 3. CẬP NHẬT FEEDBACK VÀO STATE (Điểm mấu chốt)
        self.state_manager.update_feedback(response_text)
        
        # 4. Lấy State Vector cuối cùng (đã bao gồm thông tin lỗi)
        state_vector = self.state_manager.get_feature_vector()
        
        # 5. Tính điểm
        response = type('Response', (), {'status_code': status_code, 'text': response_text})
        reward, done = self.reward_system.calculate_reward(response, payload)
        
        return state_vector, reward, done