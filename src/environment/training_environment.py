import sqlite3
import json
import configparser
import sys
import os
import random  # <--- Import thêm random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.base_environment import BaseEnvironment
from src.core.action_space import ActionSpace
from src.core.reward_system import RewardSystem
from src.core.state_manager import StateManager

class TrainingEnvironment(BaseEnvironment):
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf-8')
        train_cfg = config['Training']
        
        self.success_marker = train_cfg.get('success_marker', 'admin@juice-sh.op')
        self.error_marker = "SQLITE_ERROR" 
        
        self.action_space = ActionSpace()
        self.state_manager = StateManager()
        
        self.reward_system = RewardSystem(
            normal_count=0,
            success_marker=self.success_marker,
            error_marker=self.error_marker,
            env_type='training'
        )
        
        self.conn = None
        self.cursor = None
        self.current_hidden_col_count = 0

    def _setup_db(self):
        """Khởi tạo DB SQLite ảo trong RAM."""
        if self.conn:
            self.conn.close()
        
        self.conn = sqlite3.connect(':memory:')
        self.cursor = self.conn.cursor()
        
        # --- FIX FOR TRANSFER LEARNING: RANDOM PUZZLE FRAME ---
        # Thay vì cố định 3 cột, mỗi lần reset ta random từ 1 đến 6 cột.
        # AI buộc phải học thuật toán "dò cột" thay vì học vẹt số lượng.
        self.current_hidden_col_count = random.randint(1, 6)
        # print(f"[DEBUG] Số cột ẩn hiện tại: {self.current_hidden_col_count}") # Bật nếu muốn soi
        
        # Tạo bảng Products (Bảng bị lỗi injection)
        cols = ", ".join([f"c{i} TEXT" for i in range(1, self.current_hidden_col_count + 1)])
        self.cursor.execute(f"CREATE TABLE Products ({cols})")
        
        # Insert dummy data
        placeholders = ",".join(["?"] * self.current_hidden_col_count)
        dummy_data = ["dummy_val"] * self.current_hidden_col_count
        self.cursor.execute(f"INSERT INTO Products VALUES ({placeholders})", dummy_data)

        # Tạo bảng Users (Mục tiêu)
        self.cursor.execute("CREATE TABLE Users (id INTEGER, email TEXT, password TEXT)")
        self.cursor.execute("INSERT INTO Users VALUES (?, ?, ?)", (1, self.success_marker, "123456"))
        
        self.conn.commit()

    def reset(self):
        """Reset môi trường cho ván mới."""
        self._setup_db()
        
        # Reset reward system để tính điểm lại từ đầu
        if hasattr(self.reward_system, 'reset'):
            self.reward_system.reset()
            
        return self.state_manager.reset_state()

    def step(self, action_index):
        action_string = self.action_space.get_action_string(action_index)
        
        # Cập nhật State
        state_vector = self.state_manager.update_state(action_string, action_index)
        
        # Lấy payload hiện tại
        payload = self.state_manager.current_state
        
        # --- GIẢ LẬP LỖ HỔNG ---
        full_query = f"SELECT * FROM Products WHERE ((c1 = '{payload}'))"
        
        try:
            self.cursor.execute(full_query)
            rows = self.cursor.fetchall()
            
            # Thành công về cú pháp và logic
            response_text = json.dumps(rows)
            status_code = 200
            
        except sqlite3.OperationalError as e:
            err_msg = str(e).lower()
            status_code = 500
            response_text = self.error_marker
            
            # Gợi ý lỗi quan trọng để AI học nhanh hơn
            if "selects to the left and right of union do not have the same number of result columns" in err_msg:
                response_text += "_COLUMN_MISMATCH"
            elif "syntax error" in err_msg or "incomplete input" in err_msg:
                response_text += "_SYNTAX_ERROR"
                
        except Exception:
            status_code = 500
            response_text = "INTERNAL_ERROR"

        # Đóng gói response
        response = type('Response', (), {'status_code': status_code, 'text': response_text})
        
        # Tính thưởng
        reward, done = self.reward_system.calculate_reward(response, payload)
        
        return state_vector, reward, done
    
    def get_action_space_size(self):
        return self.action_space.get_action_space_size()