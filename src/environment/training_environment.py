import sqlite3
import json
import configparser
import sys
import os

# Xử lý import path nếu chạy local hoặc trong package
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
        
        # Khởi tạo Reward System với cấu hình Sướng trước - Khổ sau
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
        
        # 1. Random số cột để AI tập dò (hoặc cố định là 3 cho dễ trước)
        self.current_hidden_col_count = 3 
        
        # 2. Tạo bảng Products (Bảng bị lỗi injection)
        # QUAN TRỌNG: Cột c1 phải là TEXT để test lỗi ' (String Injection)
        cols = ", ".join([f"c{i} TEXT" for i in range(1, self.current_hidden_col_count + 1)])
        self.cursor.execute(f"CREATE TABLE Products ({cols})")
        
        # Insert dummy data
        placeholders = ",".join(["?"] * self.current_hidden_col_count)
        dummy_data = ["dummy_val"] * self.current_hidden_col_count
        self.cursor.execute(f"INSERT INTO Products VALUES ({placeholders})", dummy_data)

        # 3. Tạo bảng Users (Mục tiêu)
        self.cursor.execute("CREATE TABLE Users (id INTEGER, email TEXT, password TEXT)")
        self.cursor.execute("INSERT INTO Users VALUES (?, ?, ?)", (1, self.success_marker, "123456"))
        
        self.conn.commit()

    def reset(self):
        """Reset môi trường cho ván mới."""
        self._setup_db()
        
        # QUAN TRỌNG: Reset trí nhớ của hệ thống thưởng phạt
        if hasattr(self.reward_system, 'reset'):
            self.reward_system.reset()
            
        return self.state_manager.reset_state()

    def step(self, action_index):
        action_string = self.action_space.get_action_string(action_index)
        
        # Cập nhật State
        state_vector = self.state_manager.update_state(action_string, action_index)
        
        # Lấy payload hiện tại
        payload = self.state_manager.current_state
        
        # --- GIẢ LẬP LỖ HỔNG (Vulnerable Query) ---
        # Query gốc giả định: SELECT * FROM Products WHERE ((c1 = '$input'))
        # Đây là lý do tại sao action "a'))" lại hiệu quả:
        # Nó biến query thành: ... WHERE ((c1 = 'a')) UNION ... --'))
        full_query = f"SELECT * FROM Products WHERE ((c1 = '{payload}'))"
        
        try:
            # Thực thi SQL
            self.cursor.execute(full_query)
            rows = self.cursor.fetchall()
            
            # Nếu không lỗi -> Thành công (về mặt cú pháp)
            response_text = json.dumps(rows)
            status_code = 200
            
        except sqlite3.OperationalError as e:
            err_msg = str(e).lower()
            status_code = 500
            response_text = self.error_marker
            
            # Gợi ý lỗi cho AI (nếu muốn nó học nhanh hơn)
            if "selects to the left and right of union do not have the same number of result columns" in err_msg:
                response_text += "_COLUMN_MISMATCH"
            elif "syntax error" in err_msg or "incomplete input" in err_msg:
                response_text += "_SYNTAX_ERROR"
                
        except Exception:
            status_code = 500
            response_text = "INTERNAL_ERROR"

        # Đóng gói response
        response = type('Response', (), {'status_code': status_code, 'text': response_text})
        
        # Tính thưởng (Reward)
        reward, done = self.reward_system.calculate_reward(response, payload)
        
        return state_vector, reward, done
    
    def get_action_space_size(self):
        return self.action_space.get_action_space_size()