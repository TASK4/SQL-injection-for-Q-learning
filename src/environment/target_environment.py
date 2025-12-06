import requests
import configparser
import sys
import os
import urllib.parse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.base_environment import BaseEnvironment
from src.core.action_space import ActionSpace
from src.core.state_manager import StateManager
# Target không cần RewardSystem phức tạp vì ta chỉ cần hành động, nhưng cứ import để tránh lỗi nếu code cũ cần
from src.core.reward_system import RewardSystem 

class TargetEnvironment(BaseEnvironment):
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf-8')
        
        # Cấu hình Target
        target_cfg = config['Target'] if 'Target' in config else config['Training']
        
        self.url = target_cfg.get('url', 'http://localhost:3000/rest/products/search')
        self.method = target_cfg.get('method', 'GET')
        self.param_name = target_cfg.get('param_name', 'q')
        
        self.action_space = ActionSpace()
        self.state_manager = StateManager()
        
        # Chỉ dùng để validate, không dùng để train
        self.reward_system = RewardSystem(0, "", "", env_type='target')

    def reset(self):
        """Reset trạng thái trước khi bắt đầu chuỗi tấn công mới"""
        return self.state_manager.reset_state()

    def step(self, action_index):
        # 1. Lấy action và update state nội bộ
        action_string = self.action_space.get_action_string(action_index)
        self.state_manager.update_state(action_string, action_index)
        
        payload = self.state_manager.current_state
        
        # 2. Gửi Request thật đến Juice Shop
        response_text = ""
        status_code = 500
        
        try:
            # Encode payload URL
            params = {self.param_name: payload}
            
            if self.method.upper() == 'GET':
                resp = requests.get(self.url, params=params, timeout=5)
            else:
                resp = requests.post(self.url, data=params, timeout=5)
                
            status_code = resp.status_code
            response_text = resp.text
            
        except Exception as e:
            print(f"[Target] Connection Error: {e}")
            response_text = "CONNECTION_ERROR"

        # 3. --- QUAN TRỌNG: CHUẨN HÓA PHẢN HỒI (NORMALIZATION) ---
        # Biến lỗi thực tế thành tín hiệu mà Agent hiểu (Transfer Learning)
        
        normalized_feedback = response_text
        
        # Dịch lỗi lệch cột
        if "selects to the left and right of union" in response_text.lower():
            normalized_feedback += " _COLUMN_MISMATCH"
            
        # Dịch lỗi cú pháp
        if "sqlite_error" in response_text.lower() and "syntax" in response_text.lower():
            normalized_feedback += " _SYNTAX_ERROR"
        if "unrecognized token" in response_text.lower():
             normalized_feedback += " _SYNTAX_ERROR"

        # 4. Cập nhật Feedback vào não (State Manager)
        self.state_manager.update_feedback(normalized_feedback)
        
        # 5. Lấy State Vector
        next_state = self.state_manager.get_feature_vector()
        
        # Trong thực chiến, ta không biết reward thực sự (trừ khi hack thành công)
        # Nhưng để code chạy đồng bộ, ta cứ tính reward giả định
        reward, done = self.reward_system.calculate_reward(
            type('Response', (), {'status_code': status_code, 'text': normalized_feedback}), 
            payload
        )
        
        # Nếu server trả về kết quả chứa email admin -> DONE
        if "admin@juice-sh.op" in response_text:
            print(f"[!!!] BINGO! Tìm thấy admin với payload: {payload}")
            reward = 100.0
            done = True
            
        return next_state, reward, done

    def get_action_space_size(self):
        return self.action_space.get_action_space_size()