import requests
import time
from src.environment.base_environment import BaseEnvironment
from src.core.action_space import ActionSpace
from src.core.reward_system import RewardSystem
from src.core.state_manager import StateManager
import configparser

class TargetEnvironment(BaseEnvironment):
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf-8')
        
        target_cfg = config['Target']
        self.url = target_cfg.get('url')
        
        # --- LẤY CÁI HAY TỪ CODE CŨ (Config dynamic) ---
        # Cho phép đổi tham số search (vd: 'q', 'search', 'query') từ file config
        self.search_param = target_cfg.get('search_param', 'q') 
        self.success_marker = target_cfg.get('success_marker', 'admin@juice-sh.op')
        self.error_marker = "SQLITE_ERROR" 
        
        # Proxy config (Để null hoặc điền nếu cần soi qua BurpSuite)
        self.proxies = None
        # self.proxies = {"http": "http://127.0.0.1:8080", "https": "http://127.0.0.1:8080"}

        self.action_space = ActionSpace()
        self.state_manager = StateManager() # Feature Vector Mode
        
        self.reward_system = RewardSystem(
            normal_count=0,
            success_marker=self.success_marker,
            error_marker=self.error_marker,
            env_type='target' # Quan trọng: Chế độ Target
        )
        print(f"[TargetEnvironment] Đã khởi tạo. Target: {self.url} (Param: {self.search_param})")

    def reset(self):
        return self.state_manager.reset_state()

    def step(self, action_index):
        action_string = self.action_space.get_action_string(action_index)
        
        # 1. Update State -> Nhận về Vector (Tuple) để Agent học
        # (Đây là cái Agent cần để tra bảng Q-Table)
        new_state_vector = self.state_manager.update_state(action_string, action_index)
        
        # 2. Lấy chuỗi SQL thực tế -> Để gửi lên Server
        # (Đây là cái Server cần để chạy lệnh SQL)
        payload_str = self.state_manager.current_state
        
        # 3. Gửi Request
        response = self._send_payload(payload_str)
        
        # 4. Tính điểm (Truyền payload_str là chuỗi để check keyword, lỗi...)
        reward, done = self.reward_system.calculate_reward(response, payload_str)
        
        return new_state_vector, reward, done

    def _send_payload(self, payload):
        """Gửi payload lên Juice Shop"""
        try:
            # Dùng self.search_param lấy từ config thay vì fix cứng 'q'
            params = {self.search_param: payload} 
            
            # Gửi request
            # timeout=3s để tránh bị treo nếu mạng lag
            resp = requests.get(
                self.url, 
                params=params, 
                proxies=self.proxies, 
                timeout=3
            )
            return resp
        except requests.exceptions.RequestException as e:
            # Nếu lỗi mạng, trả về None để Reward System xử lý (phạt nhẹ)
            # print(f"[TargetEnv] Request Failed: {e}")
            return None

    def get_action_space_size(self):
        return self.action_space.get_action_space_size()