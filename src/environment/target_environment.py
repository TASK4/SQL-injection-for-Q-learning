from src.environment.base_environment import BaseEnvironment
from src.utils.http_client import HttpClient
from src.core.action_space import ActionSpace
from src.core.reward_system import RewardSystem
from src.core.state_manager import StateManager  # Giả sử bạn có file này
import configparser

# Giả sử bạn tạo file src/core/state_manager.py
# class StateManager:
#     def __init__(self): self.current_state = ""
#     def get_current_state(self): return self.current_state
#     def reset_state(self): self.current_state = ""; return self.current_state
#     def update_state(self, action_string): 
#         self.current_state += action_string; return self.current_state

class TargetEnvironment(BaseEnvironment):
    """
    Môi trường Juice Shop cụ thể, kế thừa từ BaseEnvironment.
    """
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf-8')
        
        target_cfg = config['Target']
        env_cfg = config['Environment']

        self.search_url = target_cfg['search_url']
        
        # Khởi tạo các thành phần cốt lõi
        self.http_client = HttpClient()
        self.action_space = ActionSpace()
        self.state_manager = StateManager() # Cần file state_manager.py
        self.reward_system = RewardSystem(
            normal_count=int(env_cfg['normal_result_count']),
            success_marker=env_cfg['success_marker'],
            error_marker=env_cfg['sql_error_marker']
        )
        print("[TargetEnvironment] Đã khởi tạo môi trường Juice Shop.")

    def reset(self):
        """Reset payload (state) về rỗng."""
        return self.state_manager.reset_state()

    def step(self, action_index):
        """Thực hiện 1 bước trong môi trường."""
        
        # 1. Lấy chuỗi hành động (ví dụ: 'UNION')
        action_string = self.action_space.get_action_string(action_index)
        
        # 2. Cập nhật trạng thái (nối chuỗi vào payload)
        new_state = self.state_manager.update_state(action_string)
        
        # 3. Gửi payload đến Juice Shop
        response = self.http_client.send_search_query(
            self.search_url, 
            new_state
        )
        
        # 4. Tính toán phần thưởng
        reward, done = self.reward_system.calculate_reward(response, new_state)
        
        return new_state, reward, done

    def get_action_space_size(self):
        """Trả về số lượng hành động từ ActionSpace."""
        return self.action_space.get_action_space_size()