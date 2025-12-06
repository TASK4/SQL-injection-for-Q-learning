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
        # Configurable query parameter name (default 'q')
        self.search_param = target_cfg.get('search_param', 'q')
        
        # Khởi tạo các thành phần cốt lõi
        self.http_client = HttpClient()
        self.action_space = ActionSpace()
        self.state_manager = StateManager() # Cần file state_manager.py
        self.reward_system = RewardSystem(
            normal_count=int(env_cfg['normal_result_count']),
            success_marker=target_cfg.get('success_marker', ''),
            error_marker=target_cfg.get('sql_error_marker', '')
        )
        print("[TargetEnvironment] Đã khởi tạo môi trường Juice Shop.")

    def reset(self):
        """Reset payload (state) về rỗng."""
        return self.state_manager.reset_state()

    def step(self, action_index):
        """Thực hiện action, simulate response, tính reward."""
        # 1. Lấy action string
        action_string = self.action_space.get_action_string(action_index)
        
        # 2. Update state (nối payload)
        new_state = self.state_manager.update_state(action_string)
        
        # 3. Simulate HTTP response dựa trên payload (extract string nếu hash)
        payload_str = self._get_payload_string(new_state)  # New helper
        response = self._simulate_response(payload_str)
        
        # 4. Tính reward & done
        reward, done = self.reward_system.calculate_reward(response, payload_str)  # Pass str to reward
        
        return new_state, reward, done  # Return hash/tuple state for Q-table

    def _get_payload_string(self, state):
        """Extract full string payload từ state (hash or str)."""
        if len(state) == 32 and all(c in '0123456789abcdef' for c in state):  # Likely MD5 hash
            return self.state_manager.get_current_state()  # Full from manager
        return state  # Already str

    def get_action_space_size(self):
        """Trả về số lượng hành động từ ActionSpace."""
        return self.action_space.get_action_space_size()