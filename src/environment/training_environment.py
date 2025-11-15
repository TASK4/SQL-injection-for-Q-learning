import configparser
from src.environment.base_environment import BaseEnvironment
from src.core.action_space import ActionSpace
from src.core.state_manager import StateManager  # Giả sử file này tồn tại như trong TargetEnvironment

class TrainingEnvironment(BaseEnvironment):
    """
    Môi trường Huấn luyện (Mock DVWA) để agent học cách tạo ra
    một payload SQLi cụ thể một cách nhanh chóng.
    """
    
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf-8')
        
        train_cfg = config['TrainingTarget']
        env_cfg = config['Environment']

        # Payload mục tiêu mà agent phải học
        self.target_payload = train_cfg['target_payload']
        
        # Định nghĩa các phần thưởng/phạt từ config
        self.reward_on_track = float(env_cfg['reward_on_track'])
        self.reward_success = float(env_cfg['reward_success'])
        self.penalty_wrong_move = float(env_cfg['penalty_wrong_move'])
        self.penalty_per_step = float(env_cfg['penalty_per_step'])
        
        # Các thành phần cốt lõi
        self.action_space = ActionSpace()
        self.state_manager = StateManager()
        
        print(f"[TrainingEnvironment] Đã khởi tạo môi trường mock DVWA.")
        print(f"[TrainingEnvironment] Mục tiêu: '{self.target_payload}'")

    def reset(self):
        """Reset payload (state) về rỗng."""
        return self.state_manager.reset_state()

    def step(self, action_index):
        """
        Thực hiện 1 bước trong môi trường MOCK.
        Không có request HTTP, chỉ so sánh chuỗi.
        """
        
        # 1. Lấy chuỗi hành động (ví dụ: "'")
        action_string = self.action_space.get_action_string(action_index)
        
        # 2. Cập nhật trạng thái (nối chuỗi vào payload)
        new_state = self.state_manager.update_state(action_string)
        
        # 3. Tính toán phần thưởng (Logic cốt lõi của Mock Env)
        
        done = False
        
        # Kiểm tra xem payload mới có khớp chính xác mục tiêu không
        if new_state == self.target_payload:
            # Thành công!
            reward = self.reward_success
            done = True
        
        # Kiểm tra xem payload mới có đang đi đúng hướng không
        elif self.target_payload.startswith(new_state):
            # Đi đúng hướng, thưởng nhẹ để khuyến khích
            # (Thêm penalty_per_step để nó vẫn cố gắng đi nhanh)
            reward = self.reward_on_track + self.penalty_per_step
        
        # Nếu đi sai hướng (spam)
        else:
            # Phạt nặng để agent "học" rằng đây là đường sai
            # Đây là cách xử lý "tránh spam" bạn đã đề cập
            reward = self.penalty_wrong_move + self.penalty_per_step
            # Nếu đi sai, ta có thể coi như episode này thất bại luôn
            # done = True # (Tùy chọn: bạn có thể bật cờ này nếu muốn kết thúc episode ngay khi đi sai)

        return new_state, reward, done

    def get_action_space_size(self):
        """Trả về số lượng hành động từ ActionSpace."""
        return self.action_space.get_action_space_size()