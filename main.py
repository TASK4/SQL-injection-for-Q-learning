import configparser
from src.environment.target_environment import TargetEnvironment
from src.agent.q_learning_agent import QLearningAgent
from tqdm import tqdm  # Thư viện progress bar cho đẹp

def run_training(config_path, model_save_path, model_load_path=None):
    
    print(f"Bắt đầu quá trình huấn luyện với config: {config_path}")
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    
    agent_cfg = config['Agent']
    train_cfg = config['Training']
    
    # 1. Khởi tạo Môi trường (Juice Shop)
    env = TargetEnvironment(config_path)
    
    # 2. Khởi tạo Agent (Bộ não Q-Learning)
    agent = QLearningAgent(
        action_space_size=env.get_action_space_size(),
        lr=float(agent_cfg['learning_rate']),
        gamma=float(agent_cfg['discount_factor']),
        epsilon=float(agent_cfg['epsilon']),
        epsilon_decay=float(agent_cfg['epsilon_decay']),
        epsilon_min=float(agent_cfg['epsilon_min'])
    )

    # 3. Tải model nếu có (phục vụ Transfer Learning)
    # Mặc dù bạn không dùng DVWA, bạn vẫn có thể demo Transfer Learning
    # bằng cách: Huấn luyện 5000 episodes -> lưu model.
    # Sau đó, huấn luyện tiếp 5000 episodes *từ model đã lưu*
    # (model_load_path != None)
    if model_load_path:
        try:
            print(f"Đang tải model tiền huấn luyện từ: {model_load_path}")
            agent.load_model(model_load_path)
            # Khi đã có "kiến thức", giảm Epsilon để khai thác nhiều hơn
            agent.epsilon = float(agent_cfg['epsilon_min'])
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy tệp model. Sẽ huấn luyện từ đầu.")
    
    total_episodes = int(train_cfg['episodes'])
    max_steps = int(train_cfg['max_steps_per_episode'])
    
    print(f"--- Bắt đầu huấn luyện {total_episodes} episodes ---")
    
    # 4. Vòng lặp huấn luyện chính
    for episode in tqdm(range(total_episodes)):
        state = env.reset()
        done = False
        
        for step in range(max_steps):
            # Agent chọn hành động
            action = agent.choose_action(state)
            
            # Môi trường phản hồi
            next_state, reward, done = env.step(action)
            
            # Agent học từ kinh nghiệm
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            
            if done:
                # Tìm thấy lỗ hổng! Kết thúc sớm episode này
                break
        
        # Giảm Epsilon sau mỗi episode
        agent.update_epsilon()

    # 5. Lưu model sau khi hoàn tất
    print(f"\n--- Huấn luyện hoàn tất ---")
    agent.save_model(model_save_path)

if __name__ == "__main__":
    
    CONFIG_FILE = "config/config_target.ini"
    
    # Model sẽ được lưu vào thư mục `target_results`
    # vì nó được huấn luyện *trên* môi trường target (Juice Shop)
    MODEL_SAVE_PATH = "results/target_results/juiceshop_model_v1.json"
    
    # Đặt là None để huấn luyện từ đầu
    MODEL_LOAD_PATH = None 
    
    # *** Để demo Transfer Learning ***
    # 1. Chạy lần 1 với MODEL_LOAD_PATH = None
    # 2. Chạy lần 2 với MODEL_LOAD_PATH = "results/target_results/juiceshop_model_v1.json"
    #    và lưu vào file mới (v2.json). Bạn sẽ thấy agent "thông minh" hơn ngay từ đầu.
    
    run_training(CONFIG_FILE, MODEL_SAVE_PATH, MODEL_LOAD_PATH)