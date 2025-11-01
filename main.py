import configparser
from src.environment.target_environment import TargetEnvironment
from src.agent.q_learning_agent import QLearningAgent
from tqdm import tqdm  # Thư viện progress bar cho đẹp

# --- THÊM DÒNG NÀY ---
import matplotlib.pyplot as plt
# ---------------------

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

    if model_load_path:
        try:
            print(f"Đang tải model tiền huấn luyện từ: {model_load_path}")
            agent.load_model(model_load_path)
            agent.epsilon = float(agent_cfg['epsilon_min'])
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy tệp model. Sẽ huấn luyện từ đầu.")
    
    total_episodes = int(train_cfg['episodes'])
    max_steps = int(train_cfg['max_steps_per_episode'])
    
    # --- THÊM BIẾN ĐỂ THEO DÕI ---
    rewards_per_episode = [] # Lưu phần thưởng mỗi episode để vẽ biểu đồ
    total_successes = 0      # Đếm số lần thành công
    # -----------------------------
    
    print(f"--- Bắt đầu huấn luyện {total_episodes} episodes ---")
    
    # 4. Vòng lặp huấn luyện chính
    for episode in tqdm(range(total_episodes)):
        state = env.reset()
        done = False
        
        episode_reward = 0 # Biến tạm để đếm phần thưởng trong episode này

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            episode_reward += reward # Cộng dồn phần thưởng
            
            if done:
                total_successes += 1 # Ghi nhận thành công
                break
        
        rewards_per_episode.append(episode_reward) # Lưu kết quả của episode này
        agent.update_epsilon()

    # 5. Lưu model và IN KẾT QUẢ
    print(f"\n--- Huấn luyện hoàn tất ---")
    agent.save_model(model_save_path)
    
    # --- TÍNH TOÁN HIỆU SUẤT VÀ VẼ BIỂU ĐỒ ---
    success_rate = (total_successes / total_episodes) * 100
    print("\n--- KẾT QUẢ HUẤN LUYỆN ---")
    print(f"Tổng số lần thành công (tìm thấy lỗ hổng): {total_successes}")
    print(f"Tổng số episodes: {total_episodes}")
    print(f"Hiệu suất học (Tỷ lệ thành công): {success_rate:.2f}%")

    # Vẽ biểu đồ
    print("Đang hiển thị biểu đồ phần thưởng...")
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_per_episode)
    plt.title("Tiến trình học của Agent (Tổng phần thưởng qua mỗi Episode)")
    plt.xlabel("Episode")
    plt.ylabel("Tổng phần thưởng (Total Reward)")
    plt.grid(True)
    plt.show()
    # -----------------------------------------

if __name__ == "__main__":
    
    CONFIG_FILE = "config/config_target.ini"
    MODEL_SAVE_PATH = "results/target_results/juiceshop_model_v1.json"
    MODEL_LOAD_PATH = None 
    
    run_training(CONFIG_FILE, MODEL_SAVE_PATH, MODEL_LOAD_PATH)