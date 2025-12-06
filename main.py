import configparser
from src.environment.training_environment import TrainingEnvironment  # Thêm import này
from src.environment.target_environment import TargetEnvironment
from src.agent.q_learning_agent import QLearningAgent
import os
import logging
import matplotlib.pyplot as plt

# --- CÀI ĐẶT LOGGING ---
os.makedirs("results/target_results", exist_ok=True)
LOG_FILE = "results/target_results/training_log.txt"
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), logging.StreamHandler()])

def run_training(config_path, model_save_path, model_load_path=None, env_type='training'):
    logging.info(f"Bắt đầu quá trình huấn luyện với config: {config_path} | Env: {env_type}")
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    
    agent_cfg = config['Agent']
    train_cfg = config['Training']
    
    # 1. Khởi tạo Môi trường
    if env_type == 'training':
        env = TrainingEnvironment(config_path)
        logging.info("Sử dụng TrainingEnvironment (mock) cho speed.")
    else:
        env = TargetEnvironment(config_path)
        logging.info("Sử dụng TargetEnvironment (real HTTP).")
    
    # 2. Khởi tạo Agent (MỘT LẦN DUY NHẤT)
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
            logging.info(f"Đang tải model từ: {model_load_path}")
            agent.load_model(model_load_path)
            agent.epsilon = float(agent_cfg['epsilon_min'])
        except FileNotFoundError:
            logging.error(f"Lỗi: Không tìm thấy model. Train từ đầu.")

    # 3. Baseline test (nếu enabled)
    if config['Training'].getboolean('baseline_enabled', False):
        logging.info("Chạy baseline test...")
        try:
            state, reward, done = env.test_baseline()
            if reward > 0:
                # Fake learn: Action 0 cho baseline (seed Q-table)
                next_state = state  # Self-loop
                agent.learn(state, 0, reward, next_state)
                logging.info(f"Baseline success! Reward: {reward} | Seeded Q-table.")
            else:
                logging.warning("Baseline reward <=0, skip seeding.")
        except Exception as e:
            logging.warning(f"Baseline test fail: {e}, skip.")

    # 4. Training loop
    total_episodes = int(train_cfg['episodes'])
    max_steps = int(train_cfg['max_steps_per_episode'])
    
    rewards_per_episode = []
    total_successes = 0
    
    logging.info(f"--- Bắt đầu {total_episodes} episodes ---")
    
    for episode in range(total_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            episode_reward += reward
            
            if done:
                total_successes += 1
                logging.info(f"EPISODE {episode}: SUCCESS! Payload: {state} (Reward: {episode_reward})")
                break
        
        rewards_per_episode.append(episode_reward)
        agent.update_epsilon()

        if (episode + 1) % 100 == 0:
            last_100 = rewards_per_episode[-100:]
            avg_reward = sum(last_100) / 100
            success_count_100 = sum(1 for r in last_100 if r > 50)  # Success >50 (partial <50)
            logging.info(f"Episode {episode + 1}: Avg Reward(100ep): {avg_reward:.2f} | Successes(100ep): {success_count_100}")

    # 5. Save model & Plot
    logging.info(f"\n--- Hoàn tất ---")
    agent.save_model(model_save_path)
    
    success_rate = (total_successes / total_episodes) * 100
    logging.info(f"Tổng success: {total_successes}/{total_episodes} ({success_rate:.2f}%)")

    plt.figure(figsize=(12, 6))
    plt.plot(rewards_per_episode)
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    
    chart_save_dir = "results/target_results"
    os.makedirs(chart_save_dir, exist_ok=True)
    chart_save_path = os.path.join(chart_save_dir, "training_rewards_chart.png")
    plt.savefig(chart_save_path)
    logging.info(f"Lưu chart: {chart_save_path}")
    plt.show()

if __name__ == "__main__":
    CONFIG_FILE = "config/config_training.ini"  # Dùng training config
    MODEL_SAVE_PATH = "results/target_results/juiceshop_model_v1.json"
    MODEL_LOAD_PATH = None 
    run_training(CONFIG_FILE, MODEL_SAVE_PATH, MODEL_LOAD_PATH, env_type='training')  # Default training