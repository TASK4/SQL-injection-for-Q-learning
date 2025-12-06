#main.py
import configparser
from src.environment.training_environment import TrainingEnvironment  # Thêm import này
from src.environment.target_environment import TargetEnvironment
from src.agent.q_learning_agent import QLearningAgent
import os
import logging
import time
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# --- CÀI ĐẶT LOGGING ---
# (Đã xóa phần logging cũ, tốt rồi)

def run_training(config_path, model_save_path, model_load_path=None, env_type='training'):
    
    # --- 1. CÀI ĐẶT ĐƯỜNG DẪN ĐỘNG ---
    if env_type == 'training':
        output_dir = "results/train_results"
    else:
        output_dir = "results/target_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 2. CÀI ĐẶT LOGGING ĐỘNG ---
    # <<< THAY ĐỔI 1: DÙNG TÊN FILE LOG ĐỘNG ---
    LOG_FILE = os.path.join(output_dir, f"{env_type}_log.txt") 
    
    # Xóa các handlers cũ (nếu có) để tránh log bị lặp lại
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), 
                                  logging.StreamHandler()])
    
    start_time = time.time()
    
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
            # --- THÊM LOGIC NÀY VÀO ---
            if env_type == 'target':
                # [SỬA] Đừng ép về 0.01 ngay. Hãy để 0.1 hoặc 0.2 để nó dám thử thêm dấu cách
                agent.epsilon = 0.3 
                logging.info(f"TARGET MODE: Epsilon đặt lại là {agent.epsilon} để fine-tune.")
            else:
                logging.info(f"TRAIN MODE: Epsilon được giữ ở: {agent.epsilon} (từ config)")
            # --- KẾT THÚC THAY ĐỔI ---
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
    successes_per_episode = []
    total_successes = 0
    
    moving_avg_rewards = []  # Lưu các điểm reward trung bình
    moving_avg_episodes = [] # Lưu mốc episode tương ứng (100, 200, 300...)
    
    logging.info(f"--- Bắt đầu {total_episodes} episodes ---")
    
    # --- SỬA ĐỔI: TÍNH LOG_FREQ ĐỘNG ---
    # Nếu chạy ít hơn 2000 ep (Target), log mỗi 10 ep hoặc 50 ep.
    # Nếu chạy nhiều (Train), log mỗi 100 ep hoặc 500 ep.
    if total_episodes <= 1000:
        log_freq = 10  # Target Mode: Log dày hơn (mỗi 10 ep) để thấy rõ biểu đồ
    else:
        log_freq = 100 # Train Mode: Giữ nguyên 
    
    for episode in range(total_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_succeeded = False

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            episode_reward += reward
            
            if reward > 50:
                total_successes += 1
                episode_succeeded = True
                logging.info(f"EPISODE {episode}: SUCCESS! Payload: {state} (Reward: {episode_reward})")
                break
            
            if done:
                break
        
        rewards_per_episode.append(episode_reward)
        successes_per_episode.append(episode_succeeded)
        agent.update_epsilon()

        # --- SỬA ĐỔI: DÙNG BIẾN log_freq ---
        if (episode + 1) % log_freq == 0:
            # Lấy trung bình của log_freq episodes gần nhất thay vì fix cứng 100
            last_n = rewards_per_episode[-log_freq:] 
            avg_reward = sum(last_n) / len(last_n)
            
            success_count = sum(successes_per_episode[-log_freq:])
            
            logging.info(f"Episode {episode + 1}: Avg Reward({log_freq}ep): {avg_reward:.2f} | Successes({log_freq}ep): {success_count}")

            moving_avg_rewards.append(avg_reward)
            moving_avg_episodes.append(episode + 1)

    # 5. Save model, Plot & Performance Metrics
    logging.info(f"\n--- Hoàn tất ---")
    agent.save_model(model_save_path) # Đường dẫn này đã được truyền từ argparse
    
    # === PHẦN TÍNH HIỆU SUẤT NÂNG CẤP ===
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    if total_episodes > 0:
        # 1. Tỷ lệ thành công (Bạn đã có)
        success_rate = (total_successes / total_episodes) * 100
        logging.info(f"Tổng success: {total_successes}/{total_episodes} ({success_rate:.2f}%)")
        
        # 2. Reward trung bình (Toàn bộ)
        avg_reward_overall = sum(rewards_per_episode) / total_episodes
        logging.info(f"Reward trung bình (toàn bộ): {avg_reward_overall:.2f}")
        
        # 3. Reward trung bình (10% episodes cuối - xem độ hội tụ)
        last_10_percent_index = int(total_episodes * 0.9)
        last_10_rewards = rewards_per_episode[last_10_percent_index:]
        
        if last_10_rewards: # Đảm bảo không chia cho 0
            avg_reward_last_10 = sum(last_10_rewards) / len(last_10_rewards)
            logging.info(f"Reward trung bình (10% cuối): {avg_reward_last_10:.2f}")
    
        stats_text = (
            f"--- Performance Metrics ---\n"
            f"Total Success: {total_successes}/{total_episodes} ({success_rate:.2f}%)\n"
            f"Avg Reward (Overall): {avg_reward_overall:.2f}\n"
            f"Avg Reward (Last 10%): {avg_reward_last_10:.2f}\n"
            f"Total Duration: {total_duration:.2f} s"
        )
    else:
        logging.warning("Không có episode nào chạy, không thể tính hiệu suất.")
        stats_text = (
            f"--- Performance Metrics ---\n"
            f"Không có episode nào chạy.\n"
            f"Total Duration: {total_duration:.2f} s"
        )
    
    logging.info(f"Tổng thời gian chạy: {total_duration:.2f} giây")
    # === KẾT THÚC PHẦN NÂNG CẤP ===

    # Vẽ biểu đồ (PHIÊN BẢN NÂNG CẤP V2 - DỜI HỘP RA NGOÀI)
    plt.figure(figsize=(12, 6))
    
    # Vẽ đường trung bình trượt (100ep)
    plt.plot(moving_avg_episodes, moving_avg_rewards, label=f'Average Reward ({log_freq} episodes)')
    
    plt.title(f"Training Progress ({env_type} mode)")
    plt.xlabel("Episode")
    plt.ylabel(f"Average Reward ({log_freq} Episodes)")
    plt.grid(True)

    # --- THAY ĐỔI 1: DỜI HỘP CHÚ THÍCH (LEGEND) LÊN TRÊN ---
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, fancybox=True, shadow=True)

    # --- THAY ĐỔI 2: DỜI HỘP METRICS XUỐNG DƯỚI ---
    plt.subplots_adjust(bottom=0.25) 
    
    plt.figtext(0.95, 0.05, stats_text, ha="right", va="bottom", fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "whitesmoke", "edgecolor": "gray"})
    
    # --- Hết thay đổi ---
    
    # <<< THAY ĐỔI 2: DÙNG TÊN FILE CHART ĐỘNG ---
    chart_save_path = os.path.join(output_dir, f"{env_type}_chart.png") 
    
    plt.savefig(chart_save_path)
    logging.info(f"Lưu chart: {chart_save_path}")
    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Q-Learning Agent cho SQLi.")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'target'], 
        required=True, 
        help="Chế độ chạy: 'train' (mock env) hoặc 'target' (real env)"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help="Đường dẫn tới file config (training hoặc target)"
    )
    parser.add_argument(
        '--save_path', 
        type=str, 
        required=True, 
        help="Nơi lưu model sau khi chạy"
    )
    parser.add_argument(
        '--load_path', 
        type=str, 
        default=None, 
        help="(Tùy chọn) Đường dẫn tới model đã huấn luyện để tải"
    )

    args = parser.parse_args()

    # Ánh xạ mode sang env_type
    env_type_str = 'training' if args.mode == 'train' else 'target'

    # Gọi hàm training với các tham số từ dòng lệnh
    run_training(
        config_path=args.config,
        model_save_path=args.save_path,
        model_load_path=args.load_path,
        env_type=env_type_str
    )