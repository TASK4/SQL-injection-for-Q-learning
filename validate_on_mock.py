# validate_on_mock_fixed.py
import configparser
import logging
import argparse
import os
import sys

# Giả định đường dẫn import như cũ
from src.environment.training_environment import TrainingEnvironment
from src.agent.q_learning_agent import QLearningAgent

def run_validation(config_path, model_path, attempts):
    output_dir = "results/train_results"
    os.makedirs(output_dir, exist_ok=True)
    LOG_FILE = os.path.join(output_dir, "validate_log.txt")

    # Reset logger
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    logging.basicConfig(level=logging.INFO, 
                        format='%(message)s', # Rút gọn format cho dễ nhìn
                        handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), 
                                  logging.StreamHandler()])
    
    logging.info(f"--- BẮT ĐẦU KIỂM TRA MODEL ---")

    if not os.path.exists(model_path):
        logging.error(f"❌ LỖI: Không tìm thấy file model: '{model_path}'")
        return

    try:
        # Khởi tạo Env
        env = TrainingEnvironment(config_path)
        
        # Khởi tạo Agent (Test mode: epsilon=0)
        agent = QLearningAgent(
            action_space_size=env.get_action_space_size(),
            lr=0.0, gamma=0.0, epsilon=0.0, epsilon_decay=0.0, epsilon_min=0.0
        )
        agent.load_model(model_path)
        logging.info("--> Đã load Model thành công.")
        
    except Exception as e:
        logging.error(f"Lỗi khởi tạo: {e}")
        return

    success_count = 0
    
    for attempt in range(attempts):
        logging.info(f"\n>>> Lần thử {attempt + 1}/{attempts}")
        state = env.reset()
        done = False
        step_count = 0
        current_payload = ""
        
        # Lấy max_steps từ config hoặc mặc định 50
        max_steps = 50 

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            step_count += 1
            
            # Cập nhật payload
            current_payload = env.state_manager.current_state

            # --- [SỬA ĐỔI QUAN TRỌNG] ---
            # Dựa vào reward system: Max reward cho bước cuối là 20.0
            # Hoặc kiểm tra cờ done = True (đã tìm thấy FROM USERS)
            if done and reward >= 20.0: 
                logging.info(f"🏆 WIN tại bước {step_count}!")
                logging.info(f"--> PAYLOAD: {current_payload}")
                logging.info(f"--> Reward: {reward}")
                success_count += 1
                break
                
            if step == max_steps - 1:
                logging.info(f"⏳ Hết giờ. Payload cuối: {current_payload}")

    logging.info(f"\n=== KẾT QUẢ: Thắng {success_count}/{attempts} ===")

if __name__ == "__main__":
    # Chạy test nhanh
    # Bạn cần đảm bảo đường dẫn config và model đúng
    parser = argparse.ArgumentParser(description="Validate Q-Learning model on mock environment")