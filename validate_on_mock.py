# validate_on_mock.py (ĐÃ SỬA)
import configparser
import logging
import argparse
import os
import sys

# Thêm đường dẫn để import được module src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environment.training_environment import TrainingEnvironment
from src.agent.q_learning_agent import QLearningAgent

def run_validation(config_path, model_path, attempts):
    # Setup Log
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info(f"--- KIỂM TRA MODEL: {model_path} ---")

    if not os.path.exists(model_path):
        logging.error(f"❌ Không tìm thấy file model: '{model_path}'")
        return

    try:
        # Load Config & Environment
        env = TrainingEnvironment(config_path)
        
        # Load Agent
        agent = QLearningAgent(
            action_space_size=env.get_action_space_size(),
            lr=0.0, gamma=0.0, epsilon=0.0, epsilon_decay=0.0, epsilon_min=0.0
        )
        agent.load_model(model_path)
        logging.info("--> Load môi trường và Model thành công.\n")
        
    except Exception as e:
        logging.error(f"Lỗi khởi tạo: {e}")
        return

    success_count = 0
    
    for attempt in range(attempts):
        logging.info(f">>> Lần thử {attempt + 1}")
        state = env.reset()
        done = False
        full_payload = ""
        
        # Lấy số cột ẩn hiện tại để đối chiếu
        hidden_cols = env.current_hidden_col_count
        logging.info(f"    (Môi trường đang có {hidden_cols} cột ẩn)")

        for step in range(50):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            
            full_payload = env.state_manager.current_state

            # --- SỬA LỖI LOGIC CHIẾN THẮNG ---
            # Reward System trả về done=True khi lấy được 'FROM USERS'
            # Hoặc reward >= 20.0 (Ngưỡng thắng thực tế)
            if done or reward >= 20.0: 
                logging.info(f"🏆 CHIẾN THẮNG tại bước {step+1}!")
                logging.info(f"✅ Payload: {full_payload}")
                logging.info(f"✅ Reward: {reward}")
                success_count += 1
                break
        
        if not done and reward < 20.0:
            logging.info(f"❌ Thất bại. Payload dừng ở: {full_payload}")

    logging.info(f"\n=== KẾT QUẢ: Thắng {success_count}/{attempts} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/config_training.ini")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--attempts', type=int, default=5)
    args = parser.parse_args()

    run_validation(args.config, args.model, args.attempts)