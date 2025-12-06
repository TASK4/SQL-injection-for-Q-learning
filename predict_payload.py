import random
import numpy as np
import configparser
import sys
import os  # Cần cho việc tắt 'print'
import logging

# --- Import các thành phần CỐT LÕI ---
from src.agent.q_table import QTable
from src.core.action_space import ActionSpace
# --- Import các thành phần MÔI TRƯỜNG (để tự kiểm tra) ---
from src.utils.http_client import HttpClient
from src.core.reward_system import RewardSystem


# --- CÁC THAM SỐ ---
CONFIG_FILE = "config/config_target.ini"
MODEL_FILE = "results/target_results/juiceshop_model_v1.json"
# Tăng số lần thử lên 200 (bạn có thể tăng lên 1000 nếu muốn)
NUMBER_OF_ATTEMPTS = 200
PREDICT_EPSILON = 0.8  # Tăng lên 50% cơ hội chọn ngẫu nhiên để explore hơn
# --------------------

# --- THÊM 3 DÒNG CÀI ĐẶT LOGGING ---
# Đảm bảo thư mục tồn tại trước khi cài đặt FileHandler
os.makedirs("results/target_results", exist_ok=True)
LOG_FILE = "results/target_results/predict_log.txt"
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), logging.StreamHandler()])
# -----------------------------------


def run_automated_prediction():
    """
    Tự động tạo payload và kiểm tra chúng cho đến khi thành công
    hoặc hết số lần thử.
    """
    logging.info(f"--- Bắt đầu tự động dự đoán (tối đa {NUMBER_OF_ATTEMPTS} lần) ---")
    
    # 1. Đọc Cấu hình
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE, encoding='utf-8')
    try:
        max_steps = int(config['Training']['max_steps_per_episode'])
        search_url = config['Target']['search_url']
        search_param = config['Target'].get('search_param', 'q')
        normal_count = int(config['Environment']['normal_result_count'])
        success_marker = config['Environment']['success_marker']
        error_marker = config['Environment']['sql_error_marker']
    except KeyError as e:
        logging.error(f"Lỗi: Thiếu key {e} trong file config '{CONFIG_FILE}'. Dừng lại.")
        return
    except Exception as e:
        logging.error(f"Lỗi khi đọc config: {e}. Dừng lại.")
        return

    # 2. Khởi tạo các thành phần
    action_space = ActionSpace()
    q_table = QTable(action_space.get_action_space_size())
    http_client = HttpClient()
    reward_system = RewardSystem(normal_count, success_marker, error_marker)

    # 3. Tải model "bộ não"
    try:
        q_table.load(MODEL_FILE)
        logging.info(f"[QTable] Đã tải model từ: {MODEL_FILE}")
    except FileNotFoundError:
        logging.error(f"Lỗi: Không tìm thấy tệp model tại '{MODEL_FILE}'. Dừng lại.")
        return
    except Exception as e:
        logging.error(f"Lỗi khi tải model: {e}. Dừng lại.")
        return

    # 4. Vòng lặp TỰ ĐỘNG THỬ NGHIỆM
    found_success = False
    for attempt in range(NUMBER_OF_ATTEMPTS):
        logging.info(f"\n--- Lần thử {attempt + 1}/{NUMBER_OF_ATTEMPTS} ---")
        
        # --- Logic tạo payload (từ các bước trước) ---
        current_state = ""
        generated_payload = ""

        for step in range(max_steps):
            q_values = q_table.get(current_state)
            
            # REMOVE BREAK EARLY: Không còn check np.all(q_values == 0) để luôn full steps
            
            # Epsilon-greedy với PREDICT_EPSILON
            if random.uniform(0, 1) < PREDICT_EPSILON:
                best_action_index = random.randint(0, action_space.get_action_space_size() - 1)
                logging.debug(f"  Step {step}: Explore random action {best_action_index}")
            else:
                best_action_index = np.argmax(q_values)
                logging.debug(f"  Step {step}: Exploit best action {best_action_index} (Q-max: {np.max(q_values):.2f})")
            
            action_string = action_space.get_action_string(best_action_index)
            generated_payload += action_string
            current_state = generated_payload  # Luôn update state
            
            # OPTIONAL: Early stop chỉ nếu quá dài (tránh spam)
            if len(generated_payload) > 120 and step > 25:
                logging.info(f"  (Payload quá dài {len(generated_payload)} chars, dừng sớm tại step {step})")
                break
        
        logging.info(f"Đã tạo Payload: {generated_payload!r} (dài {len(generated_payload)} chars)")

        # 5. TỰ ĐỘNG KIỂM TRA PAYLOAD
        logging.info("  Đang gửi payload để kiểm tra...")
        response = http_client.send_search_query(search_url, generated_payload, search_param)
        
        # Tạm thời "tắt tiếng" của RewardSystem để terminal đỡ bị spam
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            reward, done = reward_system.calculate_reward(response, generated_payload)
        finally:
            sys.stdout.close()  # Đóng file null
            sys.stdout = original_stdout  # Khôi phục print bình thường
        
        # 6. Kiểm tra kết quả
        if done:  # 'done' là True khi reward == 200 (thành công)
            logging.info(f"\n========================================")
            logging.info(f"!!! TẤN CÔNG THÀNH CÔNG (Reward={reward}) !!!")
            logging.info(f"Payload chiến thắng: {generated_payload}")
            logging.info(f"========================================")
            found_success = True
            break  # Thoát khỏi vòng lặp 'for attempt...'
        else:
            logging.info(f"  Kết quả: Thất bại (Reward={reward}). Đang thử lại...")

    if not found_success:
        logging.info(f"\n--- Đã hết {NUMBER_OF_ATTEMPTS} lần thử. Không tìm thấy payload thành công. ---")
        logging.info("Mẹo: Hãy thử tăng 'NUMBER_OF_ATTEMPTS' trong file này, hoặc huấn luyện (main.py) thêm.")


# --- Chạy hàm chính ---
if __name__ == "__main__":
    run_automated_prediction()