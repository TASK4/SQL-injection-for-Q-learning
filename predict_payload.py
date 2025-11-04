import random
import numpy as np
import configparser
from src.agent.q_table import QTable  # Import lớp QTable
from src.core.action_space import ActionSpace # Import lớp ActionSpace

def predict_best_payload(config_path, model_path):
    """
    Sử dụng Q-table đã huấn luyện để tạo ra payload tối ưu nhất.
    """
    print(f"--- Bắt đầu dự đoán payload từ model: {model_path} ---")

    # 1. Đọc cấu hình để lấy max_steps
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    try:
        max_steps = int(config['Training']['max_steps_per_episode'])
        print(f"Độ dài tối đa của payload: {max_steps} bước.")
    except KeyError:
        print("Lỗi: Không tìm thấy max_steps_per_episode trong config. Đặt mặc định là 40.")
        max_steps = 40

    # 2. Khởi tạo ActionSpace và QTable
    action_space = ActionSpace()
    q_table = QTable(action_space.get_action_space_size())

    # 3. Tải model đã huấn luyện
    try:
        q_table.load(model_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp model tại '{model_path}'. Dừng chương trình.")
        return None
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return None

    # 4. Bắt đầu xây dựng payload
    current_state = ""
    generated_payload = ""

    print("\nQuá trình xây dựng payload:")
    print(f"Bước 0: State = '{current_state}'")

    for step in range(max_steps):
        # Lấy Q-values cho trạng thái hiện tại
        q_values = q_table.get(current_state)

        # Kiểm tra nếu tất cả Q-values đều là 0 (trạng thái chưa được khám phá nhiều)
        if np.all(q_values == 0):
            print(f"Bước {step+1}: Trạng thái '{current_state}' chưa được khám phá nhiều, dừng lại.")
            best_action_index = random.randint(0, action_space.get_action_space_size() - 1)
        else:
            best_action_index = np.argmax(q_values)
        action_string = action_space.get_action_string(best_action_index)

        # Cập nhật payload và trạng thái
        generated_payload += action_string
        current_state = generated_payload # Trạng thái chính là payload hiện tại

        print(f"Bước {step+1}: Chọn action '{action_string}' (Q={q_values[best_action_index]:.2f}) -> State = '{current_state}'")

        # (Tùy chọn) Thêm điều kiện dừng sớm nếu cần
        # Ví dụ: if "success_marker" in current_state: break

    print("\n--- Hoàn tất ---")
    print(f"Payload dự đoán cuối cùng: {generated_payload}")
    return generated_payload

if __name__ == "__main__":
    CONFIG_FILE = "config/config_target.ini"
    # Đường dẫn đến file model bạn đã lưu sau khi huấn luyện
    MODEL_FILE = "results/target_results/juiceshop_model_v1.json" 

    predicted = predict_best_payload(CONFIG_FILE, MODEL_FILE)
    # Bạn có thể dùng payload 'predicted' này để thử nghiệm thủ công