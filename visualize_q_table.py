import json
import csv
import os
import sys

# Thêm thư mục 'src' vào đường dẫn để có thể import ActionSpace
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    from core.action_space import ActionSpace # Dùng để lấy tiêu đề cột
except ImportError:
    print("Lỗi: Không thể import ActionSpace từ 'src/core/action_space.py'.")
    print("Hãy đảm bảo bạn chạy file này từ thư mục gốc của dự án.")
    sys.exit(1)


def convert_json_to_csv():
    """
    Đọc file model Q-Table (JSON) và chuyển đổi nó
    thành file Bảng tính (CSV) để xem cho dễ.
    """
    
    # Đường dẫn đến file "bộ não" JSON
    model_path = "results/target_results/juiceshop_model_v1.json"
    
    # Tên file bảng tính sẽ được xuất ra
    csv_path = "results/target_results/q_table_visual.csv"

    print(f"Đang đọc model từ: {model_path}")
    print(f"Đang đọc các cột hành động từ: src/core/action_space.py")

    # 1. Lấy 23 "cột" (Hành động) từ file ActionSpace
    action_space = ActionSpace()
    # Đây sẽ là tiêu đề (header) của bảng tính
    # Ví dụ: ['State', ''', ' ', ')', '(', 'OR', 'AND', ...]
    headers = ['State (Trang_thai)'] + action_space.actions

    # 2. Đọc file JSON "bộ não"
    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            q_table_data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file model '{model_path}'.")
        print("Bạn cần chạy 'main.py' để huấn luyện và tạo file này trước.")
        return
    except json.JSONDecodeError:
        print(f"Lỗi: File model '{model_path}' bị hỏng hoặc không phải JSON.")
        return

    # 3. Ghi ra file CSV (file Bảng tính)
    try:
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Ghi hàng tiêu đề (hàng đầu tiên trong Excel)
            writer.writerow(headers)
            
            # Ghi từng hàng dữ liệu
            # (ví dụ: "'", 0.53, -0.1, 0.179...)
            for state, q_values in q_table_data.items():
                # Nối [State] với [dãy Q-values]
                writer.writerow([state] + list(q_values)) # Chuyển q_values (numpy array) thành list

        print(f"\n--- HOÀN TẤT ---")
        print(f"Đã chuyển đổi thành công Q-Table sang bảng tính.")
        print(f"File của bạn đã được lưu tại: {csv_path}")


    except Exception as e:
        print(f"Lỗi khi ghi file CSV: {e}")

if __name__ == "__main__":
    convert_json_to_csv()