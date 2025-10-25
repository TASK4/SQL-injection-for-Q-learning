class RewardSystem:
    """
    Tính toán phần thưởng (reward) dựa trên phản hồi (response) từ môi trường.
    """
    def __init__(self, normal_count, success_marker, error_marker):
        self.normal_count = normal_count
        self.success_marker = success_marker
        self.error_marker = error_marker

    def calculate_reward(self, response, payload):
        """
        Trả về: (reward, done)
        reward: Điểm thưởng (int)
        done: Kết thúc episode (bool)
        """
        if response is None:
            # Phạt nặng nếu request bị lỗi (timeout, connection...)
            return -20, False

        status = response.status_code
        text = response.text

        # 1. THÀNH CÔNG LỚN: Khai thác thành công
        if status == 200 and self.success_marker in text:
            print(f"\n[RewardSystem] !!! SUCCESS: Payload: {payload}")
            return 200, True  # Thưởng rất lớn, kết thúc episode

        # 2. DẤU HIỆU TỐT: Gây ra lỗi SQL
        if status == 500 and self.error_marker in text:
            print(f"[RewardSystem] --- GOOD SIGN (SQL Error): Payload: {payload}")
            return 50, False  # Thưởng khá, tiếp tục xây dựng payload

        # 3. DẤU HIỆU XẤU: Lỗi server chung
        if status == 500:
            return -25, False # Phạt vì gây lỗi không mong muốn

        # 4. DẤU HIỆU XẤU: Bị chặn
        if status in [401, 403, 429]:
            return -50, False # Phạt nặng (WAF, auth...)

        # 5. BÌNH THƯỜNG / KHÔNG HIỆU QUẢ
        if status == 200:
            try:
                # Kiểm tra số lượng kết quả trả về
                count = len(response.json().get('data', []))
                if count == 0:
                    return -2, False # Phạt nhẹ (không có kết quả)
                if count == self.normal_count:
                    return -1, False # Phạt rất nhẹ (kết quả bình thường)
            except:
                return -5, False # Phản hồi không phải JSON

        # Trường hợp mặc định
        return -1, False