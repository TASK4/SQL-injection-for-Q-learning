# reward_system.py 
class RewardSystem:
    """
    Tính toán phần thưởng (reward) dựa trên phản hồi (response) từ môi trường.
    """
    def __init__(self, normal_count, success_marker, error_marker):
        self.normal_count = normal_count
        # Sanitize markers: remove accidental surrounding quotes
        self.success_marker = success_marker.strip('\"').strip("'") if isinstance(success_marker, str) else success_marker
        self.error_marker = error_marker.strip('\"').strip("'") if isinstance(error_marker, str) else error_marker

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
        text = response.text or ""

        # Debug: show short snippet to help matching during training
        # (Bạn có thể comment dòng print này lại khi chạy huấn luyện chính thức)
        try:
            snippet = text[:300]
        except Exception:
            snippet = str(text)
        # print(f"[RewardSystem Debug] status={status}, payload='{payload}', snippet={snippet!r}")

        # Try parsing JSON body (if any) to search for success marker inside fields
        json_body = None
        try:
            json_body = response.json()
        except Exception:
            json_body = None

        def contains_marker(obj):
            """Recursively search strings inside JSON-like structures for the success marker."""
            if obj is None:
                return False
            if isinstance(obj, str):
                return self.success_marker in obj
            if isinstance(obj, dict):
                for v in obj.values():
                    if contains_marker(v):
                        return True
                return False
            if isinstance(obj, list):
                for i in obj:
                    if contains_marker(i):
                        return True
                return False
            return False

        # 1. THÀNH CÔNG LỚN: Khai thác thành công
        has_keywords = "UNION" in payload.upper() and "USERS" in payload.upper() and "SELECT" in payload.upper()
        has_valid_start = payload.strip().startswith("'") or payload.strip().startswith("a") or payload.strip().startswith("b")
        has_marker = self.success_marker in text or contains_marker(json_body)
        is_cheating = "PRODUCTS" in payload.upper()

        if status == 200 and has_marker and has_keywords and has_valid_start:
            if is_cheating:
                print(f"[RewardSystem] --- CHEAT DETECTED (Found marker but also 'Products'): Payload: {payload}")
                return -10, False  # Phạt vì cố tình "cheat"
            print(f"\n[RewardSystem] !!! SUCCESS: Payload: {payload}\n")
            return 200, True  # Thưởng rất lớn, kết thúc episode

        # 2. LỖI SQL CÓ GỢI Ý: Phạt nhẹ hơn, nhưng vẫn phạt
        if status == 500 and self.error_marker in text:
            if "SELECT" in payload.upper():
                return -3, False  # Hint nhưng chưa full
            return -5, False

        # 3. DẤU HIỆU XẤU: Lỗi server chung
        if status == 500:
            return -25, False

        # 4. DẤU HIỆU XẤU: Bị chặn
        if status in [401, 403, 429]:
            return -50, False

        # 5. BÌNH THƯỜNG / KHÔNG HIỆU QUẢ (Status 200)
        if status == 200:
            try:
                data_list = response.json().get('data', [])
                count = len(data_list)
            except:
                count = self.normal_count  # Fallback

            if "UNION" in payload.upper():
                # ANTI-FARM: Chỉ thưởng nếu hint thực sự (mismatch/reduced rows)
                if count == 0:
                    # Empty: Good hint for column tune
                    bonus = 10 if payload.lower().count("null") >= 5 else 5  # Bonus nếu có NULLs
                    return bonus, False
                elif count < self.normal_count:
                    # Reduced: Partial match
                    return 5, False
                else:
                    # Normal: Wasted UNION
                    return -5, False
            else:
                # No UNION: Normal search
                if count == 0:
                    return -2, False
                if count == self.normal_count:
                    return -1, False
                return 0, False  # Neutral if unexpected count

        # Trường hợp mặc định (e.g., incomplete UNION without SELECT)
        if "UNION" in payload.upper() and "SELECT" not in payload.upper():
            return -2, False  # Penalty for incomplete

        return -1, False