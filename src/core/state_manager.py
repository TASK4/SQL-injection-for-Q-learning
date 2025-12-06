# state_manager.py 
import hashlib  # Move up here

class StateManager:
    """
    Quản lý và theo dõi trạng thái hiện tại (payload).
    Trong bài toán này, "trạng thái" chính là chuỗi payload
    mà agent đang xây dựng.
    """
    def __init__(self):
        self.current_state = ""
        print("[StateManager] Đã khởi tạo.")

    def get_current_state(self):
        """Lấy payload hiện tại."""
        return self.current_state

    def reset_state(self):
        """
        Reset payload về rỗng.
        Được gọi mỗi khi bắt đầu một episode (lần thử) mới.
        """
        self.current_state = ""
        return self.current_state

    def update_state(self, action_string):
        """
        Cập nhật trạng thái bằng cách nối thêm mảnh hành động (action_string)
        vào payload hiện tại. Use hash for compact state if long.
        """
        self.current_state += action_string
        # Hash last 20 chars để state compact (giảm explosion)
        if len(self.current_state) > 20:
            state_hash = hashlib.md5(self.current_state[-20:].encode()).hexdigest()
            return state_hash  # Return hash as state for Q-table (string)
        return self.current_state  # Short: use full string (string)