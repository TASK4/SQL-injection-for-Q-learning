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
        vào payload hiện tại.
        """
        self.current_state += action_string
        return self.current_state