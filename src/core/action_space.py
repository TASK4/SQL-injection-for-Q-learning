class ActionSpace:
    """
    Định nghĩa tất cả các "hành động" (payload fragments) mà Agent có thể thực hiện.
    """
    def __init__(self):
        # Bộ actions này được tùy chỉnh cho SQLite (dùng trong Juice Shop)
        self.actions = [
            "'", " ", ")", "(", "OR", "AND", "UNION", "SELECT",
            "FROM", "Users", "Products", "password", "email",
            "1=1", "1=0", "--", "NULL", ",", "*",
            "=''",  # Thường dùng để bypass
            "a", "b", # Các ký tự đệm (filler)
            "admin"
        ]
        self.num_actions = len(self.actions)

    def get_action_string(self, index):
        """Lấy chuỗi hành động từ chỉ số (index)."""
        return self.actions[index]

    def get_action_space_size(self):
        """Trả về tổng số hành động."""
        return self.num_actions