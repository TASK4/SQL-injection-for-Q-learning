class ActionSpace:
    def __init__(self):
        # Bộ mảnh ghép (Puzzle Pieces)
        self.actions = [
            # 1. Entry & Exit (Cửa vào và Cửa ra)
            "a'))", 
            "--",
            
            # 2. SQL Syntax (Khung sườn)
            " UNION",
            " SELECT",
            " FROM Users", # Mảnh này hơi to, nhưng tạm chấp nhận cho bài toán này
            
            # 3. Values & Separators (Gạch vữa để xây)
            " NULL",
            ",",
            
            # 4. Payload Data (Nội thất bên trong)
            " id", 
            " email", 
            " password"
        ]
        self.num_actions = len(self.actions)

    def get_action_string(self, index):
        return self.actions[index]

    def get_action_space_size(self):
        return self.num_actions