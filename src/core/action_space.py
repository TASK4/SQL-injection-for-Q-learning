class ActionSpace:
    def __init__(self):
        # Bộ mảnh ghép (Puzzle Pieces)
        self.actions = [
            # 1. Entry & Exit
            "a'))", 
            "--",
            
            # 2. SQL Syntax
            " UNION",
            " SELECT",
            " FROM Users", 
            
            # 3. Values & Separators
            " NULL",
            ",",
            ", NULL",  # [NEW] Thêm hành động này để điền cột nhanh gấp đôi
            
            # 4. Payload Data
            " id", 
            " email", 
            " password"
        ]
        self.num_actions = len(self.actions)

    def get_action_string(self, index):
        return self.actions[index]

    def get_action_space_size(self):
        return self.num_actions