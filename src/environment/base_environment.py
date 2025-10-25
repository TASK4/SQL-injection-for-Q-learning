from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả các môi trường (Env).
    """
    
    @abstractmethod
    def reset(self):
        """Reset môi trường về trạng thái ban đầu và trả về state."""
        pass
    
    @abstractmethod
    def step(self, action_index):
        """
        Thực hiện một hành động (action) và trả về:
        (next_state, reward, done)
        """
        pass
        
    @abstractmethod
    def get_action_space_size(self):
        """Trả về số lượng hành động có thể."""
        pass