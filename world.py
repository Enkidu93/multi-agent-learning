from abc import ABC, abstractmethod

class World(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod    
    def reset(self):
        pass

    @abstractmethod
    def step(self, action) -> tuple: # returns tuple where tuple[0] is reward, tuple[1] is new state, tuple[2] is episode_complete if appplicable
        pass

    @abstractmethod
    def visualize(self):
        pass
