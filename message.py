
class Message():
    def __init__(self, content, time) -> None:
        self.content = content
        self.time = time
    
    def __lt__(self,other):
        return self.time < other.time