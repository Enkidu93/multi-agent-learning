from message import Message

class ComplexMessage(Message):
    def __init__(self, content, time, seq_num=None, sender="", proposal="") -> None:
        super().__init__(content, time)
        self.seq_num = seq_num
        self.sender = sender
        self.proposal = proposal # 'proposal', 'yea', 'nay'