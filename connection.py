from machine import Machine
from message import Message
class Connection:
    def __init__(self,from_machine:Machine,to_machine:Machine,generate_delay):
        self.from_machine = from_machine
        self.to_machine = to_machine
        self.generate_delay = generate_delay

    def send(self,message):
        time = self.from_machine.clock + self.generate_delay()
        self.to_machine.receive_message(Message(message,time))
