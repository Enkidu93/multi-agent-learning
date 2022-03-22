from networkagent import NetworkAgent
from queue import PriorityQueue
from message import Message
from world import World

class Connection:
    def __init__(self,from_machine,to_machine,generate_delay):
        self.from_machine = from_machine
        self.to_machine = to_machine
        self.generate_delay = generate_delay

    def send(self,message):
        time = self.from_machine.clock + self.generate_delay()
        self.to_machine.receive_message(Message(message,time))

class Machine:
    def __init__(self,agent:NetworkAgent,name:str):
        self.clock = 0
        self.agent = agent
        self.world:World = self.agent.world
        self.name = name
        self.messages = PriorityQueue()
        self.connections:dict[str,Connection] = dict()
    
    def add_connection(self,to_machine,connection:Connection):
        self.connections[to_machine.name] = connection
    
    def receive_message(self,message:Message):
        self.messages.put((message.time,message.content))
    
    def activate(self,t:int):
        self.clock = t
        while((not self.messages.empty()) and self.messages.queue[0][0] <= self.clock):
            self.world.process(self.messages.get()[1])
        update = self.agent.take_action()
        for connection in self.connections.values():
            connection.send(update)
