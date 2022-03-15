from networkagent import NetworkAgent
from connection import Connection
from machine import Machine
from queue import PriorityQueue
from message import Message
from predatorworld import PredatorWorld
class Machine:
    def __init__(self,agent:NetworkAgent,name:str):
        self.clock = 0
        self.agent = agent
        self.world:PredatorWorld = self.agent.world
        self.name = name
        self.messages = PriorityQueue()
        self.connections:dict[str,Connection] = dict()
    
    def add_connection(self,to_machine:Machine,connection:Connection):
        self.connections[to_machine.name] = connection
    
    def receive_message(self,message:Message):
        self.messages.put((message.time,message.content))
    
    def activate(self,t:int):
        self.clock = t
        while((not self.messages.empty()) and self.messages.queue[0][0] <= self.clock):
            self.world.process(self.messages.get()[1])
        update = self.agent.take_action() #TODO need to alter returns to get "message.content" 
        for connection in self.connections.values():
            connection.send(update)
