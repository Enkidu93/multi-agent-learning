from newnetworkagent import NetworkAgent
from queue import PriorityQueue
# from message import Message
from complexmessage import ComplexMessage
from world import World
from math import sqrt, atan2, pi
import sys

class Connection:
    def __init__(self,from_machine,to_machine,generate_delay):
        self.from_machine = from_machine
        self.to_machine = to_machine
        self.generate_delay = generate_delay
        self.cur_seq_num = 0

    def send(self,message):
        time = self.from_machine.clock + self.generate_delay()
        proposal = ""
        if self.from_machine.world.suspect_episode_complete:
            proposal = "proposal"
        self.to_machine.receive_message(ComplexMessage(message,time,self.cur_seq_num,proposal=proposal,sender=self.from_machine.name))
        # self.to_machine.receive_message(ComplexMessage(message,time,self.cur_seq_num,sender=self.from_machine.name))
        self.cur_seq_num += 1

class Machine:
    def __init__(self,agent:NetworkAgent,name:str):
        self.clock = 0
        self.agent = agent
        self.world = self.agent.world
        self.name = name
        self.messages = PriorityQueue()
        self.connections:dict[str,Connection] = dict()
        self.last_received_from:dict[str,int] = dict()
    
    def add_connection(self,to_machine,connection:Connection):
        self.connections[to_machine.name] = connection
    
    def receive_message(self,message:ComplexMessage):
        self.messages.put(message)
    
    def process(self, message:ComplexMessage):
        # out = ["Sender: " + str(message.sender) + " SEQ: " + str(message.seq_num) + " Time: " + str(message.time) + " Proposal: " + message.proposal for message in self.messages.queue]
        # print(self.agent.world.action_count)
        # sys.exit(0)
        # print(self.name, ":", out)
        if message.seq_num > self.last_received_from.get(message.sender,-1):
            if message.proposal == "proposal":
                sender_state = self.world.dictionary.get(message.sender) # BETTER TO MAKE THIS A FUNCTION IN WORLD!!!
                sender_x = sender_state[0]
                sender_y = sender_state[1]
                sender_theta = sender_state[2]
                for a in self.world.agents:
                    a_abs_pos = self.world.dictionary.get(a.name)
                    a_x = a_abs_pos[0]
                    a_y = a_abs_pos[1]
                    if a.name != message.sender and sqrt((a_x - sender_x)**2 + (a_y - sender_y)**2)<=2 and (atan2((a_y-sender_y),(a_x-sender_x))-pi/5)%(2*pi)<=sender_theta%(2*pi) and (atan2((a_y-sender_y),(a_x-sender_x))+pi/5)%(2*pi)>=sender_theta%(2*pi):
                        self.world.episode_complete = True
                        a.reward -= 400

            self.world.process(message)
            self.last_received_from[message.sender] = message.seq_num
        else:
            pass
            # print(self.name, "discarded packet", message.seq_num, "because it was less recent than", self.last_received_from.get(message.sender,-1))
    
    def activate(self,t:int):
        self.clock = t
        while((not self.messages.empty()) and self.messages.queue[0].time <= self.clock):
            self.process(self.messages.get())
        update = self.agent.take_action()
        for connection in self.connections.values():
            connection.send(update)
        return (update[2], update[3])
