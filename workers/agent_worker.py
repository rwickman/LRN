

import torch
from dataclasses import dataclass

@dataclass
class StateMessage:
    state: torch.Tensor
    done: bool = False
    reward: float = 0.0

class AgentWorker:
    def __init__(self, args, env, state_queue, action_queue):
        self.args = args
        self.env = env
        self.state_queue = state_queue
        self.action_queue = action_queue
        self._ex_buffer = []
        self.done = True
    

    def run(self):
        while True:
            if self.done:
                state = self.env.reset()
                self.state_queue.put(
                    StateMessage(state))
            self.env.render()
            action_msg = self.action_queue.get()
            if action_msg.stop:
                return
            
            state, reward, self.done, _ = self.env.step(action_msg.action)
            if self.done:
                state = self.env.reset()
                self.state_queue.put(
                    StateMessage(state, self.done, reward))
                self.done = False
            else:
                self.state_queue.put(
                    StateMessage(state, self.done, reward))
                
            
            if action_msg.reset:
                print("RESETING")
                self.done = True

        

        
        
