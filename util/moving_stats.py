import json
import os

class MovingStats:
    def __init__(self, args):
        self.args = args
        self.mean = None
        self.std = None
        self.max_val = None
        self.w = 0.01
        if self.args.load:
            self.load()
    
    def standardize_rewards(self, rewards):
        if self.mean is None or self.max_val is None:
            self.mean = float(rewards.mean())
            self.std = float(rewards.std())
        else:
            self.max_val = max(float(rewards.max()), self.max_val)
            self.mean = self.mean * (1-self.w) + float(rewards.mean()) * self.w
            self.std = self.std * (1-self.w) + float(rewards.std()) * self.w
            
        rewards = rewards  / self.std

        return rewards


    def save(self):
        with open(os.path.join(self.args.save_dir, "rewards_stats.json"), "w") as f:
            d = {"mean" : self.mean, "std" : self.std, "max_val": self.max_val}
            print(d)
            json.dump(d, f)
        
    def load(self):
        load_file = os.path.join(self.args.save_dir, "rewards_stats.json") 
        if os.path.exists(load_file):
            with open(load_file, "r") as f:
                d = json.load(f)
                self.std = d["std"]