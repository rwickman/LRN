import metaworld
import random
import torch
import numpy as np

from trainer import Trainer

SEED = 0  # some seed number here
# benchmark = metaworld.BENCHMARK(seed=SEED)

class MetaWorldTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def train(self):
        # print(
        #     self._fb_model(
        #         torch.randn(39).unsqueeze(0).cuda(),
        #         torch.tensor( list( range(1,5) ) ).unsqueeze(0).cuda())
        #     )
        ml10 = metaworld.MT10()

        training_envs = []
        for name, env_cls in ml10.train_classes.items():
            print("name", name)
            env = env_cls()
            task = random.choice([task for task in ml10.train_tasks
                                if task.env_name == name])
            env.set_task(task)
            training_envs.append(env)

        for env in training_envs:
            print(env.name)
            obs = env.reset()  # Reset environment
            obs = torch.tensor(obs.astype(np.float32)).cuda().unsqueeze(0)
            actor_actions, _ = self._fb_model(obs, torch.tensor( list( range(1,5) ) ).unsqueeze(0).cuda() ) 
            a = env.action_space.sample()  # Sample an action
            obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action