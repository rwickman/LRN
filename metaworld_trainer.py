import metaworld
import random
import torch
import numpy as np
import time
import json

from trainer import Trainer

SEED = 0  # some seed number here
# benchmark = metaworld.BENCHMARK(seed=SEED)

# Will need train both FB Policy and Router
# Can use PPO for both

# First sample states from all envs
# Perform router based on obs
# Send batch to FB model
# FB model will return actions

# This will continue until all eps are terminated
# Will need to pad all the rewards
# First train the FB models
## Perhaps create another trainer?
# Then train the Router


class MetaWorldTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.load_train_dict()



    def train(self):
        ml10 = metaworld.MT10()

        training_envs = []
        tasks = []

        for name, env_cls in ml10.train_classes.items():
            env = env_cls()
            #print("task.env_name", [task.env_name for task in ml10.train_tasks if task.env_name == name])
            task = random.choice([task for task in ml10.train_tasks
                                if task.env_name == name])
            print("name", name)
            env.set_task(task)
            training_envs.append(env)
            tasks.append(task)

        for e_i in range(self.args.num_episodes):
            start_e_i = time.time()
            with torch.no_grad():
                done_tasks = set()
                start_time = time.time()
                obs_batch = []#torch.zeros((self.args.max_env_steps, 10, self.args.mw_obs_size)).cuda()
                task_batch = []#torch.zeros((self.args.max_env_steps, 10), dtype=torch.int64).cuda()
                cur_obs = []
                cur_tasks = []
                action_probs_batch = []
                action_log_probs_batch = []
                valid_fbs_ep_batch = []
                actions_batch = []
                action_indices_batch = []
                critic_outputs_batch = []
                max_fbs = 0
                
                # rewards shape: (# tasks, max_env_steps)
                rewards = torch.zeros((self.args.num_tasks, self.args.max_env_steps)).cuda()#self.args.max_env_steps))
                
                # (# tasks, max_env_steps, max_depth)
                critic_outputs_batch = torch.zeros((self.args.num_tasks, self.args.max_env_steps, self.args.max_depth, 1)).cuda()
                
                done_time = torch.zeros((self.args.num_tasks, self.args.max_env_steps)).cuda()
                
                for i in range(self.args.max_env_steps):
                    if len(done_tasks) >= self.args.num_tasks:
                        break
                    if i == 0:
                        for task_id, env in enumerate(training_envs):
                            # Reset environment
                            obs = env.reset()
                            cur_obs.append(obs)
                            cur_tasks.append(task_id)        

            
                    cur_obs_tensor = torch.tensor(cur_obs).float().cuda()
                    cur_tasks_tensor = torch.tensor(cur_tasks).cuda()
                    router_results = self._router(cur_obs_tensor, cur_tasks_tensor)
                    actor_actions, actor_actions_log_prob = self._fb_model(cur_obs_tensor, router_results.actions)
                    
                    action_probs_batch.extend(router_results.action_probs)
                    action_log_probs_batch.extend(actor_actions_log_prob.tolist())
                    valid_fbs_ep_batch .extend(router_results.valid_fbs_ep)
                    actions_batch.extend(router_results.actions)
                    action_indices_batch.extend(router_results.action_indices)
                    max_fbs = max(max_fbs, router_results.max_fbs)
                    
                    #critic_outputs_batch.append(router_results.critic_outputs)
                    critic_outputs_batch[cur_tasks, i] = router_results.critic_outputs
                    obs_batch.extend(cur_obs)
                    task_batch.extend(cur_tasks)
                    cur_obs = []
                    cur_tasks = []
                    for task_id, env in enumerate(training_envs): 
                        if task_id in done_tasks:
                            continue
                        
                        a = env.action_space.sample()  # Sample an action
                        obs, reward, done, info = env.step(actor_actions[0].detach().cpu().numpy())  # Step the environoment with the sampled random action
                        if done:
                            assert task_id not in done_tasks  
                            done_time[task_id, i+1:] = 1 
                            done_tasks.add(task_id)
                        elif i + 1 < self.args.max_env_steps:
                            rewards[task_id][i] = reward
                            cur_obs.append(obs)
                            cur_tasks.append(task_id)

                print("End time", time.time() - start_time)
                
                
                task_rewards_total = rewards.sum(axis=1)
                print("task_rewards_total", task_rewards_total)
                
                # Standardize rewards
                rewards = (rewards - rewards.mean(axis=1).unsqueeze(1)) / (rewards.std(axis=1).unsqueeze(1) + 1e-7)
                
                # Compute returns for the robot policies
                returns_mt, advs_mt, tdlamret_mt = self.compute_return(rewards, critic_outputs_batch[:, :, -1])

                # Reshape to same order as occurred in running episodes
                returns_mt = returns_mt.transpose(0,1).reshape(-1)
                advs_mt = advs_mt.transpose(0,1).reshape(-1)
                tdlamret_mt = tdlamret_mt.transpose(0,1).reshape(-1)
                critic_outputs_batch = critic_outputs_batch.transpose(0,1).reshape(-1, self.args.max_depth, 1)        


                # Set rewards for Router based on return of tasks
                rewards_router = torch.zeros(self.args.num_tasks * self.args.max_env_steps, self.args.max_depth).cuda()
                rewards_router[:, -1] = returns_mt

                # Remove non-existent timesteps
                done_time = done_time.transpose(0,1).reshape(-1)
                returns_mt = returns_mt[done_time == 0]
                advs_mt = advs_mt[done_time == 0]
                tdlamret_mt = tdlamret_mt[done_time == 0]
                critic_outputs_batch = critic_outputs_batch[done_time == 0]
                rewards_router = rewards_router[done_time == 0]
                
                # Compute returns for Router policy
                returns_router, advs_router, tdlamret_router = self.compute_return(rewards_router, critic_outputs_batch)
                
                # Reshape to same order as occurred in running episodes
                returns_router = returns_router.transpose(0,1).reshape(-1)
                advs_router = advs_router.transpose(0,1).reshape(-1)
                tdlamret_router = tdlamret_router.transpose(0,1).reshape(-1)


            # Train the robot policies
            robot_actor_losses = self.train_robot(
                torch.tensor(action_log_probs_batch).cuda(),
                torch.tensor(obs_batch).float().cuda(),
                torch.tensor(actions_batch, dtype=torch.int32),
                advs_mt)


            # Train the Router
            actor_losses, critic_losses, load_balance_losses = self.train_router(
                valid_fbs_ep_batch,
                actions_batch,
                action_indices_batch,
                action_probs_batch,
                max_fbs,
                torch.tensor(obs_batch).float().cuda(),
                torch.tensor(task_batch, dtype=torch.int64).cuda(),
                returns_router,
                advs_router,
                tdlamret_router
            )

            # Save episode results
            self.train_dict["router_actor_losses"].append(actor_losses)
            self.train_dict["critic_losses"].append(critic_losses)
            self.train_dict["load_balance_losses"].append(load_balance_losses)
            self.train_dict["robot_actor_losses"].append(robot_actor_losses)
            self.train_dict["task_rewards_total"].append(task_rewards_total.cpu().tolist())
            self.save_episode()

            # Save models
            self.save()
            print("Total Episode time", time.time() - start_e_i)
        #returns_router, advs_router, tdlamret_router = self.compute_return(rewards_router, critic_outputs_batch.reshape(-1, self.args.max_depth, 1))
        # print("returns_router", returns_router)
        # print("rewards", rewards[:, -1])
        # print("returns", returns_mt[:, -1])


    def train_robot(self, old_log_prob_batch, obs_batch, actions_batch, advs_mt):
        num_exs = len(obs_batch)
        robot_actor_losses = []
        for e_i in range(self.args.mw_noptepochs):
            batch_ids = torch.randperm(num_exs)
            for m_i in range(self.args.mw_nminibatches):
                start = m_i * (num_exs // self.args.mw_nminibatches)
                end = (m_i + 1) * (num_exs // self.args.mw_nminibatches)
                m_batch_idxs = batch_ids[start:end]
                _, new_log_prob = self._fb_model(obs_batch[m_batch_idxs], actions_batch[m_batch_idxs])
                policy_loss = self.ppo_loss_robot(advs_mt[m_batch_idxs], old_log_prob_batch[m_batch_idxs], new_log_prob)
                print("policy_loss", policy_loss)
                policy_loss.backward()
                self.fb_optim.step()
                self.fb_optim.zero_grad()
                robot_actor_losses.append(float(policy_loss.detach().cpu().numpy()))

        return robot_actor_losses

    def ppo_loss_robot(self, advs, old_log_prob, new_log_prob):
        ratio = torch.exp(new_log_prob - old_log_prob)
        surrogate_1 = ratio * advs

         # Clipped policy gradient, ratio clipped between [0.8, 1.2]
        surrogate_2 = torch.clip(ratio, 1.0 - self.args.eps_clip, 1.0 + self.args.eps_clip) * advs
        policy_loss = -torch.mean(torch.min(surrogate_1, surrogate_2))
        return policy_loss
    
    
    def load_train_dict(self):
        if self.args.load:
            with open("models/train_dict.json") as f:
                self.train_dict = json.load(f)
        else:
            self.train_dict = {
                "critic_losses" : [],
                "router_actor_losses" : [],
                "robot_actor_losses" : [],
                "load_balance_losses" : [],
                "task_rewards_total" : []
            }

    def save_episode(self):
        print(self.train_dict)
        with open("models/train_dict.json", "w") as f:
            json.dump(self.train_dict, f)