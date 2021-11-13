import metaworld
import random
import torch
import numpy as np
import time
import json
import os

from trainer import Trainer
from configs import ale_config
from torch.distributions import Categorical
import cv2
import os, json

class ALETrainer(Trainer):
    def __init__(self, args, mb_model, env):
        super().__init__(args, mb_model)
        self.args = args
        self.env = env
        #self.env = AtariPreprocessing(gym.make('PongNoFrameskip-v4'), frame_skip=4, scale_obs=True)
       
        
        if self.args.load:
            self.load_results()
        else:
            self._ep_returns = []
            self._train_dict = {
                "ep_returns" : []
            }



    def train(self):
        for e_i in range(self.args.num_episodes):
            num_steps = 0
            states = []
            rewards = []
            actions = []
            ac_critic_preds = []
            ac_actor_preds = []
            router_results_list = []
            done = False
            dones = []
            
            # TODO: Look more into frame-skipping and stacking
            # Run through an episode
            self._mb_model.save_states(True)
            self._mb_model.reset()

            for i in range(self.args.num_env):
                print("EP", i)
                self._mb_model.setup_next_episode()
                num_steps = 0
                done = False
                state = self.env.reset()
                
                # Create frame stack
                state = state.unsqueeze(0).repeat(self.args.in_channels, 1, 1).unsqueeze(0)

                while not done:
                    if num_steps > 0:
                        # Update first state in frame stack
                        state = state.roll(1, dims=0)
                        state[0, 0] = next_state

                    self.env.render()
                    state = torch.tensor(state, device=ale_config.device)                   
                    states.append(state)
                    

                    with torch.no_grad():
                        # Get current actor-critic policy
                        router_results = self._router(state, torch.tensor([[0]], device=ale_config.device))
                    
                        # Use router module selections to get actor and critic predictions
                        actor_pred, critic_pred = self._mb_model(state, router_results.actions)
                        ac_critic_preds.append(critic_pred)
                        ac_actor_preds.append(actor_pred)

                    # Sample an action
                    action = self.sample_action(actor_pred[0])
                    next_state, reward, done, _ = self.env.step(action)
                    
                    # Save trajectory
                    rewards.append(reward)
                    actions.append(action)
                    router_results_list.append(router_results)

                    # Increment number of episode steps
                    num_steps += 1

            
                # Add time episode terminates for returns/advantage computation
                dones += [0] * (num_steps - 1)
                dones.append(1)
            

                
            """Train router."""
            # Set rewards of router to returns of episode
            dones = torch.tensor(dones, dtype=torch.float32, device=ale_config.device)

            ac_critic_preds = torch.cat(ac_critic_preds)    
            rewards = torch.tensor(rewards, dtype=torch.float32, device=ale_config.device) * 0.1
            returns, advs, tdlamret = self.compute_return(rewards.reshape(1, -1), ac_critic_preds.reshape(1, -1, 1), dones)
            router_rewards = torch.zeros(len(states), self.args.max_depth, device=ale_config.device)
            router_rewards[:, -1] = returns
            print("returns", returns)
            
            states = torch.cat(states)
            
            # Combine router_results
            router_results = self.agg_router_results(router_results_list)
            task_ids = torch.zeros(len(router_results_list), 1, dtype=torch.int64, device=ale_config.device)

            actor_losses, critic_losses = self.train_router(states, task_ids, router_rewards, router_results)
            self._lr_scheduler.step()
            

            
            """Train modules."""
            self._mb_model.save_states(False)
            actions = torch.tensor(actions, dtype=torch.int64, device=ale_config.device).unsqueeze(1)
            ac_actor_preds = torch.cat(ac_actor_preds)
            advs = advs.view(-1)
            tdlamret = tdlamret.view(-1)
            self.train_rl_modules(advs, tdlamret, ac_actor_preds, actions, states, router_results, ac_critic_preds)
            
            # Decay the learning rate
            self.decay_lr()
            
            # # Save models
            self.save()
            # print("Total Episode time", time.time() - start_e_i, "\n")

            self._ep_returns.append(float(rewards.sum() / self.args.num_env))
            self.save_results()


            

    def load_results(self):
        with open(os.path.join(self.args.save_dir, "ep_return.json"), "r") as f:
            self._ep_returns = json.load(f)

    def save_results(self):
        with open(os.path.join(self.args.save_dir, "ep_return.json"), "w") as f:
            json.dump(self._ep_returns, f)


    def train_rl_modules(self, advs, tdlamret, old_actor_preds, actions, states, router_results, old_critic_preds):
        batch_size = len(actions)
        print("batch_size", batch_size)

        for e_i in range(self.args.noptepochs):
            batch_ids = torch.randperm(batch_size)

            for m_i in range(self.args.nminibatches):
                loss = torch.zeros(1).cuda()
                start = m_i * (batch_size // self.args.nminibatches)
                end = (m_i + 1) * (batch_size // self.args.nminibatches)
                m_batch_idxs = batch_ids[start:end]
    

                actor_pred, critic_pred = self._mb_model(states[m_batch_idxs], router_results.actions[m_batch_idxs], m_batch_idxs)


                actor_loss, critic_loss = self.ppo_loss(
                    advs[m_batch_idxs],
                    old_actor_preds[m_batch_idxs],
                    actor_pred,
                    tdlamret[m_batch_idxs],
                    critic_pred.view(-1),
                    actions[m_batch_idxs],
                    old_critic_preds[m_batch_idxs].view(-1)
                )
                print("MODULES actor_loss", actor_loss)
                print("MODULES critic_loss", critic_loss)
                print("tdlamret.view(-1)", tdlamret.view(-1)[:10])
                print("critic_pred.view(-1)", critic_pred.view(-1)[:10])
                print("actor_pred", actor_pred[0:2])

                self.mb_optim.zero_grad()
                loss = actor_loss + self.args.critic_lam * critic_loss 
                loss.backward()
                print(torch.nn.utils.clip_grad_norm_(self._mb_model.parameters(), self.args.max_grad_norm))
                self.mb_optim.step()

        
        #self.ppo_loss(adv, )


    def sample_action(self, actor_pred):
        """Sample an action from the policy."""
        actor_distr = Categorical(actor_pred)
        return int(actor_distr.sample())

    def agg_router_results(self, router_results_list):
        """Aggregate router results."""
        # Important
        # action_probs, cat by dim 0
        # actions, extend by dim 0
        # critic_outputs, cat by dim 0
        # action_masks, cat by dim 0
        # TODO: Add mb_count

        router_results_batch = router_results_list[0]
        action_probs = torch.zeros(
            len(router_results_list),
            router_results_list[0].action_probs.shape[1],
            router_results_list[0].action_probs.shape[2],
            device=ale_config.device)
        

        critic_outputs = torch.zeros(
            len(router_results_list),
            router_results_list[0].critic_outputs.shape[1],
            router_results_list[0].critic_outputs.shape[2],
            device=ale_config.device)
        print("action_masks.shape", router_results_list[0].action_masks.shape)
        action_masks = torch.zeros(
            len(router_results_list),
            router_results_list[0].action_masks.shape[1],
            router_results_list[0].action_masks.shape[2],
            device=ale_config.device)
        
        mb_count = torch.zeros(
            router_results_list[0].mb_count.shape[0],
            device=ale_config.device)
        
        print("router_results_list[0].actions.shape", router_results_list[0].actions.shape)
        actions = torch.zeros(
            len(router_results_list),
            router_results_list[0].actions.shape[1],
            dtype=torch.int64,
            device=ale_config.device)

        for i in range(len(router_results_list)):
            action_probs[i] = router_results_list[i].action_probs
            critic_outputs[i] = router_results_list[i].critic_outputs
            action_masks[i] = router_results_list[i].action_masks
            mb_count += router_results_list[i].mb_count.to(ale_config.device)
            
            actions[i] = router_results_list[i].actions
        print("mb_count", mb_count)
        router_results = router_results_list[0]
        router_results.action_probs = action_probs
        router_results.critic_outputs = critic_outputs
        router_results.action_masks = action_masks
        router_results.actions = actions
        return router_results

        
    def load_train_dict(self):
        if self.args.load and os.path.exists("models/train_dict.json"):
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
        with open("models/train_dict.json", "w") as f:
            json.dump(self.train_dict, f)
