import random
import torch
import torch.multiprocessing as mp
from torch.distributions import Categorical
import numpy as np
import time
import json
import os

from dataclasses import dataclass

from trainer import Trainer
from configs import ale_config
#from configs import ale_config_vit as ale_config
from util.moving_stats import MovingStats
from workers.agent_worker import AgentWorker
from envs.ale_env import ALEEnv
from envs.cartpole_env import CartpoleEnv
from model_util import RouterResult

mp.set_start_method("spawn", force=True)
mp.set_sharing_strategy('file_system')


@dataclass
class ChildProcess:
    process: mp.Process
    state_queue: mp.Queue
    action_queue: mp.Queue

@dataclass
class ActionMessage:
    action: int
    stop: bool = False
    reset: bool = False


class ExpBuffer:
    """Stores experiences from the episodes."""
    def __init__(self, args):
        self.args = args
        self.num_env = len(ale_config.envs)
        self.max_timesteps = self.num_env * self.args.max_env_steps

    def reset(self):
        # TODO: Change all of these to tensor since you know max environment size
        self.cur_step = 0
        self.states = torch.zeros(self.max_timesteps, ale_config.state_size, device=ale_config.device)

        self.rewards = torch.zeros(self.num_env, self.args.max_env_steps, device=ale_config.device)
        self.actions = torch.zeros(self.max_timesteps, device=ale_config.device)
        self.ac_critic_preds = torch.zeros(self.num_env, self.args.max_env_steps, device=ale_config.device)
        self.ac_actor_preds = torch.zeros(self.max_timesteps, ale_config.num_actions, device=ale_config.device)
        self.hidden_states = torch.zeros(self.max_timesteps, 1, ale_config.ac_hidden_size, device=ale_config.device)
        self.dones = torch.zeros(self.num_env, self.args.max_env_steps, device=ale_config.device)
        self.router_results = None
        self.task_ids = torch.zeros(self.max_timesteps, 1, device=ale_config.device, dtype=torch.int32)


    def add_ex(self, state, reward, action, ac_actor_pred, ac_critic_pred, hidden_state, env_idx, done, task_id):
        self.states[self.cur_step] = state
        self.rewards[env_idx, self.cur_step//self.num_env] = reward
        self.actions[self.cur_step] = action
        self.ac_critic_preds[env_idx, self.cur_step//self.num_env] = ac_critic_pred
        self.ac_actor_preds[self.cur_step] = ac_actor_pred
        self.hidden_states[self.cur_step] = hidden_state
        self.dones[env_idx, self.cur_step//self.num_env] = done
        self.task_ids[self.cur_step] = task_id
        self.cur_step += 1
    
    def add_router_results(self, router_results):
        if self.router_results is None:
            self.router_results = RouterResult(
                    torch.zeros(self.max_timesteps, router_results.action_probs.shape[1], router_results.action_probs.shape[2], device=ale_config.device),
                    torch.zeros(self.max_timesteps, router_results.actions.shape[1], dtype=torch.int64, device=ale_config.device),
                    torch.zeros(self.max_timesteps, router_results.critic_outputs.shape[1], device=ale_config.device),
                    None,
                    None,
                    torch.zeros_like(router_results.mb_count, device=ale_config.device),
                    torch.zeros(self.max_timesteps, router_results.action_masks.shape[1], router_results.action_masks.shape[2], device=ale_config.device)) 
        
        for i in range(self.num_env):
            self.router_results.action_probs[self.cur_step + i] = router_results.action_probs[i]
            self.router_results.actions[self.cur_step + i] = router_results.actions[i]
            self.router_results.critic_outputs[self.cur_step + i] = router_results.critic_outputs[i]
            self.router_results.action_masks[self.cur_step + i] = router_results.action_masks[i]
            
        self.router_results.mb_count += router_results.mb_count


            
    
    def get_rewards_and_critic(self):
        return self.rewards, self.ac_critic_preds, self.dones





class ALETrainer(Trainer):
    def __init__(self, args, mb_model):
        super().__init__(args, mb_model)
        self.args = args
        self.moving_stats = MovingStats(args)
        self.exp_buffer = ExpBuffer(args)

        self._init_workers()
        if self.args.load:
            self.load_results()
        else:
            self._train_dict = {
                "ep_returns" : [],
                "mb_critic_loss": [],
                "mb_actor_loss": [],
                "router_critic_loss": [],
                "router_actor_loss": []
            }

    def _init_workers(self):
        """Create workers that will parallelize environments."""
        self.child_workers = []
        for i in range(len(ale_config.envs)):
            state_queue = mp.Queue(1)
            action_queue = mp.Queue(1)
            # if "cartpole" in ale_config.envs[i].lower(): 
            #     pass
            # else:
            env = ALEEnv(
                self.args, ale_config.envs[i], ale_config.reward_scales[i])

            agent_worker = AgentWorker(self.args, env, state_queue, action_queue)
            agent_process = mp.Process(target=agent_worker.run)
            agent_process.start()
            self.child_workers.append(
                ChildProcess(agent_process, state_queue, action_queue))

    def train_step(self):
        # Set rewards of router to returns of episode
        #dones = torch.tensor(dones, dtype=torch.float32, device=ale_config.device)
        #ac_critic_preds = torch.cat(ac_critic_preds)    
        
        #rewards = torch.tensor(rewards, dtype=torch.float32, device=ale_config.device)
        #print("rewards", rewards.max(), rewards.min(), rewards.mean())
        #rewards = self.moving_stats.standardize_rewards(rewards)
        rewards, ac_critic_preds, dones = self.exp_buffer.get_rewards_and_critic()
        print("dones", dones, dones.sum(axis=-1))
        returns, advs, tdlamret = self.compute_return(rewards, ac_critic_preds, dones)

        router_rewards = torch.zeros(self.exp_buffer.max_timesteps, self.args.max_depth, device=ale_config.device)
        returns = returns.permute(1, 0).reshape(-1)
        router_rewards[:, -1] = returns

        """Train modules."""
        self._mb_model.save_states(False)
        #self.exp_buffer.act = torch.tensor(actions, dtype=torch.int64, device=ale_config.device).unsqueeze(1)
        advs = advs.permute(1, 0).reshape(-1)
        tdlamret = tdlamret.permute(1, 0).reshape(-1)
        

        critic_step_loss, mb_actor_losses, mb_critic_losses = self.train_rl_modules(
            advs,
            tdlamret,
            returns,
            self.exp_buffer.ac_actor_preds,
            self.exp_buffer.actions.unsqueeze(1),
            self.exp_buffer.states,
            self.exp_buffer.router_results,
            self.exp_buffer.ac_critic_preds.permute(1, 0).reshape(-1),
            self.exp_buffer.dones.permute(1, 0).reshape(-1))


        """Train router."""

        #print("router_rewards[:, -1]", router_rewards[:, -1])            
        # Clip the critic loss
        critic_step_loss = -torch.clip(critic_step_loss, 0, 1.0)
        print("critic_step_loss[:25]", critic_step_loss[:25])
        print("router_rewards[:25, -1]", router_rewards[:25, -1])
        print("router_rewards[:, -1]", router_rewards[:, -1])
        # Add critic loss to router rewards to encourage decreasing this value as well
        router_rewards[:, -1] += critic_step_loss.detach()
        print("router_rewards.shape", router_rewards.shape)
        print("self.exp_buffer.router_results.critic_outputs.shape", self.exp_buffer.router_results.critic_outputs.shape)
        router_actor_losses, router_critic_losses = self.train_router(
            self.exp_buffer.states.unsqueeze(1),
            self.exp_buffer.task_ids,
            router_rewards,
            self.exp_buffer.router_results,
            self.exp_buffer.hidden_states)
        

        self._train_dict["mb_actor_loss"].append(sum(mb_actor_losses)/len(mb_actor_losses))
        self._train_dict["mb_critic_loss"].append(sum(mb_critic_losses)/len(mb_critic_losses))
        self._train_dict["router_actor_loss"].append(sum(router_actor_losses)/len(router_actor_losses))
        self._train_dict["router_critic_loss"].append(sum(router_critic_losses)/len(router_critic_losses))


        # Decay the learning rate
        self.decay_lr()
        
        # # Save models
        self.save()
        self.moving_stats.save()
        # print("Total Episode time", time.time() - start_e_i, "\n")

        self._train_dict["ep_returns"].append(float(returns.mean()))
        self.save_results()

    def setup_next_batch(self):
        self.exp_buffer.reset()
        # Run through an episode
        self._mb_model.save_states(True)
        self._mb_model.reset()
        

    def train(self):
        self.debug = True
        #cur_hidden_states = torch.zeros(len(self.child_workers), self.args.hidden_size, device=ale_config.device)
        prev_hidden_states = torch.zeros(len(self.child_workers), self.args.hidden_size, device=ale_config.device)
                            # Reset buffer for next batch of epsidoes
        self.setup_next_batch()
        self._mb_model.setup_next_episode()

        prev_state_msgs = []
        prev_states = []

        for e_i in range(self.args.num_episodes):


            num_steps = 0

            while num_steps < self.args.max_env_steps + 1:
                
                # Get states in envs
                cur_states = []
                cur_state_msgs = []
                for env_idx in range(len(self.child_workers)):
                    
                    state_msg = self.child_workers[env_idx].state_queue.get()
                    cur_state = state_msg.state.unsqueeze(0).to(ale_config.device)
                    
                    if len(prev_state_msgs) > 0:
                        # Add timestep experience
                        self.exp_buffer.add_ex(
                            prev_states[env_idx],
                            state_msg.reward,
                            actions[env_idx],
                            actor_pred[env_idx],
                            critic_pred[env_idx],
                            prev_hidden_states[env_idx],
                            env_idx,
                            prev_state_msgs[env_idx].done,
                            task_ids[env_idx])    
                        if env_idx == len(self.child_workers) - 1:
                            prev_hidden_states = cur_hidden_states
                    cur_states.append(cur_state)                
                    cur_state_msgs.append(state_msg)

                
                # Break after adding last experiences
                if  num_steps >= self.args.max_env_steps:
                    num_steps = 0
                    # Train the models
                    self.train_step()
                    self.setup_next_batch()
                    
                    
                
                cur_states = torch.cat(cur_states)
                task_ids = ale_config.envs_ids.clone()

                with torch.no_grad():
                    # Get current actor-critic policy
                    router_results = self._router(cur_states.unsqueeze(1), task_ids, prev_hidden_states.unsqueeze(1))
                    # TODO: Handle receiving all hidden states
                    # Use router module selections to get actor and critic predictions
                    actor_pred, critic_pred, cur_hidden_states = self._mb_model(
                        cur_states, router_results.actions, return_hidden_state=True)


                # Sample an action
                actions = self.sample_action(actor_pred)
                prev_states = cur_states
                prev_state_msgs = cur_state_msgs
                # Pass actions to environments
                for env_idx in range(len(self.child_workers)):
                    self.child_workers[env_idx].action_queue.put(
                        ActionMessage(int(actions[env_idx])))
                

                self.exp_buffer.add_router_results(router_results)
                num_steps += 1
                if num_steps % 100 == 0:
                    print("Current step", num_steps)
                    # Increment number of episode steps
                    #num_steps += 1
                    
                    # if num_steps >= self.args.max_env_steps:
                    #     done = True

                 
            



            

    def load_results(self):
        with open(os.path.join(self.args.save_dir, "train_dict.json"), "r") as f:
            self._train_dict = json.load(f)

    def save_results(self):
        with open(os.path.join(self.args.save_dir, "train_dict.json"), "w") as f:
            json.dump(self._train_dict, f)


    def create_seq_batches(self, batch_size: int, dones: list, router_results):
        """Create minibatches that will be sequential for seq_len steps."""
        #cur_seq_len = random.randint(1, self.args.seq_len)
        
        cur_seq_len = self.args.seq_len#min(self.args.seq_len, min_ep_len)
        #print("cur_seq_len", cur_seq_len)
        
        # Create batch_ids and split into groups of length seq_len
        num_left = batch_size % cur_seq_len
        print("num_left", num_left)
        num_groups = batch_size//cur_seq_len
        seq_batch_ids = torch.arange(num_groups * cur_seq_len).view(-1, cur_seq_len)

        # Shuffle group order
        seq_batch_ids = seq_batch_ids[torch.randperm(seq_batch_ids.shape[0])]
        
        # Create minibatches from groups
        # print("batch_size", batch_size)
        # print("seq_batch_ids", seq_batch_ids.shape)
         
        #seq_batch_ids = seq_batch_ids.reshape(self.args.nminibatches, -1)
        seq_batch_ids_list = [[] for i in range(self.args.nminibatches)]
        m_j = 0
        for i in range(seq_batch_ids.shape[0]):
            seq_batch_ids_list[m_j] += seq_batch_ids[i].tolist()
            m_j = (m_j + 1) % self.args.nminibatches


        
        # Create dictionary mapping timestep to batch_id
        step_to_batch_idx_batch = []

        # # Set of batch_idx for when a new episode starts, meaning states need to be updated 
        # ep_start_batch_idx = set((dones.nonzero().reshape(-1) + 1).tolist())

        for i in range(len(seq_batch_ids_list)):
            step_to_batch_idx = {}

            # Used to keep up with if the initial LSTM state has been set
            init_lstm_state_set = set()
            # Iterate over indices in the minibatch
            for t in range(len(seq_batch_ids_list[i])):
                
                batch_idx = seq_batch_ids_list[i][t]

                lstm_module_idx = int(
                    router_results.actions[batch_idx][ale_config.lstm_timestep])

                # If beginning of sequence or end of an episode
                if t % cur_seq_len == 0:
                    # Reset the states that have been updates since new sequence starting
                    init_lstm_state_set = set()
                    
                    # Add to indicate this LSTM block states will be set in this sequence already
                    init_lstm_state_set.add(lstm_module_idx)

                    # Set this value to indicate hidden states need to be updated
                    step_to_batch_idx[t] = batch_idx
                    
                elif lstm_module_idx not in init_lstm_state_set:
                    # print("init_lstm_state_set", init_lstm_state_set)
                    init_lstm_state_set.add(lstm_module_idx)
                    step_to_batch_idx[t] = batch_idx

            
            step_to_batch_idx_batch.append(step_to_batch_idx)

        
        # TODO: ADD last few skipped timesteps to last batch

        return seq_batch_ids_list, step_to_batch_idx_batch
        

    def train_rl_modules(self, advs, tdlamret, returns, old_actor_preds, actions, states, router_results, old_critic_preds, dones):
        critic_losses = []
        actor_losses = []
        batch_size = len(actions)
        critic_step_loss = torch.zeros(actions.shape[0], device=ale_config.device)
        for e_i in range(self.args.noptepochs):
            #batch_ids = torch.arange(batch_size)
            # Create batches of timesteps
            seq_batch_ids_list, step_to_batch_idx_batch = self.create_seq_batches(batch_size, dones, router_results)
            # print("dones", dones)
            # print("seq_batch_ids_list", seq_batch_ids_list)
            # print("step_to_batch_idx_batch", step_to_batch_idx_batch)
            for m_i in range(self.args.nminibatches):
                print(f"\nEpoch {e_i} Batch {m_i}\n------------------")
                
                loss = torch.zeros(1).cuda()
                # start = m_i * (batch_size // self.args.nminibatches)
                # end = (m_i + 1) * (batch_size // self.args.nminibatches)
                m_batch_idxs = seq_batch_ids_list[m_i]
                
                #print("m_batch_idxs", m_batch_idxs)
                #print("router_results.actions[m_batch_idxs]", router_results.actions[m_batch_idxs])
                #print("actions[m_batch_idxs]", actions[m_batch_idxs])
    

                actor_pred, critic_pred, hidden_states = self._mb_model(
                    states[m_batch_idxs],
                    router_results.actions[m_batch_idxs],
                    step_to_batch_idx_batch[m_i],
                    batch_idxs=m_batch_idxs,
                    return_hidden_state=True)
                
                # Orthoganility contraint on only input module
                # Get output of modules
                # forb_loss = None
                # for i in range(self.args.num_dup_layers-1):
                #     for j in range(i+1, self.args.num_dup_layers):
                #         idx_1 = i * ale_config.num_modules
                #         idx_2 = j * ale_config.num_modules
                #         mb_out_1 = self._mb_model.get_mb(idx_1)(states[m_batch_idxs])
                #         mb_out_2 = self._mb_model.get_mb(idx_2)(states[m_batch_idxs])
                #         cur_forb_loss = torch.linalg.matrix_norm(torch.mm(mb_out_1.T, mb_out_2))
                #         #print("cur_forb_loss", cur_forb_loss)
                #         if forb_loss is None:
                #             forb_loss = cur_forb_loss
                #         else:
                #             forb_loss += cur_forb_loss
                
                # for d in range(1, len(ale_config.fc_hidden_config)):
                #     for i in range(self.args.num_dup_layers-1):
                #         for j in range(i+1, self.args.num_dup_layers):
                #             rand_inp = torch.randn(len(m_batch_idxs), ale_config.fc_hidden_config[d][0], device=ale_config.device)
                #             idx_1 = d + i * ale_config.num_modules
                #             idx_2 = d + j * ale_config.num_modules
                #             mb_out_1 = self._mb_model.get_mb(idx_1)(rand_inp)
                #             mb_out_2 = self._mb_model.get_mb(idx_2)(rand_inp)
                #             cur_forb_loss = torch.linalg.matrix_norm(torch.mm(mb_out_1.T, mb_out_2))
                #             # print("cur_forb_loss", cur_forb_loss)
                #             forb_loss += cur_forb_loss
                
                
                # print("TOTAL forb_loss", forb_loss, forb_loss * ale_config.forb_lam)

                
                # if m_i == 0 and e_i == 0:
                #     print(self.exp_buffer.hidden_states[m_batch_idxs][0].squeeze(1) == hidden_states.detach()[0])
                #     print("self.exp_buffer.hidden_states[m_batch_idxs][0]", self.exp_buffer.hidden_states[m_batch_idxs][0].squeeze(1))
                #     print("self.exp_buffer.hidden_states[m_batch_idxs][0].squeeze(1)", self.exp_buffer.hidden_states[m_batch_idxs][0].squeeze(1))
                #     self.exp_buffer.hidden_states[m_batch_idxs] = hidden_states.detach().unsqueeze(1)


                actor_loss, critic_loss, critic_step_loss_batch, should_stop = self.ppo_loss(
                    advs[m_batch_idxs],
                    old_actor_preds[m_batch_idxs],
                    actor_pred,
                    returns[m_batch_idxs],
                    critic_pred.view(-1),
                    actions[m_batch_idxs],
                    old_critic_preds[m_batch_idxs].view(-1),
                    True)
                

                critic_losses.append(self.args.critic_lam * critic_loss.detach().cpu().numpy())
                    
                actor_losses.append(actor_loss.detach().cpu().numpy())
                
                if should_stop:
                    # #loss = actor_loss + self.args.critic_lam * critic_loss + ale_config.forb_lam * forb_loss
                    # loss = actor_loss + self.args.critic_lam * critic_loss
                    # loss.backward()
                    # self.mb_optim.zero_grad()    
                    return critic_step_loss, actor_losses, critic_losses
                

                critic_step_loss[m_batch_idxs] = critic_step_loss_batch

                print("MODULES actor_loss", actor_loss)
                print("MODULES critic_loss", critic_loss)
                #print("tdlamret.view(-1)", tdlamret[m_batch_idxs].view(-1)[:10])
                print("returns.view(-1)", returns[m_batch_idxs].view(-1)[:10])
                print("critic_pred.view(-1)", critic_pred.view(-1)[:10])
                print("actor_pred", actor_pred[0:2])
                print("len(m_batch_idxs)", len(m_batch_idxs))
                

                self.mb_optim.zero_grad()
                #loss = actor_loss + self.args.critic_lam * critic_loss + ale_config.forb_lam * forb_loss
                loss = actor_loss + self.args.critic_lam * critic_loss
                loss.backward()
                #print("LSTM GRAD weight_ih", self._mb_model.get_mb(8).layer.weight_ih.grad.max(), self._mb_model.get_mb(8).layer.weight_ih.grad.min(), self._mb_model.get_mb(8).layer.weight_ih.grad.mean(), self._mb_model.get_mb(8).layer.weight_ih.grad.std())
                #print("LSTM GRAD weight_hh", self._mb_model.get_mb(8).layer.weight_hh.grad.max(), self._mb_model.get_mb(8).layer.weight_hh.grad.min(), self._mb_model.get_mb(8).layer.weight_hh.grad.mean(), self._mb_model.get_mb(8).layer.weight_hh.grad.std())
                # print("CRITIC LAST GRAD", self._mb_model.get_mb(12).layer.weight.grad.max(), self._mb_model.get_mb(12).layer.weight.grad.min(), self._mb_model.get_mb(12).layer.weight.grad.mean(), self._mb_model.get_mb(12).layer.weight.grad.std())
                torch.nn.utils.clip_grad_norm_(self._mb_model.parameters(), self.args.max_grad_norm)
                print("LSTM GRAD weight_ih", self._mb_model.get_mb(ale_config.lstm_timestep).layer.weight_ih.grad.max(), self._mb_model.get_mb(ale_config.lstm_timestep).layer.weight_ih.grad.min(), self._mb_model.get_mb(ale_config.lstm_timestep).layer.weight_ih.grad.mean(), self._mb_model.get_mb(ale_config.lstm_timestep).layer.weight_ih.grad.std())
                print("LSTM GRAD weight_hh", self._mb_model.get_mb(ale_config.lstm_timestep).layer.weight_hh.grad.max(), self._mb_model.get_mb(ale_config.lstm_timestep).layer.weight_hh.grad.min(), self._mb_model.get_mb(ale_config.lstm_timestep).layer.weight_hh.grad.mean(), self._mb_model.get_mb(ale_config.lstm_timestep).layer.weight_hh.grad.std())

                self.mb_optim.step()

                print("\n")

        return critic_step_loss, actor_losses, critic_losses
        
        #self.ppo_loss(adv, )


    def sample_action(self, actor_pred):
        """Sample an action from the policy."""
        actor_distr = Categorical(actor_pred)
        return actor_distr.sample()

    def agg_router_results(self, router_results_list):
        """Aggregate router results."""
        # Important
        # action_probs, cat by dim 0
        # actions, extend by dim 0
        # critic_outputs, cat by dim 0
        # action_masks, cat by dim 0

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

        action_masks = torch.zeros(
            len(router_results_list),
            router_results_list[0].action_masks.shape[1],
            router_results_list[0].action_masks.shape[2],
            device=ale_config.device)
        
        mb_count = torch.zeros(
            router_results_list[0].mb_count.shape[0],
            device=ale_config.device)
        
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
        router_results.mb_count = mb_count
        return router_results

        
    def save_episode(self):
        with open("models/train_dict.json", "w") as f:
            json.dump(self.train_dict, f)
