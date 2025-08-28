import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tud_rl.common.buffer as buffer
import tud_rl.common.nets as nets
from tud_rl import logger
from tud_rl.agents.base import BaseAgent
from tud_rl.common.configparser import ConfigFile

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean.detach()
        self.var = new_var.detach()
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean.to(x.device)) / (torch.sqrt(self.var.to(x.device)) + 1e-8)




class SACLAGAgent(BaseAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

        # attributes and hyperparameters
        self.lr_actor         = c.lr_actor
        self.lr_critic        = c.lr_critic
        self.tau              = c.tau
        self.actor_weights    = c.actor_weights
        self.critic_weights   = c.critic_weights
        self.critic_cost_weights = c.critic_cost_weights
        self.net_struc_actor  = c.net_struc_actor
        self.net_struc_critic = c.net_struc_critic
        

        self.lr_temp     = getattr(c.Agent, agent_name)["lr_temp"]
        self.temp_tuning = getattr(c.Agent, agent_name)["temp_tuning"]
        self.init_temp   = getattr(c.Agent, agent_name)["init_temp"]
        self.use_cost = getattr(c.Agent, agent_name)["use_cost"]


        # checks
        assert not (self.mode == "test" and (self.actor_weights is None or self.critic_weights is None)), "Need prior weights in test mode."

        if self.state_type == "image":
            raise NotImplementedError("Currently, image input is not supported for continuous action spaces.")

        if self.net_struc_actor is not None or self.net_struc_critic is not None:
            logger.warning("The net structure cannot be controlled via the config-spec for LSTM-based agents.")

        # dynamic or static temperature
        if self.temp_tuning:

            # define target entropy
            self.target_entropy = -self.num_actions

            # optimize log(temperature) instead of temperature
            self.log_temperature = torch.zeros(1, requires_grad=True, device=self.device)

            # define temperature optimizer
            self.temp_optimizer = optim.Adam([self.log_temperature], lr=self.lr_temp)

        else:
            self.temperature = self.init_temp

        # replay buffer
        if self.mode == "train":
            self.replay_buffer = buffer.UniformReplayBuffer_SRL(state_type     = self.state_type, 
                                                                     state_shape    = self.state_shape, 
                                                                     buffer_length  = self.buffer_length,
                                                                     batch_size     = self.batch_size,
                                                                     device         = self.device,
                                                                     disc_actions   = False,
                                                                     action_dim     = self.num_actions)
        # init actor and critic
        if self.state_type == "feature":
            self.actor = nets.GaussianActor(state_shape = self.state_shape,
                                                 action_dim  = self.num_actions).to(self.device)
            
            self.critic = nets.Double_MLP(in_size   = self.state_shape + self.num_actions,
                                          out_size  = 1,
                                          net_struc = self.net_struc_critic).to(self.device)
            
            self.critic_cost = nets.Double_MLP(in_size   = self.state_shape + self.num_actions,
                                          out_size  = 1,
                                          net_struc = self.net_struc_critic).to(self.device)

        # number of parameters for actor and critic
        self.n_params = self._count_params(self.actor), self._count_params(self.critic)

        # load prior weights if available
        if self.actor_weights is not None and self.critic_weights is not None:
            self.actor.load_state_dict(torch.load(self.actor_weights, map_location=self.device))            
            self.critic.load_state_dict(torch.load(self.critic_weights, map_location=self.device))
            self.critic_cost.load_state_dict(torch.load(self.critic_cost_weights, map_location=self.device))

        # init target net
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.target_critic_cost = copy.deepcopy(self.critic_cost).to(self.device)
        
        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_critic.parameters():
            p.requires_grad = False
        
        for p in self.target_critic_cost.parameters():
            p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
            self.critic_cost_optimizer = optim.Adam(self.critic_cost.parameters(), lr=self.lr_critic)
        else:
            self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=self.lr_actor, alpha=0.95, centered=True, eps=0.01)
            self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)
            self.critic_cost_optimizer = optim.RMSprop(self.critic_cost.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)

        # Initialize Lagrangian multiplier and its optimizer
        self.lambda_lr = getattr(c.Agent, agent_name)["lambda_lr"]
        self.constraint_threshold = getattr(c.Agent, agent_name)["constraint_threshold"]
        self.lagrangian_multiplier = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)
        self.lagrangian_optimizer = optim.Adam([self.lagrangian_multiplier], lr=self.lambda_lr)
        self.normalize_cost = True
        self.cost_gamma = 0.9
        if self.normalize_cost:
            self.cost_normalizer = RunningMeanStd(shape=(1,))


    @torch.no_grad()
    def select_action(self, s):
        """Selects action via actor network for a given state. Adds exploration bonus from noise and clips to action scale.
        s:        np.array with shape (state_shape,)
        s_hist:   np.array with shape (history_length, state_shape)
        a_hist:   np.array with shape (history_length, action_dim)
        hist_len: int
        
        returns: np.array with shape (action_dim,)
        """
        # reshape arguments and convert to tensors
        s = torch.tensor(s, dtype=torch.float32).view(1, self.state_shape).to(self.device)

        # forward pass
        if self.mode == "train":
            a, _ = self.actor(s, deterministic=False, with_logprob=False)
        else:
            a, _ = self.actor(s, deterministic=True, with_logprob=False)
        
        # reshape actions
        return a.cpu().numpy().reshape(self.num_actions)

    def memorize(self, s, a, r, s2, d, info):
        """Stores current transition in replay buffer."""
        self.replay_buffer.add(s, a, r, s2, d, info)

    def _compute_target(self, r, s2, d):
        with torch.no_grad():
            # target actions come from current policy (no target actor)
            target_a, target_logp_a = self.actor(s=s2, deterministic=False, with_logprob=True)

            # Q-value of next state-action pair
            Q_next1, Q_next2 = self.target_critic(torch.cat([s2, target_a], dim=1))
            Q_next = torch.min(Q_next1, Q_next2)

            # target Q-value
            y = r + self.gamma * (1 - d) * (Q_next - self.temperature * target_logp_a)
        return y

    def _compute_target_cost(self, r, s2, d, cost):
        with torch.no_grad():
            # target actions come from current policy (no target actor)
            target_a, _ = self.actor(s=s2, deterministic=False, with_logprob=False)

            # Q-value of next state-action pair   
            Q_next1, Q_next2 = self.target_critic_cost(torch.cat([s2, target_a], dim=1))
            Q_next = torch.min(Q_next1, Q_next2)

            # target Q-value
            y = cost + self.cost_gamma * (1 - d) * Q_next
        return y

    def _compute_loss(self, Q, y, reduction="mean"):
        if self.loss == "MSELoss":
            return F.mse_loss(Q, y, reduction=reduction)

        elif self.loss == "SmoothL1Loss":
            return F.smooth_l1_loss(Q, y, reduction=reduction)

    def train(self):
        """Samples from replay_buffer, updates actor, critic and their target networks."""        
        # sample batch
        batch = self.replay_buffer.sample()
        
        # unpack batch
        s, a, r, s2, d, info = batch
        sa = torch.cat([s, a], dim=1)

        # get current temperature
        if self.temp_tuning:
            self.temperature = torch.exp(self.log_temperature).detach()

        # Extract cost from info
        cost = torch.FloatTensor([i['cost'] for i in info]).unsqueeze(-1).to(self.device)

        if self.normalize_cost:
            self.cost_normalizer.update(cost)
            normalized_cost = self.cost_normalizer.normalize(cost)
        else:
            normalized_cost = cost
            
        #-------- train critic --------
        # clear gradients
        self.critic_optimizer.zero_grad()
        
        # calculate current estimated Q-values
        Q1, Q2 = self.critic(sa)
 
        # calculate targets
        y = self._compute_target(r, s2, d)

        # calculate loss
        critic_loss = self._compute_loss(Q1, y) + self._compute_loss(Q2, y) 

        # compute gradients
        critic_loss.backward()
        
        # gradient scaling and clipping
        if self.grad_rescale:
            for p in self.critic.parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10)
        
        
        
        # perform optimizing step
        self.critic_optimizer.step()


        if self.use_cost:
            self.critic_cost_optimizer.zero_grad()
            Q1_cost, Q2_cost = self.critic_cost(sa)
            y_cost = self._compute_target_cost(r, s2, d, normalized_cost)
            critic_loss_cost = self._compute_loss(Q1_cost, y_cost) + self._compute_loss(Q2_cost, y_cost)
            critic_loss_cost.backward()
            if self.grad_rescale:
                for p in self.critic_cost.parameters():
                    p.grad *= 1 / math.sqrt(2)
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.critic_cost.parameters(), max_norm=10)
            
            self.critic_cost_optimizer.step()

        # -------- log critic training --------
        self.logger.store(Critic_loss=critic_loss.detach().cpu().numpy().item())
        self.logger.store(Q_val=Q1.detach().mean().cpu().numpy().item())
        self.logger.store(Cost_Critic_loss=critic_loss_cost.detach().cpu().numpy().item())
        self.logger.store(Cost_Q_val=Q1_cost.detach().mean().cpu().numpy().item())
        
        #-------- train actor --------
        # freeze critic so no gradient computations are wasted while training actor
        for param in self.critic.parameters():
            param.requires_grad = False
        
        if self.use_cost:
            for param in self.critic_cost.parameters():
                param.requires_grad = False


        # clear gradients
        self.actor_optimizer.zero_grad()

        # get current actions via actor
        curr_a, curr_a_logprob = self.actor(s=s, deterministic=False, with_logprob=True)

        # Correct actor loss computation
        Q1_reward, Q2_reward = self.critic(torch.cat([s, curr_a], dim=1))
        Q_reward = torch.min(Q1_reward, Q2_reward)

        if self.use_cost:
            Q1_cost, Q2_cost = self.critic_cost(torch.cat([s, curr_a], dim=1))
            Q_cost = torch.min(Q1_cost, Q2_cost)
            actor_loss = (self.temperature * curr_a_logprob - (Q_reward - self.lagrangian_multiplier * Q_cost)).mean()
        else:
            actor_loss = (self.temperature * curr_a_logprob - Q_reward).mean()

        # compute gradients
        actor_loss.backward()

        # gradient scaling and clipping
        if self.grad_rescale:
            for p in self.actor.parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
        
        # perform step with optimizer
        self.actor_optimizer.step()


        # Add clamping to prevent negative values
        if self.use_cost:
            lagrangian_loss = -self.lagrangian_multiplier * (cost - self.constraint_threshold).mean()
            self.lagrangian_optimizer.zero_grad()
            lagrangian_loss.backward()
            self.lagrangian_optimizer.step()
            self.lagrangian_multiplier.data.clamp_(min=0)
        
        
        

        # unfreeze critic so it can be trained in next iteration
        for param in self.critic.parameters():
            param.requires_grad = True
        
        if self.use_cost:
            for param in self.critic_cost.parameters():
                param.requires_grad = True
        
        # log actor training
        self.logger.store(Actor_loss=actor_loss.detach().cpu().numpy().item())
        self.logger.store(Lambda=self.lagrangian_multiplier.item())
        

        # Log learning rates
        current_lr_critic = self.critic_optimizer.param_groups[0]['lr']
        self.logger.store(Critic_LR=current_lr_critic)

        if self.use_cost:
            current_lr_cost = self.critic_cost_optimizer.param_groups[0]['lr']
            self.logger.store(Cost_Critic_LR=current_lr_cost)


        #------- update temperature --------
        if self.temp_tuning:

            # clear gradients
            self.temp_optimizer.zero_grad()

            # calculate loss
            temperature_loss = -self.log_temperature * (curr_a_logprob + self.target_entropy).detach().mean()

            # compute gradients
            temperature_loss.backward()

            # perform optimizer step
            self.temp_optimizer.step()

        #------- Update target networks -------
        self.polyak_update()

    @torch.no_grad()
    def polyak_update(self):
        """Soft update of target network weights."""

        for target_p, main_p in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1-self.tau) * target_p.data)
        
        if self.use_cost:
            for target_p, main_p in zip(self.target_critic_cost.parameters(), self.critic_cost.parameters()):
                target_p.data.copy_(self.tau * main_p.data + (1-self.tau) * target_p.data)