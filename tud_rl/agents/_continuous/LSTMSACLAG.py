import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tud_rl.common.buffer as buffer
import tud_rl.common.nets as nets
from tud_rl import logger
from tud_rl.agents.base import BaseAgent
from tud_rl.common.configparser import ConfigFile
import torch.optim.lr_scheduler as sched
import torch
import scipy

class RunningMeanStdStable:
    def __init__(self, shape=(), epsilon=1e-4, clip=1e6):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon
        self.sumsq = torch.zeros(shape)
        self.clip = clip
        self.initialized = False

    def update(self, x):
        x = x.to(self.mean.device)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.size(0)

        if not self.initialized:
            self.mean = batch_mean
            self.var = batch_var
            self.sumsq = batch_var * batch_count
            self.count = batch_count
            self.initialized = True
            return

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # update mean
        new_mean = self.mean + delta * batch_count / total_count

        # update sumsq
        sumsq = self.sumsq + batch_var * batch_count + delta.pow(2) * self.count * batch_count / total_count

        # assign new values
        self.mean = new_mean
        self.sumsq = sumsq
        self.count = total_count
        self.var = sumsq / (total_count - 1 + 1e-8)  # unbiased var
        self.var = torch.clamp(self.var, min=1e-4)

    def normalize(self, x):
        std = torch.sqrt(self.var + 1e-8)
        normalized = (x - self.mean.to(x.device)) / std.to(x.device)
        return normalized

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
        

class LSTMSACLAGAgent(BaseAgent):
    def __init__(self, c: ConfigFile, agent_name, normalize_cost=True):
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
        self.normalize_cost   = normalize_cost
        self.cost_priority_weight = 0.6
        self.lr_decay = True

        if self.normalize_cost:
            #self.cost_normalizer = RunningMeanStd(shape=(1,))
            self.cost_normalizer = RunningMeanStd(shape=(1,))
        
        self.cost_gamma = 0.99

        self.use_per  = True

        self.lr_temp     = getattr(c.Agent, agent_name)["lr_temp"]
        self.temp_tuning = getattr(c.Agent, agent_name)["temp_tuning"]
        self.init_temp   = getattr(c.Agent, agent_name)["init_temp"]
        self.use_cost = getattr(c.Agent, agent_name)["use_cost"]

        self.needs_history    = True
        self.history_length   = getattr(c.Agent, agent_name)["history_length"]
        self.use_past_actions = getattr(c.Agent, agent_name)["use_past_actions"]

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
            if self.use_per:
                self.replay_buffer = buffer.PrioritizedReplayBuffer_LSTM_SRL(
                    state_type     = self.state_type,
                    state_shape    = self.state_shape,
                    buffer_length  = self.buffer_length,
                    batch_size     = self.batch_size,
                    device         = self.device,
                    disc_actions   = False,
                    history_length = self.history_length,
                    action_dim     = self.num_actions,
                    alpha          = 0.6,
                    beta_start     = 0.4,
                    beta_frames    = 100000,
                )
            else:
                self.replay_buffer = buffer.UniformReplayBuffer_LSTM_SRL(state_type     = self.state_type, 
                                                                        state_shape    = self.state_shape, 
                                                                        buffer_length  = self.buffer_length,
                                                                        batch_size     = self.batch_size,
                                                                        device         = self.device,
                                                                        disc_actions   = False,
                                                                        action_dim     = self.num_actions,
                                                                        history_length = self.history_length)
        # init actor and critic
        if self.state_type == "feature":
            self.actor = nets.LSTM_GaussianActor(state_shape = self.state_shape,
                                                 action_dim  = self.num_actions,
                                                 use_past_actions = self.use_past_actions).to(self.device)
            
            self.critic = nets.LSTM_Double_Critic(state_shape      = self.state_shape,
                                                  action_dim       = self.num_actions,
                                                  use_past_actions = self.use_past_actions).to(self.device)
            
            self.critic_cost = nets.LSTM_Double_Critic(state_shape      = self.state_shape,
                                                       action_dim       = self.num_actions,
                                                       use_past_actions = self.use_past_actions).to(self.device)

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
            self.critic_cost_optimizer = optim.Adam(self.critic_cost.parameters(), lr=self.lr_critic*10)
        else:
            self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=self.lr_actor, alpha=0.95, centered=True, eps=0.01)
            self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)
            self.critic_cost_optimizer = optim.RMSprop(self.critic_cost.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)
        # define learning rate schedulers
        if self.lr_decay:
            self.lr_scheduler_critic = optim.lr_scheduler.ReduceLROnPlateau(
                self.critic_optimizer,
                mode='min',
                factor= 0.5,
                patience=1000,
                threshold=0.01,
                min_lr=0.00001,
                verbose=True
            )
            
            self.lr_scheduler_cost = optim.lr_scheduler.ReduceLROnPlateau(
                self.critic_optimizer,
                mode='min',
                factor= 0.5,
                patience=1000,
                threshold=0.01,
                min_lr=0.00001,
                verbose=True
            )

        # Initialize Lagrangian multiplier and its optimizer
        self.lambda_lr = getattr(c.Agent, agent_name)["lambda_lr"]
        self.constraint_threshold = getattr(c.Agent, agent_name)["constraint_threshold"]
        #self.lagrangian_multiplier = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)
        self.lagrangian_multiplier = nn.Parameter(torch.tensor([0.1], device=self.device), requires_grad=True)


        self.lagrangian_optimizer = optim.Adam([self.lagrangian_multiplier], lr=self.lambda_lr)

        """ def phased_lr_lambda(step):
            if step < 100_000:
                return 5e-4 / self.lr_actor
            else:
                return 1.0

        def phased_lambda_lr_lambda(step):
            if step < 100_000:
                return 1e-3 / self.lambda_lr
            else:
                return 1.0

        self.actor_scheduler = sched.LambdaLR(self.actor_optimizer,
                                              lr_lambda=phased_lr_lambda)
        self.lambda_scheduler = sched.LambdaLR(self.lagrangian_optimizer,
                                               lr_lambda=phased_lambda_lr_lambda)
        
        self.global_step = 0 """

    @torch.no_grad()
    def select_action(self, s, s_hist, a_hist, hist_len):
        """Selects action via actor network for a given state. Adds exploration bonus from noise and clips to action scale.
        s:        np.array with shape (state_shape,)
        s_hist:   np.array with shape (history_length, state_shape)
        a_hist:   np.array with shape (history_length, action_dim)
        hist_len: int
        
        returns: np.array with shape (action_dim,)
        """
        # reshape arguments and convert to tensors
        s = torch.tensor(s, dtype=torch.float32).view(1, self.state_shape).to(self.device)
        s_hist = torch.tensor(s_hist, dtype=torch.float32).view(1, self.history_length, self.state_shape).to(self.device)
        a_hist = torch.tensor(a_hist, dtype=torch.float32).view(1, self.history_length, self.num_actions).to(self.device)
        hist_len = torch.tensor(hist_len).to(self.device)

        # forward pass
        if self.mode == "train":
            a, _, _ = self.actor(s, s_hist, a_hist, hist_len, deterministic=False, with_logprob=False)
        else:
            a, _, _ = self.actor(s, s_hist, a_hist, hist_len, deterministic=True, with_logprob=False)
        
        # reshape actions
        return a.cpu().numpy().reshape(self.num_actions)

    def memorize(self, s, a, r, s2, d, info):
        """Stores current transition in replay buffer."""
        self.replay_buffer.add(s, a, r, s2, d, info)

    def _compute_target(self, s2_hist, a2_hist, hist_len2, r, s2, d):
        with torch.no_grad():
            # target actions come from current policy (no target actor)
            target_a, target_logp_a, _ = self.actor(s=s2, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2, deterministic=False, with_logprob=True)

            # Q-value of next state-action pair
            Q_next1, Q_next2, _ = self.target_critic(s=s2, a=target_a, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2)
            Q_next = torch.min(Q_next1, Q_next2)

            # target Q-value
            y = r + self.gamma * (1 - d) * (Q_next - self.temperature * target_logp_a)
        return y

    def _compute_target_cost(self, s2_hist, a2_hist, hist_len2, r, s2, d, cost):
        with torch.no_grad():
            # target actions come from current policy (no target actor)
            target_a, _, _ = self.actor(s=s2, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2, deterministic=False, with_logprob=False)

            # Q-value of next state-action pair
            Q_next1, Q_next2, _ = self.target_critic_cost(s=s2, a=target_a, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2)
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
        batch = self.replay_buffer.sample_ranked()

        if self.use_per:
            (s_hist, a_hist, hist_len,
             s2_hist, a2_hist, hist_len2,
             s, a, r, s2, d,
             is_weights, indices, info) = batch
        else:
            (s_hist, a_hist, hist_len,
             s2_hist, a2_hist, hist_len2,
             s, a, r, s2, d, info) = batch
            is_weights = torch.ones((self.batch_size, 1), device=self.device)
            indices = None
        
        
        # unpack batch
        #s_hist, a_hist, hist_len, s2_hist, a2_hist, hist_len2, s, a, r, s2, d, info = batch

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
        Q1, Q2, critic_net_info = self.critic(s=s, a=a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
 
        # calculate targets
        y = self._compute_target(s2_hist, a2_hist, hist_len2, r, s2, d)

        loss1 = self._compute_loss(Q1, y,reduction="none")
        loss2 = self._compute_loss(Q2, y,reduction="none")
        critic_loss = (is_weights * loss1).mean() + (is_weights * loss2).mean()


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
            Q1_cost, Q2_cost, critic_cost_net_info = self.critic_cost(s=s, a=a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            y_cost = self._compute_target_cost(s2_hist, a2_hist, hist_len2, r, s2, d, normalized_cost)

            loss1c, loss2c = self._compute_loss(Q1_cost, y_cost, reduction="none"), self._compute_loss(Q2_cost, y_cost, reduction="none")

            critic_loss_cost = (is_weights * loss1c).mean() + (is_weights * loss2c).mean()

            critic_loss_cost.backward()
            if self.grad_rescale:
                for p in self.critic_cost.parameters():
                    p.grad *= 1 / math.sqrt(2)
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.critic_cost.parameters(), max_norm=10)
            
            self.critic_cost_optimizer.step()

        # if lr decay is used, update learning rate
        if self.lr_decay:
            # Step schedulers based on losses
            self.lr_scheduler_critic.step(critic_loss)
            if self.use_cost:
                self.lr_scheduler_cost.step(critic_loss_cost)
        

        # Log learning rates
        if self.lr_decay:
            current_lr_critic = self.critic_optimizer.param_groups[0]['lr']
            self.logger.store(Critic_LR=current_lr_critic)
            
            if self.use_cost:
                current_lr_cost = self.critic_cost_optimizer.param_groups[0]['lr']
                self.logger.store(Cost_Critic_LR=current_lr_cost)


        # log critic training
        if self.use_cost:
            self.logger.store(Cost_Critic_loss=critic_loss_cost.detach().cpu().numpy().item(), **critic_cost_net_info)
            self.logger.store(Cost_Q_val=Q1_cost.detach().mean().cpu().numpy().item())
        
        self.logger.store(Critic_loss=critic_loss.detach().cpu().numpy().item(), **critic_net_info)
        self.logger.store(Q_val=Q1.detach().mean().cpu().numpy().item())
        
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
        curr_a, curr_a_logprob, act_net_info = self.actor(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len, deterministic=False, with_logprob=True)


        # compute Q1, Q2 values for current state and actor's actions
        Q1_reward, Q2_reward, _ = self.critic(s=s, a=curr_a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)

        Q_reward = torch.min(Q1_reward, Q2_reward)



        Q1_cost, Q2_cost, _ = self.critic_cost(s=s, a=curr_a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)

        Q_cost = torch.min(Q1_cost, Q2_cost)

        # compute policy loss (which is based on min Q1, Q2 instead of just Q1 as in TD3, plus consider entropy regularization)
        actor_loss = (self.temperature * curr_a_logprob - (Q_reward - self.lagrangian_multiplier.item() * Q_cost)).mean()
        # If λ grows large during training, normalize the actor loss
        #actor_loss = actor_loss / (1 + self.lagrangian_multiplier.detach())


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

        #-------- update Lagrangian multiplier --------
        if self.use_cost:
            # Lagrangian relaxation term
            lagrangian_loss = -self.lagrangian_multiplier * (cost - self.constraint_threshold).mean()
            # Update Lagrangian multiplier
            self.lagrangian_optimizer.zero_grad()
            lagrangian_loss.backward()
            self.lagrangian_optimizer.step()
            self.lagrangian_multiplier.data.clamp_(min=0)

        if self.use_per:
            # Use per-sample losses (before weighting) for priority updates
            # loss1, loss2, loss1c, loss2c computed above with reduction='none'
            td1_vals = loss1.detach().cpu().numpy().flatten()
            td2_vals = loss2.detach().cpu().numpy().flatten()
            reward_td = (td1_vals + td2_vals) / 2

            """ if self.use_cost: 
                loss1c_vals = loss1c.detach().cpu().numpy().flatten()
                loss2c_vals = loss2c.detach().cpu().numpy().flatten()
                cost_td = (loss1c_vals + loss2c_vals) / 2
                self.cost_priority_weight = np.mean(np.abs(cost_td)) / (np.mean(np.abs(reward_td)) + 1e-6)
                combined = reward_td + self.cost_priority_weight * cost_td
                combined = combined / (1 + self.cost_priority_weight)
                print("cost pror ", self.cost_priority_weight)
                print("combined: ", combined)
                print("Reward TD: ", reward_td)
                print("Cost TD: ", cost_td)
            else:
                combined = reward_td """
            
            # Option 1 - Dynamic ratio with clamping
            """ # Calculate ratio of errors but clamp to reasonable range
            error_ratio = min(5.0, max(0.2, np.mean(np.abs(reward_td)) / (np.mean(np.abs(cost_td)) + 1e-6)))
            reward_weight = 1.0 / (1.0 + error_ratio)
            cost_weight = error_ratio / (1.0 + error_ratio)

            # Weighted combination
            combined = reward_weight * reward_td + cost_weight * cost_td """
            
            # Option 2 -  Simple fixed balance
            """ alpha = 0.5  # Equal weight to reward and cost
            combined = alpha * reward_td + (1-alpha) * cost_td """

            # Option 3 - Normalized combination

            # Normalize both error types
            """ reward_mean = np.mean(np.abs(reward_td)) + 1e-6
            cost_mean = np.mean(np.abs(cost_td)) + 1e-6

            # Normalize each term by its own mean
            reward_normalized = reward_td / reward_mean
            cost_normalized = cost_td / cost_mean

            # Equal importance to both components after normalization
            combined = 0.5 * reward_normalized + 0.5 * cost_normalized """

            if self.use_cost: 
                loss1c_vals = loss1c.detach().cpu().numpy().flatten()
                loss2c_vals = loss2c.detach().cpu().numpy().flatten()
                cost_td = (loss1c_vals + loss2c_vals) / 2


                error_ratio = min(5.0, max(0.2, np.mean(np.abs(reward_td)) / (np.mean(np.abs(cost_td)) + 1e-6)))
                reward_weight = error_ratio / (1.0 + error_ratio)
                cost_weight = 1.0 / (1.0 + error_ratio)

                # Weighted combination
                combined = reward_weight * reward_td + cost_weight * cost_td

            else:
                combined = reward_td
            
            self.replay_buffer.update_priorities(indices, combined)

        
        # unfreeze critic so it can be trained in next iteration
        for param in self.critic.parameters():
            param.requires_grad = True
        
        if self.use_cost:
            for param in self.critic_cost.parameters():
                param.requires_grad = True
        
        # log actor training
        self.logger.store(Actor_loss=actor_loss.detach().cpu().numpy().item(), **act_net_info)
        self.logger.store(Lambda=self.lagrangian_multiplier.item())

        """ self.actor_scheduler.step()
        self.lambda_scheduler.step() """

        

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