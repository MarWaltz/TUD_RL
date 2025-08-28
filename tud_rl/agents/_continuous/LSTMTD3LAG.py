import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tud_rl.common.buffer as buffer
import tud_rl.common.nets as nets
from tud_rl.agents._continuous.LSTMDDPG import LSTMDDPGAgent
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

class LSTMTD3LAGAgent(LSTMDDPGAgent):
    def __init__(self, c: ConfigFile, agent_name, normalize_cost=True):
        super().__init__(c, agent_name, init_critic=False)

        # TD3 params
        self.tgt_noise      = getattr(c.Agent, agent_name)["tgt_noise"]
        self.tgt_noise_clip = getattr(c.Agent, agent_name)["tgt_noise_clip"]
        self.pol_upd_delay  = getattr(c.Agent, agent_name)["pol_upd_delay"]

        # Cost constraint params
        self.use_cost = getattr(c.Agent, agent_name).get("use_cost", True)
        self.cost_gamma = getattr(c.Agent, agent_name).get("cost_gamma", 0.9)
        self.lambda_lr = getattr(c.Agent, agent_name).get("lambda_lr", 1e-3)
        self.constraint_threshold = getattr(c.Agent, agent_name).get("constraint_threshold", 1.0)

        self.use_per = True
        self.normalize_cost = normalize_cost
        if self.normalize_cost:
            self.cost_normalizer = RunningMeanStd(shape=(1,))

        # Critic networks
        if self.state_type == "feature":
            self.critic = nets.LSTM_Double_Critic(
                state_shape=self.state_shape,
                action_dim=self.num_actions,
                use_past_actions=self.use_past_actions
            ).to(self.device)
            self.critic_cost = nets.LSTM_Double_Critic(
                state_shape=self.state_shape,
                action_dim=self.num_actions,
                use_past_actions=self.use_past_actions
            ).to(self.device)

        self.n_params = self._count_params(self.actor), self._count_params(self.critic)

        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.target_critic_cost = copy.deepcopy(self.critic_cost).to(self.device)
        for p in self.target_critic.parameters():
            p.requires_grad = False
        for p in self.target_critic_cost.parameters():
            p.requires_grad = False

        # Optimizers
        if self.optimizer == "Adam":
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
            self.critic_cost_optimizer = optim.Adam(self.critic_cost.parameters(), lr=self.lr_critic)
        else:
            self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)
            self.critic_cost_optimizer = optim.RMSprop(self.critic_cost.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)

        # Lagrangian multiplier
        self.lagrangian_multiplier = nn.Parameter(torch.tensor([0.1], device=self.device), requires_grad=True)
        self.lagrangian_optimizer = optim.Adam([self.lagrangian_multiplier], lr=self.lambda_lr)

        # Replay buffer
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
                self.replay_buffer = buffer.UniformReplayBuffer_LSTM_SRL(
                    state_type     = self.state_type, 
                    state_shape    = self.state_shape, 
                    buffer_length  = self.buffer_length,
                    batch_size     = self.batch_size,
                    device         = self.device,
                    disc_actions   = False,
                    action_dim     = self.num_actions,
                    history_length = self.history_length)

        self.pol_upd_cnt = 0

    def _compute_target(self, s2_hist, a2_hist, hist_len2, r, s2, d):
        with torch.no_grad():
            target_a, _ = self.target_actor(s=s2, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2)
            eps = torch.randn_like(target_a) * self.tgt_noise
            eps = torch.clamp(eps, -self.tgt_noise_clip, self.tgt_noise_clip)
            target_a = torch.clamp(target_a + eps, -1, 1)
            Q_next1, Q_next2, _ = self.target_critic(s=s2, a=target_a, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2)
            Q_next = torch.min(Q_next1, Q_next2)
            y = r + self.gamma * Q_next * (1 - d)
        return y

    def _compute_target_cost(self, s2_hist, a2_hist, hist_len2, cost, s2, d):
        with torch.no_grad():
            target_a, _ = self.target_actor(s=s2, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2)
            eps = torch.randn_like(target_a) * self.tgt_noise
            eps = torch.clamp(eps, -self.tgt_noise_clip, self.tgt_noise_clip)
            target_a = torch.clamp(target_a + eps, -1, 1)
            Q_next1, Q_next2, _ = self.target_critic_cost(s=s2, a=target_a, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2)
            Q_next = torch.min(Q_next1, Q_next2)
            y = cost + self.cost_gamma * Q_next * (1 - d)
        return y

    def _compute_loss(self, Q, y, reduction="mean"):
        return nn.MSELoss(reduction=reduction)(Q, y)

    def memorize(self, s, a, r, s2, d, info):
        """Stores current transition in replay buffer."""
        self.replay_buffer.add(s, a, r, s2, d, info)


    def train(self):
        # Sample batch with PER
        batch = self.replay_buffer.sample_ranked()
        (s_hist, a_hist, hist_len,
         s2_hist, a2_hist, hist_len2,
         s, a, r, s2, d,
         is_weights, indices, info) = batch

        # Extract and normalize cost
        cost = torch.FloatTensor([i['cost'] for i in info]).unsqueeze(-1).to(self.device)
        if self.normalize_cost:
            self.cost_normalizer.update(cost)
            normalized_cost = self.cost_normalizer.normalize(cost)
        else:
            normalized_cost = cost

        # -------- train reward critic --------
        self.critic_optimizer.zero_grad()
        Q1, Q2, critic_net_info = self.critic(s=s, a=a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
        y = self._compute_target(s2_hist, a2_hist, hist_len2, r, s2, d)
        loss1 = self._compute_loss(Q1, y, reduction="none")
        loss2 = self._compute_loss(Q2, y, reduction="none")
        critic_loss = (is_weights * loss1).mean() + (is_weights * loss2).mean()
        critic_loss.backward()
        if self.grad_rescale:
            for p in self.critic.parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10)
        self.critic_optimizer.step()

        # -------- train cost critic --------
        self.critic_cost_optimizer.zero_grad()
        Q1_cost, Q2_cost, critic_cost_net_info = self.critic_cost(s=s, a=a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
        y_cost = self._compute_target_cost(s2_hist, a2_hist, hist_len2, normalized_cost, s2, d)
        loss1c = self._compute_loss(Q1_cost, y_cost, reduction="none")
        loss2c = self._compute_loss(Q2_cost, y_cost, reduction="none")
        critic_loss_cost = (is_weights * loss1c).mean() + (is_weights * loss2c).mean()
        critic_loss_cost.backward()
        if self.grad_rescale:
            for p in self.critic_cost.parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.critic_cost.parameters(), max_norm=10)
        self.critic_cost_optimizer.step()

        # -------- train actor --------
        if self.pol_upd_cnt % self.pol_upd_delay == 0:
            for param in self.critic.parameters():
                param.requires_grad = False
            for param in self.critic_cost.parameters():
                param.requires_grad = False

            self.actor_optimizer.zero_grad()
            curr_a, act_net_info = self.actor(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            Q1_reward, Q2_reward, _ = self.critic(s=s, a=curr_a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            Q_reward = torch.min(Q1_reward, Q2_reward)
            Q1_cost, Q2_cost, _ = self.critic_cost(s=s, a=curr_a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            Q_cost = torch.min(Q1_cost, Q2_cost)
            # Lagrangian actor loss
            actor_loss = -(Q_reward - self.lagrangian_multiplier.item() * Q_cost).mean()
            actor_loss.backward()
            if self.grad_rescale:
                for p in self.actor.parameters():
                    p.grad *= 1 / math.sqrt(2)
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
            self.actor_optimizer.step()
            for param in self.critic.parameters():
                param.requires_grad = True
            for param in self.critic_cost.parameters():
                param.requires_grad = True

            self.logger.store(Actor_loss=actor_loss.detach().cpu().numpy().item(), **act_net_info)
            self.polyak_update()

        # -------- update Lagrangian multiplier --------
        lagrangian_loss = -self.lagrangian_multiplier * (cost - self.constraint_threshold).mean()
        self.lagrangian_optimizer.zero_grad()
        lagrangian_loss.backward()
        self.lagrangian_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(min=0)

        # -------- update PER priorities --------
        td1_vals = loss1.detach().cpu().numpy().flatten()
        td2_vals = loss2.detach().cpu().numpy().flatten()
        reward_td = (td1_vals + td2_vals) / 2
        cost_td = (loss1c.detach().cpu().numpy().flatten() + loss2c.detach().cpu().numpy().flatten()) / 2

        error_ratio = min(5.0, max(0.2, np.mean(np.abs(reward_td)) / (np.mean(np.abs(cost_td)) + 1e-6)))
        reward_weight = error_ratio / (1.0 + error_ratio)
        cost_weight = 1.0 / (1.0 + error_ratio)
        combined = reward_weight * reward_td + cost_weight * cost_td

        self.replay_buffer.update_priorities(indices, combined)

        # -------- log critic training --------
        self.logger.store(Critic_loss=critic_loss.detach().cpu().numpy().item(), **critic_net_info)
        self.logger.store(Q_val=Q1.detach().mean().cpu().numpy().item())
        self.logger.store(Cost_Critic_loss=critic_loss_cost.detach().cpu().numpy().item(), **critic_cost_net_info)
        self.logger.store(Cost_Q_val=Q1_cost.detach().mean().cpu().numpy().item())
        self.logger.store(Lambda=self.lagrangian_multiplier.item())
        
        # Log learning rates
        current_lr_critic = self.critic_optimizer.param_groups[0]['lr']
        self.logger.store(Critic_LR=current_lr_critic)

        if self.use_cost:
            current_lr_cost = self.critic_cost_optimizer.param_groups[0]['lr']
            self.logger.store(Cost_Critic_LR=current_lr_cost)

        self.pol_upd_cnt += 1

    @torch.no_grad()
    def polyak_update(self):
        for target_p, main_p in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1 - self.tau) * target_p.data)
        for target_p, main_p in zip(self.target_critic_cost.parameters(), self.critic_cost.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1 - self.tau) * target_p.data)
