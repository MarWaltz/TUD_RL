import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Actor(nn.Module):
    """Defines recurrent deterministic actor."""
    
    def __init__(self, action_dim, obs_dim, use_past_actions) -> None:
        super(LSTM_Actor, self).__init__()

        self.use_past_actions = use_past_actions

        # current feature extraction
        self.curr_fe_dense1 = nn.Linear(obs_dim, 64)
        self.curr_fe_dense2 = nn.Linear(64, 64)
        
        # memory
        if use_past_actions:
            self.mem_dense = nn.Linear(obs_dim + action_dim, 64)
        else:
            self.mem_dense = nn.Linear(obs_dim, 64)
        self.mem_LSTM = nn.LSTM(input_size = 64, hidden_size = 64, num_layers = 1, batch_first = True)
        
        # post combination
        #self.post_comb_dense1 = nn.Linear(64 + 64, 64)
        self.post_comb_dense2 = nn.Linear(64, action_dim)


    def forward(self, o, o_hist, a_hist, hist_len) -> tuple:
        """o, o_hist, hist_len are torch tensors. Shapes:
        o:        torch.Size([batch_size, obs_dim])
        o_hist:   torch.Size([batch_size, history_length, obs_dim])
        a_hist:   torch.Size([batch_size, history_length, action_dim])
        hist_len: torch.Size(batch_size)
        
        returns: output with shape torch.Size([batch_size, action_dim]), act_net_info (dict)
        
        Note: 
        The one-layer LSTM is defined with batch_first=True, hence it expects input in form of:
        x = (batch_size, seq_length, obs_dim)
        
        The call <out, (hidden, cell) = LSTM(x)> results in: 
        out:    Output (= hidden state) of LSTM for each time step with shape (batch_size, seq_length, hidden_size).
        hidden: The hidden state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        cell:   The cell state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        """

        #------ current feature extraction ------
        curr_fe = F.relu(self.curr_fe_dense1(o))
        curr_fe = F.relu(self.curr_fe_dense2(curr_fe))

        #------ memory ------
        # dense layer
        if self.use_past_actions:
            x_mem = F.relu(self.mem_dense(torch.cat([o_hist, a_hist], dim=2)))
        else:
            x_mem = F.relu(self.mem_dense(o_hist))
        
        # LSTM
        #self.mem_LSTM.flatten_parameters()
        extracted_mem, (_, _) = self.mem_LSTM(x_mem)

        # get selection index according to history lengths (no-history cases will be masked later)
        h_idx = copy.deepcopy(hist_len)
        h_idx[h_idx == 0] = 1
        h_idx -= 1
        
        # select LSTM output, resulting shape is (batch_size, hidden_dim)
        hidden_mem = extracted_mem[torch.arange(extracted_mem.size(0)), h_idx]
        
        # mask no-history cases to yield zero extracted memory
        hidden_mem[hist_len == 0] = 0

        #------ post combination ------
        # concate current feature extraction with generated memory
        x = torch.cat([curr_fe, hidden_mem], dim=1)

        # final dense layers
        #x = F.relu(self.post_comb_dense1(x))
        x = torch.tanh(self.post_comb_dense2(x))
        
        # create dict for logging
        act_net_info = dict(Actor_CurFE = curr_fe.detach().mean().cpu().numpy(),
                            Actor_ExtMemory = hidden_mem.detach().mean().cpu().numpy())
        
        # return output
        return x, act_net_info


class LSTM_Critic(nn.Module):
    """Defines recurrent critic network to compute Q-values."""
    
    def __init__(self, action_dim, obs_dim, use_past_actions) -> None:
        super(LSTM_Critic, self).__init__()
        
        self.use_past_actions = use_past_actions

        # current feature extraction
        self.curr_fe_dense1 = nn.Linear(obs_dim + action_dim, 64)
        self.curr_fe_dense2 = nn.Linear(64, 64)
        
        # memory
        if use_past_actions:
            self.mem_dense = nn.Linear(obs_dim + action_dim, 64)
        else:
            self.mem_dense = nn.Linear(obs_dim, 64)
        self.mem_LSTM = nn.LSTM(input_size = 64, hidden_size = 64, num_layers = 1, batch_first = True)
        
        # post combination
        #self.post_comb_dense1 = nn.Linear(64 + 64, 64)
        self.post_comb_dense2 = nn.Linear(64, 1)
        

    def forward(self, o, a, o_hist, a_hist, hist_len) -> tuple:
        """o, o_hist, a_hist are torch tensors. Shapes:
        o:        torch.Size([batch_size, obs_dim])
        a:        torch.Size([batch_size, action_dim])
        o_hist:   torch.Size([batch_size, history_length, obs_dim])
        a_hist:   torch.Size([batch_size, history_length, action_dim])
        hist_len: torch.Size(batch_size)
        
        returns: output with shape torch.Size([batch_size, 1]), critic_net_info (dict)
        
        Note: 
        The one-layer LSTM is defined with batch_first=True, hence it expects input in form of:
        x = (batch_size, seq_length, obs_dim)
        
        The call <out, (hidden, cell) = LSTM(x)> results in: 
        out:    Output (= hidden state) of LSTM for each time step with shape (batch_size, seq_length, hidden_size).
        hidden: The hidden state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        cell:   The cell state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        """

        #------ current feature extraction ------
        # concatenate obs and act
        oa = torch.cat([o, a], dim=1)
        curr_fe = F.relu(self.curr_fe_dense1(oa))
        curr_fe = F.relu(self.curr_fe_dense2(curr_fe))
        
        #------ memory ------
        # dense layer
        if self.use_past_actions:
            x_mem = F.relu(self.mem_dense(torch.cat([o_hist, a_hist], dim=2)))
        else:
            x_mem = F.relu(self.mem_dense(o_hist))
        
        # LSTM
        #self.mem_LSTM.flatten_parameters()
        extracted_mem, (_, _) = self.mem_LSTM(x_mem)

        # get selection index according to history lengths (no-history cases will be masked later)
        h_idx = copy.deepcopy(hist_len)
        h_idx[h_idx == 0] = 1
        h_idx -= 1

        # select LSTM output, resulting shape is (batch_size, hidden_dim)
        hidden_mem = extracted_mem[torch.arange(extracted_mem.size(0)), h_idx]

        # mask no-history cases to yield zero extracted memory
        hidden_mem[hist_len == 0] = 0
        
        #------ post combination ------
        # concatenate current feature extraction with generated memory
        x = torch.cat([curr_fe, hidden_mem], dim=1)
        
        # final dense layers
        #x = F.relu(self.post_comb_dense1(x))
        x = self.post_comb_dense2(x)

        # create dict for logging
        critic_net_info = dict(Critic_CurFE = curr_fe.detach().mean().cpu().numpy(),
                               Critic_ExtMemory = hidden_mem.detach().mean().cpu().numpy())

        # return output
        return x, critic_net_info

class LSTM_Double_Critic(nn.Module):
    """Defines two recurrent critic networks to compute Q-values."""
    
    def __init__(self, action_dim, obs_dim, use_past_actions) -> None:
        super(LSTM_Double_Critic, self).__init__()
        
        self.use_past_actions = use_past_actions

        # ----------------------- Q1 ---------------------------
        # current feature extraction
        self.curr_fe_dense1_q1 = nn.Linear(obs_dim + action_dim, 64)
        self.curr_fe_dense2_q1 = nn.Linear(64, 64)
        
        # memory
        if use_past_actions:
            self.mem_dense_q1 = nn.Linear(obs_dim + action_dim, 64)
        else:
            self.mem_dense_q1 = nn.Linear(obs_dim, 64)
        self.mem_LSTM_q1 = nn.LSTM(input_size = 64, hidden_size = 64, num_layers = 1, batch_first = True)
        
        # post combination
        #self.post_comb_dense1_q1 = nn.Linear(64 + 64, 64)
        self.post_comb_dense2_q1 = nn.Linear(64, 1)
        
        # ----------------------- Q2 ---------------------------
        # current feature extraction
        self.curr_fe_dense1_q2 = nn.Linear(obs_dim + action_dim, 64)
        self.curr_fe_dense2_q2 = nn.Linear(64, 64)
        
        # memory
        if use_past_actions:
            self.mem_dense_q2 = nn.Linear(obs_dim + action_dim, 64)
        else:
            self.mem_dense_q2 = nn.Linear(obs_dim, 64)
        self.mem_LSTM_q2 = nn.LSTM(input_size = 64, hidden_size = 64, num_layers = 1, batch_first = True)
        
        # post combination
        #self.post_comb_dense1_q2 = nn.Linear(64 + 64, 64)
        self.post_comb_dense2_q2 = nn.Linear(64, 1)
        

    def forward(self, o, a, o_hist, a_hist, hist_len) -> tuple:
        """o, a, o_hist, a_hist, hist_len are torch tensors. Shapes:
        o:        torch.Size([batch_size, obs_dim])
        a:        torch.Size([batch_size, action_dim])
        o_hist:   torch.Size([batch_size, history_length, obs_dim])
        a_hist:   torch.Size([batch_size, history_length, action_dim])
        hist_len: torch.Size(batch_size)
        
        returns: tuple with two elements, each having shape torch.Size([batch_size, 1])
        
        Note: 
        The one-layer LSTM is defined with batch_first=True, hence it expects input in form of:
        x = (batch_size, seq_length, obs_dim)
        
        The call <out, (hidden, cell) = LSTM(x)> results in: 
        out:    Output (= hidden state) of LSTM for each time step with shape (batch_size, seq_length, hidden_size).
        hidden: The hidden state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        cell:   The cell state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        """

        #-------------------- Q1 ------------------------
        
        #------ current feature extraction ------
        # concatenate obs and act
        oa = torch.cat([o, a], dim=1)
        curr_fe_q1 = F.relu(self.curr_fe_dense1_q1(oa))
        curr_fe_q1 = F.relu(self.curr_fe_dense2_q1(curr_fe_q1))
        
        #------ memory ------
        # dense layer
        if self.use_past_actions:
            x_mem_q1 = F.relu(self.mem_dense_q1(torch.cat([o_hist, a_hist], dim=2)))
        else:
            x_mem_q1 = F.relu(self.mem_dense_q1(o_hist))
        
        # LSTM
        #self.mem_LSTM_q1.flatten_parameters()
        extracted_mem_q1, (_, _) = self.mem_LSTM_q1(x_mem_q1)

        # get selection index according to history lengths (no-history cases will be masked later)
        h_idx = copy.deepcopy(hist_len)
        h_idx[h_idx == 0] = 1
        h_idx -= 1
        
        # select LSTM output, resulting shape is (batch_size, hidden_dim)
        hidden_mem_q1 = extracted_mem_q1[torch.arange(extracted_mem_q1.size(0)), h_idx]

        # mask no-history cases to yield zero extracted memory
        hidden_mem_q1[hist_len == 0] = 0
        
        #------ post combination ------
        # concatenate current feature extraction with generated memory
        q1 = torch.cat([curr_fe_q1, hidden_mem_q1], dim=1)
        
        # final dense layers
        #q1 = F.relu(self.post_comb_dense1_q1(q1))
        q1 = self.post_comb_dense2_q1(q1)
        
        #-------------------- Q2 ------------------------
        
        #------ current feature extraction ------
        # concatenate obs and act
        oa = torch.cat([o, a], dim=1)
        curr_fe_q2 = F.relu(self.curr_fe_dense1_q2(oa))
        curr_fe_q2 = F.relu(self.curr_fe_dense2_q2(curr_fe_q2))
        
        #------ memory ------
        # dense layer
        if self.use_past_actions:
            x_mem_q2 = F.relu(self.mem_dense_q2(torch.cat([o_hist, a_hist], dim=2)))
        else:
            x_mem_q2 = F.relu(self.mem_dense_q2(o_hist))
        
        # LSTM
        #self.mem_LSTM_q2.flatten_parameters()
        extracted_mem_q2, (_, _) = self.mem_LSTM_q2(x_mem_q2)

        # get selection index according to history lengths (no-history cases will be masked later)
        h_idx = copy.deepcopy(hist_len)
        h_idx[h_idx == 0] = 1
        h_idx -= 1
        
        # select LSTM output, resulting shape is (batch_size, hidden_dim)
        hidden_mem_q2 = extracted_mem_q2[torch.arange(extracted_mem_q2.size(0)), h_idx]
        
        # mask no-history cases to yield zero extracted memory
        hidden_mem_q2[hist_len == 0] = 0
        
        #------ post combination ------
        # concatenate current feature extraction with generated memory
        q2 = torch.cat([curr_fe_q2, hidden_mem_q2], dim=1)
        
        # final dense layers
        #q2 = F.relu(self.post_comb_dense1_q2(q2))
        q2 = self.post_comb_dense2_q2(q2)

        #--------------- return output -----------------
        # create dict for logging
        critic_net_info = dict(Q1_CurFE = curr_fe_q1.detach().mean().cpu().numpy(),
                               Q2_CurFE = curr_fe_q2.detach().mean().cpu().numpy(),
                               Q1_ExtMemory = hidden_mem_q1.detach().mean().cpu().numpy(),
                               Q2_ExtMemory = hidden_mem_q2.detach().mean().cpu().numpy())
        
        return q1, q2, critic_net_info
    
    def Q1(self, o, a, o_hist, a_hist, hist_len) -> torch.tensor:
        """Same as forward, but only with first critic. Does NOT return logging info."""
        #-------------------- Q1------------------------
        
        #------ current feature extraction ------
        # concatenate obs and act
        oa = torch.cat([o, a], dim=1)
        curr_fe_q1 = F.relu(self.curr_fe_dense1_q1(oa))
        curr_fe_q1 = F.relu(self.curr_fe_dense2_q1(curr_fe_q1))
        
        #------ memory ------
        # dense layer
        if self.use_past_actions:
            x_mem_q1 = F.relu(self.mem_dense_q1(torch.cat([o_hist, a_hist], dim=2)))
        else:
            x_mem_q1 = F.relu(self.mem_dense_q1(o_hist))
        
        # LSTM
        #self.mem_LSTM_q1.flatten_parameters()
        extracted_mem_q1, (_, _) = self.mem_LSTM_q1(x_mem_q1)

        # get selection index according to history lengths (no-history cases will be masked later)
        h_idx = copy.deepcopy(hist_len)
        h_idx[h_idx == 0] = 1
        h_idx -= 1
        
        # select LSTM output, resulting shape is (batch_size, hidden_dim)
        hidden_mem_q1 = extracted_mem_q1[torch.arange(extracted_mem_q1.size(0)), h_idx]
        
        # mask no-history cases to yield zero extracted memory
        hidden_mem_q1[hist_len == 0] = 0
        
        #------ post combination ------
        # concatenate current feature extraction with generated memory
        q1 = torch.cat([curr_fe_q1, hidden_mem_q1], dim=1)
        
        # final dense layers
        #q1 = F.relu(self.post_comb_dense1_q1(q1))
        q1 = self.post_comb_dense2_q1(q1)

        # return output
        return q1
