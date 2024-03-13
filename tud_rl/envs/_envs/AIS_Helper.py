import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tud_rl.envs._envs.VesselFnc import bng_abs
from tud_rl.envs._envs.VesselPlots import rotate_point


class CPANet(nn.Module):
    """Defines a recurrent network."""
    
    def __init__(self, n_forecasts : int) -> None:
        super(CPANet, self).__init__()

        # memory
        self.mem_LSTM = nn.LSTM(input_size=2, hidden_size=64, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(64, 64)
        
        # post combination
        self.dense2 = nn.Linear(64, 2 * n_forecasts)

    def forward(self, x0, y0) -> tuple:
        """Shapes:
        x0:   torch.Size([batch_size, history_length, 1])
        y0:   torch.Size([batch_size, history_length, 1])
 
        Returns: 
        torch.Size([batch_size, 2 * n_forecasts])
        """

        #------ memory ------
        # LSTM
        _, (mem, _) = self.mem_LSTM(torch.cat([x0, y0], dim=2))
        mem = mem[0]

        # dense
        x = F.relu(self.dense1(mem))
        
        # final dense layers
        x = self.dense2(x)
        return x

class AIS_Ship:
    def __init__(self, data : dict) -> None:
        # Step size, already considered in the trajectories
        self.dt = 5

        # Select traj
        while True:
            tr = data[random.randrange(0, len(data))]
            tr_len = len(tr["ts"])

            # Should cover at least 90 min (90 min * (60 s / min) / (5 s) = 1080)
            if tr_len > 1080:

                # Select random part of the traj, not always from the beginning
                t_start = random.randrange(0, tr_len-1080)
                self.e_traj = tr["e"][t_start:]
                self.n_traj = tr["n"][t_start:]
                break

        # Careful: We start at the 21th point!
        self.ptr = 20

        # Compute speed since we need it for spawning
        ve = (self.e_traj[self.ptr+1] - self.e_traj[self.ptr]) / self.dt
        vn = (self.n_traj[self.ptr+1] - self.n_traj[self.ptr]) / self.dt
        self.v = np.sqrt(ve**2 + vn**2)

    def place_on_map(self, E_init : float, N_init : float, head_init : float) -> None:
        """Places vessel correctly by ensuring the trajectory has the correct origin and initial heading."""
        # Rotate traj to match initial heading
        data_head = bng_abs(N0=self.n_traj[self.ptr], E0=self.e_traj[self.ptr], N1=self.n_traj[self.ptr+1], E1=self.e_traj[self.ptr+1])
        rot_angle = head_init-data_head
        self.e_traj, self.n_traj = rotate_point(x=self.e_traj, y=self.n_traj, cx=self.e_traj[self.ptr], cy=self.n_traj[self.ptr],
                                                angle=-rot_angle)

        # Shift traj to match initial position
        self.e_traj = self.e_traj - self.e_traj[self.ptr] + E_init
        self.n_traj = self.n_traj - self.n_traj[self.ptr] + N_init
        
        # Init dynamics
        self.e = self.e_traj[self.ptr]
        self.n = self.n_traj[self.ptr]
        self.head = head_init

    def update_dynamics(self) -> None:
        # Increment ptr if it does not exceed the traj lenght (-1 for Python index and -1 for heading computation)
        if self.ptr < len(self.e_traj)-2:
            self.ptr += 1

            # Update position
            self.e = self.e_traj[self.ptr]
            self.n = self.n_traj[self.ptr]
            self.head = bng_abs(N0=self.n_traj[self.ptr-1], E0=self.e_traj[self.ptr-1], N1=self.n, E1=self.e)

            # Update speed
            ve = (self.e - self.e_traj[self.ptr-1]) / self.dt
            vn = (self.n - self.n_traj[self.ptr-1]) / self.dt
            self.v = np.sqrt(ve**2 + vn**2)


if __name__ == "__main__":
    # Load data
    import pickle

    import matplotlib.pyplot as plt

    with open("C:/Users/Martin Waltz/Desktop/Forschung/RL/Obstacle Avoidance/Code for Revision/traj_1day_utm.pickle","rb") as f:
        data = pickle.load(f)

    for _ in range(1):
        ship = AIS_Ship(E_init=10.0, N_init=10.0, head_init=0.0, data=data)
        for _ in range(100):
            print(ship.v)
            ship.update_dynamics()
