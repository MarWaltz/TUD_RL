import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tud_rl.envs._envs.VesselFnc import bng_abs


class CPANet(nn.Module):
    """Defines a recurrent network."""
    
    def __init__(self, n_forecasts : int) -> None:
        super(CPANet, self).__init__()

        # memory
        self.mem_LSTM = nn.LSTM(input_size=2, hidden_size=256, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(256 + 2, 256)
        
        # post combination
        self.dense_add = nn.Linear(256, 256)
        self.dense2 = nn.Linear(256, 2 * n_forecasts)

    def forward(self, x0, y0, pos) -> tuple:
        """Shapes:
        x0:   torch.Size([batch_size, history_length, 1])
        y0:   torch.Size([batch_size, history_length, 1])
        pos:  torch.Size([batch_size, 2])
 
        Returns: 
        torch.Size([batch_size, 2 * n_forecasts])
        """

        #------ memory ------
        # LSTM
        _, (mem, _) = self.mem_LSTM(torch.cat([x0, y0], dim=2))
        mem = mem[0]

        # dense
        combo = torch.cat([mem, pos], dim=1)
        x = F.leaky_relu(self.dense1(combo))
        x = F.leaky_relu(self.dense_add(x))
        
        # final dense layers
        x = self.dense2(x)
        return x

class AIS_Ship:
    def __init__(self, data : dict, sit : int, ttpt : float) -> None:
        assert sit in [0, 1], "Unknown situation."
        assert ttpt > 0, "Need positive time to turning point."
        
        # Step size, already considered in the trajectories
        self.dt = 5

        # Try until success since a few trajectories might be too short
        while True:
            try:
                # Select trajectory
                tr = data[random.randrange(0, len(data))]
                self.e_traj = tr["e"]
                self.n_traj = tr["n"]

                # Add some noise
                self._add_noise_to_traj()

                # Find turning point
                i_turn = self._find_turning_point()

                # Init position according to ttpt
                n = ttpt // self.dt

                if sit == 0:
                    self.ptr = int(i_turn - n)
                else:
                    self.ptr = int(i_turn + n)

                self.e = self.e_traj[self.ptr]
                self.n = self.n_traj[self.ptr]

                # Compute speed
                self.ve = (self.e_traj[self.ptr+1] - self.e_traj[self.ptr]) / self.dt
                self.vn = (self.n_traj[self.ptr+1] - self.n_traj[self.ptr]) / self.dt
                self.v = np.sqrt(self.ve**2 + self.vn**2)

                # Compute heading
                self.head = bng_abs(N0=self.n_traj[self.ptr], E0=self.e_traj[self.ptr], N1=self.n_traj[self.ptr+1], E1=self.e_traj[self.ptr+1])
            except:
                print("Out")
                continue
            else:
                break

    def _upd_dynamics(self) -> None:
        # Increment ptr if it does not exceed the traj lenght (-1 for Python index and -1 for heading computation)
        if self.ptr < len(self.e_traj)-2:
            self.ptr += 1

            # Update position
            self.e = self.e_traj[self.ptr]
            self.n = self.n_traj[self.ptr]
            self.head = bng_abs(N0=self.n_traj[self.ptr-1], E0=self.e_traj[self.ptr-1], N1=self.n, E1=self.e)

            # Update speed
            self.ve = (self.e - self.e_traj[self.ptr-1]) / self.dt
            self.vn = (self.n - self.n_traj[self.ptr-1]) / self.dt
            self.v = np.sqrt(self.ve**2 + self.vn**2)

    def _find_turning_point(self) -> int:
        """Finds the turning point in form of an index of the trajectory. Ignores 'north' for the moment."""
        return np.argmin(self.e_traj)
        #return np.argmin(np.abs(np.diff(self.e_traj)))

    def _add_noise_to_traj(self) -> None:
        """Adds some Gaussian white noise to the trajectory to increase diversity."""
        self.e_traj += np.random.randn(len(self.e_traj)) * 0.2 # 0.1 worked
        self.n_traj += np.random.randn(len(self.n_traj)) * 0.2 # 0.1 worked


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
