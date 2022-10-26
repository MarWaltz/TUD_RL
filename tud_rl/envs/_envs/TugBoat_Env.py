import math

import gym
from matplotlib import pyplot as plt
from tud_rl.envs._envs.CybershipII import CyberShipII
from tud_rl.envs._envs.TitoNeri import TitoNeri
from tud_rl.envs._envs.VesselFnc import ED, angle_to_2pi, xy_from_polar
from tud_rl.envs._envs.VesselPlots import rotate_point


class TugBoat_Env(gym.Env):
    """This environment contains two tug boats of type TitoNeri pulling a CyberShip II."""
    def __init__(self):
        super().__init__()

        # simulation settings
        self.delta_t = 5.0   # simulation time interval (in s)
        self.E_max = 10
        self.N_max = 10


    def reset(self):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # init ship
        N_S = self.N_max/2
        E_S = self.E_max/2
        psi_S = 0.0
        self.ship = CyberShipII(N_init   = N_S,
                                E_init   = E_S, 
                                psi_init = psi_S,
                                u_init   = 0.0,
                                v_init   = 0.0,
                                r_init   = 0.0,
                                delta_t  = self.delta_t,
                                N_max    = self.N_max,
                                E_max    = self.E_max)
       
        # compute initial positions of tugs
        self.alpha1, self.alpha2 = 0.0, 0.0
        self.ltow1, self.ltow2   = 1.0, 1.0
        self.lT1, self.lT2       = 0.5, 0.5

        N1 = N_S - math.cos(psi_S)*self.ship.l1 - math.cos(psi_S + self.alpha1)*(self.ltow1 + self.lT1)
        E1 = E_S - math.sin(psi_S)*self.ship.l1 - math.sin(psi_S + self.alpha1)*(self.ltow1 + self.lT1)
        psi1 = psi_S + self.alpha1

        N2 = N_S + math.cos(psi_S)*self.ship.l2 + math.cos(psi_S + self.alpha2)*(self.ltow2 + self.lT2)
        E2 = E_S + math.sin(psi_S)*self.ship.l2 + math.sin(psi_S + self.alpha2)*(self.ltow2 + self.lT2)
        psi2 = psi_S + self.alpha2

        # init two tug boats
        self.tug1 = TitoNeri(N_init   = N1, 
                             E_init   = E1, 
                             psi_init = psi1, 
                             u_init   = 0.0, 
                             v_init   = 0.0, 
                             r_init   = 0.0, 
                             delta_t  = self.delta_t, 
                             N_max    = self.N_max, 
                             E_max    = self.E_max, 
                             tug1     = True)

        self.tug2 = TitoNeri(N_init   = N2, 
                             E_init   = E2, 
                             psi_init = psi2, 
                             u_init   = 0.0, 
                             v_init   = 0.0, 
                             r_init   = 0.0, 
                             delta_t  = self.delta_t, 
                             N_max    = self.N_max, 
                             E_max    = self.E_max, 
                             tug1     = True)
        # init state
        self._set_state()
        self.state_init = self.state
        return self.state

    def step(self, a):
        pass

    def _set_state(self):
        self.state = 0.0
        return

    def _calculate_reward(self, a):
        self.r = 0
        return

    def _done(self):
        return False


    def _render_ship(self, ax, vessel, color):
        """Draws the ship on the axis. Returns the ax."""
        # quick access
        l = vessel.length/2
        b = vessel.width/2
        N, E, head = vessel.eta

        # get rectangle/polygon end points in UTM
        if type(vessel).__name__ == "CyberShipII":
            A = (E - b, N + vessel.l2)
            B = (E + b, N + vessel.l2)
            C = (E - b, N - vessel.l1)
            D = (E + b, N - vessel.l1)
        else:
            A = (E - b, N + l)
            B = (E + b, N + l)
            C = (E - b, N - l)
            D = (E + b, N - l)

        # rotate them according to heading
        A = rotate_point(x=A[0], y=A[1], cx=E, cy=N, angle=-head)
        B = rotate_point(x=B[0], y=B[1], cx=E, cy=N, angle=-head)
        C = rotate_point(x=C[0], y=C[1], cx=E, cy=N, angle=-head)
        D = rotate_point(x=D[0], y=D[1], cx=E, cy=N, angle=-head)

        xs = [A[0], B[0], D[0], C[0], A[0]]
        ys = [A[1], B[1], D[1], C[1], A[1]]
        ax.plot(xs, ys, color=color, linewidth=2.0)
        return ax

    def _connect_ships(self, ax, ship : CyberShipII, tug : TitoNeri, tug1 : bool):
        if tug1:
            E_add, N_add = xy_from_polar(r=ship.l1, angle=angle_to_2pi(ship.eta[2] + math.pi))
            rope_end_N = ship.eta[0] + N_add
            rope_end_E = ship.eta[1] + E_add

            E_add, N_add = xy_from_polar(r=self.lT1, angle=tug.eta[2])
            rope_start_N = tug.eta[0] + N_add
            rope_start_E = tug.eta[1] + E_add
        else:
            E_add, N_add = xy_from_polar(r=ship.l2, angle=ship.eta[2])
            rope_end_N = ship.eta[0] + N_add
            rope_end_E = ship.eta[1] + E_add

            E_add, N_add = xy_from_polar(r=self.lT2, angle=angle_to_2pi(tug.eta[2] + math.pi))
            rope_start_N = tug.eta[0] + N_add
            rope_start_E = tug.eta[1] + E_add

        ax.plot([rope_start_E, rope_end_E], [rope_start_N, rope_end_N], color="black")
        ax.text(rope_start_E + 0.2, rope_start_N,\
             f"L: {ED(N0=rope_start_N, E0=rope_start_E, N1=rope_end_N, E1=rope_end_E):.2f}")
        return ax

    def render(self, data=None):
        """Renders the current environment. Note: The 'data' argument is needed since a recent update of the 'gym' package."""

        # check whether figure has been initialized
        if len(plt.get_fignums()) == 0:
            self.f, self.ax1 = plt.subplots(1, 1, figsize=(10, 10))
            plt.ion()
            plt.show()

        if self.step_cnt % 1 == 0:
            for ax in [self.ax1]:
                ax.clear()
                ax.set_xlim(0.0, self.E_max)
                ax.set_ylim(0.0, self.N_max)

                ax = self._render_ship(ax=ax, vessel=self.ship, color="red")
                ax = self._render_ship(ax=ax, vessel=self.tug1, color="green")
                ax = self._render_ship(ax=ax, vessel=self.tug2, color="blue")

                ax = self._connect_ships(ax=ax, ship=self.ship, tug=self.tug1, tug1=True)
                ax = self._connect_ships(ax=ax, ship=self.ship, tug=self.tug2, tug1=False)

            plt.pause(0.001)
