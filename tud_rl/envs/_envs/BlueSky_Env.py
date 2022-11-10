import bluesky as bs
import gym
import numpy as np
from bluesky.stack import simstack
from bluesky.tools.aero import (metres_to_feet_rounded,
                                metric_spd_to_knots_rounded)
from bluesky.tools.geo import latlondist, qdrpos
from gym import spaces
from matplotlib import pyplot as plt
from mycolorpy import colorlist as mcp

COLORS = [plt.rcParams["axes.prop_cycle"].by_key()["color"][i] for i in range(8)] + mcp.gen_color(cmap="tab20b", n=20) 


class BlueSky_Env(gym.Env):
    """Aircraft simulation env based on the BlueSky simulator of Ellerbroek and Hoekstra."""
    def __init__(self):
        super(BlueSky_Env, self).__init__()

        # config
        obs_size = 1
        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Discrete(3)

        self.delta_t = 0.1
        self._max_episode_steps = 500

        # initialize BlueSky simulator
        bs.init(mode='sim', detached=True)

        # activate CD&R with only horizontal MVP resolution
        bs.stack.stack("ASAS ON")
        bs.stack.stack("RESO MVP")
        bs.stack.stack("RMETHH HDG")
        bs.stack.stack("RMETHV OFF")

        # set simulation time
        bs.stack.stack(f"DT {self.delta_t}")
        simstack.process()


    def reset(self):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # useful commands:
        # CRE acid, type, lat, lon, hdg, alt (FL or ft), spd (kts)
        # ADDWPT acid, (wpname/lat,lon),[alt],[spd],[afterwp],[beforewp]

        # place some aircrafts
        #bs.traf.cre(acid="001", actype="A320", aclat=10.3, aclon=10.0, achdg=180, acalt=12000, acspd=250)
        #bs.traf.cre(acid="002", actype="A320", aclat=10.0, aclon=10.0, achdg=0, acalt=12000, acspd=250)
        bs.stack.stack("CRE 001, A320, 10.3, 10.0, 180, FL120, 250")
        bs.stack.stack("CRE 002, A320, 10.0, 10.0, 0, FL120, 250")

        # turn on LNAV, turn off VNAV (we stay at one altitude)
        bs.stack.stack("LNAV 001, ON")
        bs.stack.stack("LNAV 002, ON")
        bs.stack.stack("VNAV 001, OFF")
        bs.stack.stack("VNAV 002, OFF")
        simstack.process()

        # compute and set destinations
        dest_lat1, dest_lon1 = qdrpos(latd1=bs.traf.lat[0], lond1=bs.traf.lon[0], qdr=bs.traf.hdg[0], dist=100)
        dest_lat2, dest_lon2 = qdrpos(latd1=bs.traf.lat[1], lond1=bs.traf.lon[1], qdr=bs.traf.hdg[1], dist=100)

        # add some waypoints
        bs.stack.stack(f"ADDWPT 001, {dest_lat1}, {dest_lon1}, 12000, 250")
        bs.stack.stack(f"ADDWPT 002, {dest_lat2}, {dest_lon2}, 12000, 250")
        simstack.process()

        # init state
        self._set_state()
        self.state_init = self.state
        return self.state
  
    def _set_state(self):
        self.state = None

    def step(self, a):
        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.delta_t
 
        # update simulator
        bs.sim.step()

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward(a)
        d = self._done()
        return self.state, self.r, d, {}
 
    def _calculate_reward(self, a):
        self.r = 0.0

    def _done(self):
        return False

    def render(self, mode=None):
        """Renders the current environment."""

        # plot every nth timestep
        if self.step_cnt % 10 == 0: 
            
            # init figure
            if len(plt.get_fignums()) == 0:
                self.f, self.ax1 = plt.subplots(1, 1, figsize=(10, 10))
                plt.ion()
                plt.show()           
            
            # set screen
            self.ax1.clear()
            self.ax1.set_xlim(8, 12)
            self.ax1.set_ylim(8, 12)
            self.ax1.set_xlabel("Lon [°]")
            self.ax1.set_ylabel("Lat [°]")

            for i, acid in enumerate(bs.traf.id):
                # show aircraft
                self.ax1.scatter(bs.traf.lon[i], bs.traf.lat[i], color=COLORS[i])

                # information
                s = f"ID:  {acid}" + "\n" +\
                    f"hdg: {bs.traf.hdg[i]:.1f}" + "\n" +\
                    f"alt: {metres_to_feet_rounded(bs.traf.alt[i])}" + "\n" +\
                    f"spd: {metric_spd_to_knots_rounded(bs.traf.cas[i])}"
                self.ax1.text(x=bs.traf.lon[i], y=bs.traf.lat[i], s=s, color=COLORS[i])
                
                # waypoints
                self.ax1.scatter(bs.traf.actwp.lon[i], bs.traf.actwp.lat[i], color=COLORS[i])

            plt.pause(0.001)
