import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from tud_rl.envs._envs.VesselFnc import (COLREG_COLORS, ED, NM_to_meter,
                                         angle_to_2pi, bng_rel,
                                         get_ship_domain, rtd)


def rotate_point(x, y, cx, cy, angle):
    """Rotates a point (x,y) around origin (cx,cy) by an angle (defined counter-clockwise with zero at y-axis)."""

    # translate point to origin
    tempX = x - cx
    tempY = y - cy

    # apply rotation
    rotatedX = tempX * math.cos(angle) - tempY * math.sin(angle)
    rotatedY = tempX * math.sin(angle) + tempY * math.cos(angle)

    # translate back
    return rotatedX + cx, rotatedY + cy


def get_rect(E, N, width, length, heading, **kwargs):
    """Returns a patches.rectangle object. heading in rad."""

    # quick access
    x = E - width/2
    y = N - length/2, 
    cx = E
    cy = N
    heading = -heading   # negate since our heading is defined clockwise, contrary to plt rotations

    E0, N0 = rotate_point(x=x, y=y, cx=cx, cy=cy, angle=heading)

    # create rect
    return patches.Rectangle((E0, N0), width, length, rtd(heading), **kwargs)


def get_triangle(E, N, l, heading, **kwargs):
    """Returns a patches.polygon object. heading in rad."""

    # quick access 
    cx = E
    cy = N
    heading = -heading   # negate since our heading is defined clockwise, contrary to plt rotations

    topx = cx
    topy = cy + 2*l

    rightx = cx + l
    righty = cy - l

    leftx = cx - l
    lefty = cy - l

    topE, topN     = rotate_point(x=topx, y=topy, cx=cx, cy=cy, angle=heading)
    rightE, rightN = rotate_point(x=rightx, y=righty, cx=cx, cy=cy, angle=heading)
    leftE, leftN   = rotate_point(x=leftx, y=lefty, cx=cx, cy=cy, angle=heading)

    # create rect
    return patches.Polygon(xy=np.array([[topE, topN], [rightE, rightN], [leftE, leftN]]), **kwargs)


def plot_jet(axis, E, N, l, angle, **kwargs):
    """Adds a line to an axis (plt-object) originating at (E,N), having a given length l, 
        and following the angle (in rad). Returns the new axis."""

    # transform angle in [0, 2pi)
    angle = angle_to_2pi(angle)

    # 1. Quadrant
    if angle <= math.pi/2:
        E1 = E + math.sin(angle) * l
        N1 = N + math.cos(angle) * l
    
    # 2. Quadrant
    elif 3/2 *math.pi < angle <= 2*math.pi:
        angle = 2*math.pi - angle

        E1 = E - math.sin(angle) * l
        N1 = N + math.cos(angle) * l

    # 3. Quadrant
    elif math.pi < angle <= 3/2*math.pi:
        angle -= math.pi

        E1 = E - math.sin(angle) * l
        N1 = N - math.cos(angle) * l

    # 4. Quadrant
    elif math.pi/2 < angle <= math.pi:
        angle = math.pi - angle

        E1 = E + math.sin(angle) * l
        N1 = N - math.cos(angle) * l
    
    # draw on axis
    axis.plot([E, E1], [N, N1], **kwargs)
    return axis


class TrajPlotter:
    """This class provides a TrajPlotter customized for visualizing trajectories from the MMG-Env."""

    def __init__(self, plot_every, delta_t) -> None:
        self.plot_every = plot_every                             # seconds between markers in trajectory plotting
        self.delta_t = delta_t                                   # simulation step size

        self.plot_every_step = self.plot_every / self.delta_t    # number of timesteps between markers in trajectory plotting

    def step_to_minute(self, t):
        """Converts a simulation time to real time minutes."""
        return t * self.delta_t / 60.0

    def reset(self, OS, TSs, N_TSs):
        """Args:
        OS:    KVLCC2
        TSs:   list of KVLCC2-objects
        N_TSs: int"""

        self.N_TSs = N_TSs

        self.OS_traj_rud_angle = [OS.rud_angle]

        self.OS_traj_N = [OS.eta[0]]
        self.OS_traj_E = [OS.eta[1]]
        self.OS_traj_h = [OS.eta[2]]

        self.OS_col_N = []
        self.OS_col_E = []

        self.TS_traj_N = [[] for _ in range(N_TSs)]
        self.TS_traj_E = [[] for _ in range(N_TSs)]
        self.TS_traj_h = [[] for _ in range(N_TSs)]

        self.TS_spawn_steps = [[0] for _ in range(N_TSs)]

        for TS_idx, TS in enumerate(TSs):             
            self.TS_traj_N[TS_idx].append(TS.eta[0])
            self.TS_traj_E[TS_idx].append(TS.eta[1])
            self.TS_traj_h[TS_idx].append(TS.eta[2])

    def step(self, OS, TSs, respawn_flags, step_cnt):

        # agent update
        self.OS_traj_rud_angle.append(OS.rud_angle)
        self.OS_traj_N.append(OS.eta[0])
        self.OS_traj_E.append(OS.eta[1])
        self.OS_traj_h.append(OS.eta[2])

        if self.N_TSs > 0:

            # check whether we had a collision
            for TS in TSs:
                D = get_ship_domain(A=OS.ship_domain_A, B=OS.ship_domain_B, C=OS.ship_domain_C, D=OS.ship_domain_D, OS=OS, TS=TS)
                if ED(N0=OS.eta[0], E0=OS.eta[1], N1=TS.eta[0], E1=TS.eta[1], sqrt=True) <= D:
                    self.OS_col_N.append(OS.eta[0])
                    self.OS_col_E.append(OS.eta[1])
                    break

            # check TS respawning
            for TS_idx, flag in enumerate(respawn_flags):
                if flag:
                    self.TS_spawn_steps[TS_idx].append(step_cnt)

            # TS update
            for TS_idx, TS in enumerate(TSs):
                self.TS_traj_N[TS_idx].append(TS.eta[0])
                self.TS_traj_E[TS_idx].append(TS.eta[1])
                self.TS_traj_h[TS_idx].append(TS.eta[2])

    def plot_traj_fnc(self, 
                      E_max,
                      N_max,
                      goal_reach_dist,
                      Lpp,
                      step_cnt,
                      goal     = None,
                      goals    = None,
                      ax       = None, 
                      sit      = None, 
                      r_dist   = None, 
                      r_head   = None, 
                      r_coll   = None,
                      r_COLREG = None, 
                      r_comf   = None, 
                      star     = False):
        if ax is None:
            _, ax = plt.subplots()
            create_pdf = True
        else:
            create_pdf = False

        # E-axis
        ax.set_xlim(0, E_max)
        ax.set_xticks([NM_to_meter(nm) for nm in range(15) if nm % 2 == 1])
        ax.set_xticklabels([nm - 7 for nm in range(15) if nm % 2 == 1])
        ax.set_xlabel("East [NM]", fontsize=8)

        # N-axis
        ax.set_ylim(0, N_max)
        ax.set_yticks([NM_to_meter(nm) for nm in range(15) if nm % 2 == 1])
        ax.set_yticklabels([nm - 7 for nm in range(15) if nm % 2 == 1])
        ax.set_ylabel("North [NM]", fontsize=8)

        if not star:
            ax.scatter(goal["E"], goal["N"])
        else:
            for g in goals:
                ax.scatter(g["E"], g["N"])
        
        # OS trajectory
        ax.plot(self.OS_traj_E, self.OS_traj_N, color='black')

        # triangle at beginning
        rec = get_triangle(E = self.OS_traj_E[0], N = self.OS_traj_N[0], l=Lpp, heading = self.OS_traj_h[0],\
                facecolor="white", edgecolor="black", linewidth=1.5, zorder=10)
        ax.add_patch(rec)

        # OS markers
        OS_traj_E_m = [ele for idx, ele in enumerate(self.OS_traj_E) if idx % self.plot_every_step == 0]
        OS_traj_N_m = [ele for idx, ele in enumerate(self.OS_traj_N) if idx % self.plot_every_step == 0]
        ax.scatter(OS_traj_E_m, OS_traj_N_m, color="black", s=5)

        # OS collision
        if len(self.OS_col_N) > 0:
            ax.scatter(self.OS_col_E, self.OS_col_N, color="red", s=5)

        # TS
        for TS_idx in range(self.N_TSs):

            col = COLREG_COLORS[TS_idx]

            # add final step cnt since we finish here
            spawn_steps = self.TS_spawn_steps[TS_idx]
            if spawn_steps[-1] != step_cnt:
                spawn_steps.append(step_cnt)

            # trajectories
            if len(spawn_steps) == 1:
                ax.plot(self.TS_traj_E[TS_idx], self.TS_traj_N[TS_idx])
            else:
                for step_idx in range(len(spawn_steps)-1):
                    start = spawn_steps[step_idx]
                    end   = spawn_steps[step_idx+1]

                    E_traj = self.TS_traj_E[TS_idx][start:end]
                    N_traj = self.TS_traj_N[TS_idx][start:end]

                    # triangle at beginning
                    rec = get_triangle(E = E_traj[0], N = N_traj[0], l=Lpp, heading = self.TS_traj_h[TS_idx][start],\
                                       facecolor="white", edgecolor=col, linewidth=1.5, zorder=10)
                    ax.add_patch(rec)

                    # trajectory
                    ax.plot(E_traj, N_traj, color=col)

            # markers
            TS_traj_E_m = [ele for idx, ele in enumerate(self.TS_traj_E[TS_idx]) if idx % self.plot_every_step == 0]
            TS_traj_N_m = [ele for idx, ele in enumerate(self.TS_traj_N[TS_idx]) if idx % self.plot_every_step == 0]
            ax.scatter(TS_traj_E_m, TS_traj_N_m, color=col, s=5)

        # goal
        if not star:
            circ = patches.Circle((goal["E"], goal["N"]), radius=goal_reach_dist, edgecolor='blue', facecolor='none', alpha=0.3)
            ax.add_patch(circ)
        else:
            for i in range(len(goals)):
                if i == 0:
                    col = "black"
                else:
                    col = COLREG_COLORS[i-1]
                circ = patches.Circle((goals[i]["E"], goals[i]["N"]), radius=goal_reach_dist, edgecolor=col, facecolor='none', alpha=0.4)
                ax.add_patch(circ)

        if create_pdf:
            plt.savefig("TrajPlot.pdf")
        else:
            ax.grid(linewidth=1.0, alpha=0.425)

            if not star:

                #if all([ele is not None for ele in [r_dist, r_head, r_coll, r_COLREG, r_comf]]):
                    #ax.text(NM_to_meter(0.5), NM_to_meter(11.5), r"$r_{\rm dist}$: " + format(r_dist, '.2f'), fontdict={"fontsize" : 7})
                    #ax.text(NM_to_meter(0.5), NM_to_meter(10.5), r"$r_{\rm head}$: " + format(r_head, '.2f'), fontdict={"fontsize" : 7})
                    #ax.text(NM_to_meter(0.5), NM_to_meter(9.5), r"$r_{\rm coll}$: " + format(r_coll, '.2f'), fontdict={"fontsize" : 7})
                    #ax.text(NM_to_meter(0.5), NM_to_meter(8.5),  r"$r_{\rm COLR}$: " + format(r_COLREG, '.2f'), fontdict={"fontsize" : 7})
                    #ax.text(NM_to_meter(0.5), NM_to_meter(7.5),  r"$r_{\rm comf}$: " + format(r_comf, '.2f'), fontdict={"fontsize" : 7})
                    #ax.text(NM_to_meter(0.5), NM_to_meter(6.2),  r"$\sum r$: " + format(r_dist + r_head + r_coll + r_COLREG + r_comf, '.2f'), fontdict={"fontsize" : 7})
                
                ax.text(NM_to_meter(0.5), NM_to_meter(12.5), f"Case {sit}", fontdict={"fontsize" : 7})

                if sit not in [20, 21, 22, 23]:
                    ax.tick_params(axis='x', labelsize=8, which='both', bottom=False, top=False, labelbottom=False)
                    ax.set_xlabel("")
                else:
                    ax.tick_params(axis='x', labelsize=8)

                if sit not in [1, 5, 9, 13, 17, 21]:
                    ax.tick_params(axis='y', labelsize=8, which='both', left=False, right=False, labelleft=False)
                    ax.set_ylabel("")
                else:
                    ax.tick_params(axis='y', labelsize=8)
            
            return ax


    def step_to_minute(self, t):
        """Converts a simulation time to real time minutes."""
        return t * self.delta_t / 60.0


    def plot_dist(self, ax, sit, N_max, ship_domain_A, ship_domain_B, ship_domain_C, ship_domain_D):
        """Adds the distance plot to a given axis."""

        # Time axis
        T_min = math.ceil(self.step_to_minute(2000))
        ax.set_xlim(0, T_min)
        ax.set_xticks([t for t in range(T_min) if (t + 1) % 15 == 1])
        ax.set_xticklabels([t for t in range(T_min) if (t + 1) % 15 == 1])
        ax.set_xlabel("Time [min]", fontsize=8)

        # N-axis
        ax.set_ylim(-1000, N_max + 750)
        ax.set_yticks([NM_to_meter(nm) for nm in range(15) if (nm + 1) % 2 == 1])
        ax.set_yticklabels([nm for nm in range(15) if (nm + 1) % 2 == 1])
        ax.set_ylabel("Distance [NM]", fontsize=8)

        # horizontal line at zero distance
        ax.hlines(y=0, xmin=0, xmax=T_min, colors="grey", linestyles="dashed", linewidth=1.5)

        # TS
        for TS_idx in range(self.N_TSs):

            col = COLREG_COLORS[TS_idx]

            # get TS traj data
            TS_E_traj = self.TS_traj_E[TS_idx]
            TS_N_traj = self.TS_traj_N[TS_idx]

            # get distances
            TS_dists = []
            TS_coll_t = []

            for t in range(len(self.OS_traj_E)):

                N0 = self.OS_traj_N[t]
                E0 = self.OS_traj_E[t]
                N1 = TS_N_traj[t]
                E1 = TS_E_traj[t]

                # get relative bearing
                bng_rel_TS = bng_rel(N0=N0, E0=E0, N1=N1, E1=E1, head0=self.OS_traj_h[t])

                # compute ship domain
                D = get_ship_domain(A=ship_domain_A, B=ship_domain_B, C=ship_domain_C, D=ship_domain_D, OS=None, TS=None, ang=bng_rel_TS)

                # get euclidean distance between OS and TS
                ED_TS = ED(N0=N0, E0=E0, N1=N1, E1=E1, sqrt=True)

                TS_dists.append(ED_TS - D)

                # catch collisions
                if ED_TS - D <= 0:
                    TS_coll_t.append(t)

            # plot
            ax.plot(self.step_to_minute(np.arange(len(TS_dists))), TS_dists, color=col)

            # plot collisions
            if len(TS_coll_t) > 0:
                ax.vlines(x=self.step_to_minute(np.array(TS_coll_t)), ymin=0, ymax=N_max + 750, color="red", linewidth=1.0)

        ax.text(self.step_to_minute(100), NM_to_meter(12.5), f"Case {sit}", fontdict={"fontsize" : 7})

        if sit not in [20, 21, 22, 23]:
            ax.tick_params(axis='x', labelsize=8, which='both', bottom=False, top=False, labelbottom=False)
            ax.set_xlabel("")
        else:
            ax.tick_params(axis='x', labelsize=8)

        if sit not in [1, 5, 9, 13, 17, 21]:
            ax.tick_params(axis='y', labelsize=8, which='both', left=False, right=False, labelleft=False)
            ax.set_ylabel("")
        else:
            ax.tick_params(axis='y', labelsize=8)
        
        return ax


    def plot_rudder(self, ax, sit, rud_angle_max):
        """Adds the trajectory plot to a given axis."""

        # Time axis
        T_min = math.ceil(self.step_to_minute(2000))
        ax.set_xlim(0, T_min)
        ax.set_xticks([t for t in range(T_min) if (t + 1) % 15 == 1])
        ax.set_xticklabels([t for t in range(T_min) if (t + 1) % 15 == 1])
        ax.set_xlabel("Time [min]", fontsize=8)

        # N-axis
        ax.set_ylim(-rtd(rud_angle_max) - 2.0, rtd(rud_angle_max) + 2.0)
        #ax.set_yticks([NM_to_meter(nm) for nm in range(15) if (nm + 1) % 2 == 1])
        #ax.set_yticklabels([nm for nm in range(15) if (nm + 1) % 2 == 1])
        ax.set_ylabel("Rudder angle [°]", fontsize=8)

        # plot
        ax.plot(self.step_to_minute(np.arange(len(self.OS_traj_rud_angle))), [rtd(ang) for ang in self.OS_traj_rud_angle], color="black", linewidth=0.75)

        ax.text(self.step_to_minute(100), 17.5, f"Case {sit}", fontdict={"fontsize" : 7})

        if sit not in [20, 21, 22, 23]:
            ax.tick_params(axis='x', labelsize=8, which='both', bottom=False, top=False, labelbottom=False)
            ax.set_xlabel("")
        else:
            ax.tick_params(axis='x', labelsize=8)

        if sit not in [1, 5, 9, 13, 17, 21]:
            ax.tick_params(axis='y', labelsize=8, which='both', left=False, right=False, labelleft=False)
            ax.set_ylabel("")
        else:
            ax.tick_params(axis='y', labelsize=8)
        
        return ax
