# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import gym
from gym import spaces
import random
import re
import weakref
import time
import cv2
import numpy as np
import pygame
import subprocess
from utils import *
from queue import Queue
from gym.utils import seeding
from sensors import RGBCamera, Lidar, CollisionSensor, LaneInvasionSensor, Vehicle
from affordance import AffordanceGetter
from human_control import KeyboardControl
from hud import HUD
from planner import RoadOption, compute_route_waypoints

from no_rendering import World as no_rendering_world
from no_rendering import HUD as no_rendering_HUD
from pygame.locals import *
from controller import PIDLongitudinalController
# Module Defines
TITLE_WORLD = 'WORLD'
TITLE_HUD = 'HUD'
COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

USE_ROUTYE_WAYPOINT = True


"""Parses the arguments received from commandline and runs the game loop"""

# Define arguments that will be received and parsed
argparser = argparse.ArgumentParser(
    description='CARLA No Rendering Mode Visualizer')
argparser.add_argument(
    '-v', '--verbose',
    action='store_true',
    dest='debug',
    help='print debug information')
argparser.add_argument(
    '--host',
    metavar='H',
    default='127.0.0.1',
    help='IP of the host server (default: 127.0.0.1)')
argparser.add_argument(
    '-p', '--port',
    metavar='P',
    default=2000,
    type=int,
    help='TCP port to listen to (default: 2000)')
argparser.add_argument(
    '--res',
    metavar='WIDTHxHEIGHT',
    default='1280x720',
    help='window resolution (default: 1280x720)')
argparser.add_argument(
    '--filter',
    metavar='PATTERN',
    default='vehicle.*',
    help='actor filter (default: "vehicle.*")')
argparser.add_argument(
    '--map',
    metavar='TOWN',
    default=None,
    help='start a new episode at the given TOWN')
argparser.add_argument(
    '--no-rendering',
    action='store_true',
    help='switch off server rendering')
argparser.add_argument(
    '--show-triggers',
    action='store_true',
    help='show trigger boxes of traffic signs')
argparser.add_argument(
    '--show-connections',
    action='store_true',
    help='show waypoint connections')
argparser.add_argument(
    '--show-spawn-points',
    action='store_true',
    help='show recommended spawn points')

# Parse arguments
args = argparser.parse_args()
args.description = argparser.description
args.width, args.height = [int(x) for x in args.res.split('x')]


class CarlaEnv(gym.Env):

    def __init__(self, host="127.0.0.1", port=2000, viewer_res=(640, 480),fps=20, synchronous=False):
        super().__init__()
        pygame.init()
        pygame.font.init()
        self.width, self.height = viewer_res
        self.display = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.display.fill((0,0,0))
        pygame.display.flip()
        self.clock = pygame.time.Clock()

        self.seed()

        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.rgb_camera = None
        self.lidar_sensor = None
        self.affordancegetter = None
        self.observation = self.observation_buffer = None   # Last received observation
        self.viewer_image = self.viewer_image_buffer = None # Last received image to show in the viewer
        self.command = carla.VehicleControl()
        self.closed = False
        self.last_action = np.zeros((2,1))
        self.action_space = spaces.Box(
                low=-1,
                high=1,
                shape=(2,),
                dtype=np.float32
        )
        self.observation_space = spaces.Box(
                low=-1,
                high=1,
                shape=(11,),
                dtype=np.float32
        )
        self._control = carla.VehicleControl()
        self._lights = carla.VehicleLightState.NONE
        self._steer_cache = 0.0
        self.fps = fps
        self.synchronous = False
        self.current_maneuver = 0

        # to be deleted
        self.distance_traveled_buffer = 0
        self.rest_counter = 0 
        self.last_reward = 0.0
        self.distance_from_center = 0.0
        self.previous_yaw = 0.0
        self.current_yaw = 0.0
        self.desired_speed = 6 # original desired speed 30km/h
        self.max_desired_speed = 6

        self.action_smoothing = 0.1
        self.past_steering = 0.0
        self.max_target_speed = 90
        self.steering_ratio = 1.0
        self.speed_reward = 0.0
        self.cross_track_error_reward = 0.0
        self.heading_error_reward = 0.0

        self._max_speed = 25 # max speed 120 km/h
        self._max_accleration = self._max_speed/(1/self.fps)
        self._world = no_rendering_world("no rendering mode", args, timeout=2.0)
        self.client = carla.Client(host, port)
        self._map = self.client.get_world().get_map()
        self.carla_world = self.client.get_world()

        self.hud_no_rendering = no_rendering_HUD(TITLE_HUD, self.width, self.height )

        self._world.start(self.hud_no_rendering)
        

    def reset(self): 

        self._world.select_hero_actor()
        
        self.player = self._world.hero_actor

        
        self.collision_sensor = self._world.collision_sensor
        self.affordancegetter = AffordanceGetter(self.carla_world, self.player)

        self._lon_controller = PIDLongitudinalController(self.player)
        

        self.terminal_state = False
        self.num_routes_completed = -1
        self.set_route()


        self.extra_info = []        # List of extra info shown on the HUD

        self.observation = self.observation_buffer = None   # Last received observation
        self.viewer_image = self.viewer_image_buffer = None # Last received image to show in the viewer

        self.previous_location = self.player.get_transform().location
        self.distance_traveled = 0.0
        self.rest_counter = 0 
        self.distance_traveled_buffer = 0.0
        self.routes_completed = 0.0
        self.average_heading_error = 0.0
        self.player.set_light_state(self._lights)

        final_affordance = self.affordancegetter.get_affordance()
        
        # Reset desired speed
        self.distance_to_tl, self.desired_speed = tl_affecting_vehicle(self.player)
        

        #Set current waypoint as start waypoint and next waypoint from route

        self.current_waypoint = self.start_wp

        if len(self.route_waypoints) > 2:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[1]
        else:
            self.next_waypoint = self.end_wp


        # Calculate the Deviations according to the waypoints of the route
        distance_to_lane_center_wp, heading_error_wp, cross_track_error_wp = distance_to_lane_center(self.current_waypoint, self.next_waypoint, self.player)

        # Call external states fn
        last_state = self.state_fn(final_affordance,self.last_action, distance_to_lane_center_wp, heading_error_wp, cross_track_error_wp, self.distance_to_tl)

        # Set up surrounding vehicles

        return last_state

    def step(self, action):

         # Asynchronous update logic
        if not self.synchronous:
            if self.fps <= 0:
                # Go as fast as possible
                self.clock.tick()
            else:
                # Sleep to keep a steady fps
                self.clock.tick_busy_loop(self.fps)

        # Create new route on route completion
        if self.current_waypoint_index >= len(self.route_waypoints)-1:
            self.set_route()

        if self.player is None:
            self.reset()

        # Take action
        if action is not None:
            #self.process_control(action)
            self.run_step(action)


        # Tick game
        self._world.tick(self.clock)
        self.hud_no_rendering.tick(self.clock)
        

        final_affordance = self.affordancegetter.get_affordance()

        self.distance_to_tl, self.desired_speed = tl_affecting_vehicle(self.player)


        # Get vehicle transform
        transform = self.player.get_transform()

        # Keep track of closest waypoint on the route
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                        vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0: # Did we pass the waypoint?
                waypoint_index += 1 # Go to next waypoint
            else:
                break
        
        self.current_waypoint_index = waypoint_index

        # Check for route completion
        if self.current_waypoint_index < len(self.route_waypoints)-1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
            self.average_heading_error = self.compute_average_heading_error(self.route_waypoints)
        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(self.route_waypoints)
        
        # Terminal on max distance
        if self.routes_completed > 1.0:
            self.terminal_state = True
        

        # Calculate the Deviations according to the waypoints of the route
        distance_to_lane_center_wp, heading_error_wp, cross_track_error_wp = distance_to_lane_center(self.current_waypoint, self.next_waypoint, self.player)


        # Convert road_maneuver to digital 
        self.road_maneuver() 


        # Call external states fn
        last_state = self.state_fn(final_affordance, action, distance_to_lane_center_wp, heading_error_wp, cross_track_error_wp, self.distance_to_tl)


        # Call external reward fn
        self.last_reward, self.speed_reward, self.cross_track_error_reward, self.heading_error_reward = self.setReward(final_affordance, distance_to_lane_center_wp, heading_error_wp, cross_track_error_wp)

        # Calculate distance traveled
        self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location
        self.distance_from_center = distance_to_lane_center_wp


        #Reset if the vehicle stand still for 20s
        self.distance_traveled_buffer +=self.distance_traveled

        if self.distance_traveled_buffer == self.distance_traveled:
            self.rest_counter += 1
        
        if self.rest_counter >= 400:
            self.terminal_state = True
        
        
        #self._draw_path(life_time=60.0, skip=10)

        # Check for ESC press
        pygame.event.pump()
        if pygame.key.get_pressed()[K_ESCAPE]:
            self.close()

        # for sb3

        self.render()
        
        return last_state, self.last_reward, self.terminal_state, {}    

    def setReward(self, final_affordance, distance_to_lane_center_wp, heading_error_wp, cross_track_error_wp):
        
        
        # Reward for circle around    
        self.current_yaw = math.radians(self.player.get_transform().rotation.yaw)
        if abs(self.current_yaw - self.previous_yaw) > 0.1:
            yaw_penalty = -0.1
        else:
            yaw_penalty = 0.0

        #unwrapp the affordance
    
        ego_speed = final_affordance["ego_speed"]
        traffic_lights = final_affordance["traffic_light"]
        stop_sign = final_affordance["stop_sign"]        
        ego_acceleration =  final_affordance["ego_acceleration"]
        speed_limit = final_affordance["speed_limit"]/ 3.6
        distance_to_current_lane_center = final_affordance["distance_to_current_lane_center"]
        heading_error = final_affordance["heading_error"]
        crosstrack_error = final_affordance["crosstrack_error"]


        if USE_ROUTYE_WAYPOINT:
            crosstrack_error = cross_track_error_wp
            heading_error = heading_error_wp
            distance_to_current_lane_center = distance_to_lane_center_wp
        else:
            pass

        if traffic_lights == 0 and self.distance_to_tl <= 5: 
            additional_reward_for_redl = 11.0 - (self.distance_to_tl + ego_speed)
        else:
            additional_reward_for_redl = 0.0



        """
        # Speed reward
        if traffic_lights == 0: 
            desired_speed = 0.0
        else:
            desired_speed = speed_limit
        """
        #speed_reward = max(0, (1 - (abs(ego_speed - self.desired_speed) / self.max_desired_speed)))
        speed_reward = 1 - (abs(ego_speed - self.desired_speed) / self.max_desired_speed)

        cross_track_error_reward = self.cross_track_reward(distance_to_lane_center_wp)

        heading_error_reward = -heading_error

        # If distance from center > 5, stop
        if distance_to_current_lane_center > 5.0:
            overall_reward = -1
            self.terminal_state = True
        elif self.collision_sensor.impulse is not None:
            overall_reward = -1 - ego_speed
            self.terminal_state = True
        elif traffic_lights == 0 and self.distance_to_tl < 0.5 and self.desired_speed > 0:
            overall_reward = -1 - ego_speed
            self.terminal_state = True
        else:
            #overall_reward = np.clip((0.5*speed_reward + 0.3*cross_track_error_reward + 0.3*heading_error_reward - 2*yaw_penalty), a_min=-1, a_max=1)
            overall_reward = np.clip((0.5*speed_reward + 0.3*cross_track_error_reward + 0.3*heading_error_reward + yaw_penalty + 0.01*additional_reward_for_redl), a_min=-1, a_max=1)
        self.previous_yaw = self.current_yaw
        #print(speed_reward, cross_track_error_reward, heading_error_reward, 2*yaw_penalty, overall_reward)
        return overall_reward, speed_reward, cross_track_error_reward, heading_error_reward       


    def cross_track_reward(self, e_y):
         a = 0.005
         k_c = 0.15

         return math.pow(a,(abs(e_y)*k_c)) - 1
    

    def render_waypoints(self, surface):

        current_waypoint = self._map.get_waypoint(self.player.get_location(), project_to_road=True)
        way_point_x = int(current_waypoint.transform.location.x)
        way_point_y = int(current_waypoint.transform.location.y)
        radius = 20
        pygame.draw.circle(surface, COLOR_SCARLET_RED_1, (way_point_x, way_point_y), radius)

    def state_fn(self, final_affordance, action, distance_to_lane_center_wp, heading_error_wp, cross_track_error_wp, distance_to_tl):


        #unwrapp the affordance
    
        ego_speed = final_affordance["ego_speed"]
        traffic_lights = final_affordance["traffic_light"]
        stop_sign = final_affordance["stop_sign"]        
        ego_acceleration =  final_affordance["ego_acceleration"]
        speed_limit = final_affordance["speed_limit"]/ 3.6
        distance_to_current_lane_center = final_affordance["distance_to_current_lane_center"]
        heading_error = final_affordance["heading_error"]
        crosstrack_error = final_affordance["crosstrack_error"]

        if USE_ROUTYE_WAYPOINT:
            crosstrack_error = cross_track_error_wp
            heading_error = heading_error_wp
            distance_to_current_lane_center = distance_to_lane_center_wp
        else:
            pass

        status1 = ego_speed/self._max_speed
        status2 = ego_acceleration/self._max_accleration
        status3 = speed_limit*3.6/90
        status4 = self.average_heading_error/(math.pi)
        status5 = heading_error/(math.pi)
        status6 = (crosstrack_error)/5.0
        status7 = traffic_lights/4.0
        status8 = distance_to_tl/10.0
        status9 = self.current_maneuver/5
        status10 = action[0]
        status11 = action[1]
          
        self.last_action = action
                 
        state = [status1,status2,status3,status4,status5,status6,status7,status8,status9,status10,status11]
        return state



    def set_route(self):

        self.command.steer = float(0.0)
        self.command.throttle = float(0.0)
        #self.vehicle.control.brake = float(0.0)
        self.player.apply_control(self.command)

        # Generate waypoints along the lap
        self.start_wp, self.end_wp = [self._map.get_waypoint(spawn.location) for spawn in np.random.choice(self._map.get_spawn_points(), 2, replace=False)]
    
        self.route_waypoints = compute_route_waypoints(self._map, self.start_wp, self.end_wp, resolution=1.0)
        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        # Put vehicle at start point
        self.player.set_transform(self.start_wp.transform)

        self.player.set_simulate_physics(False) # Reset the car's physics
        self.player.set_simulate_physics(True)

        # Give 2 seconds to reset
        if self.synchronous:
            ticks = 0
            while ticks < self.fps * 2:
                self._world.tick()
                try:
                    self._world.wait_for_tick(seconds=1.0/self.fps + 0.1)
                    ticks += 1
                except:
                    pass
        else:
            time.sleep(2.0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def road_maneuver(self):



        if self.current_road_maneuver == RoadOption.VOID:              self.current_maneuver = -2
        elif self.current_road_maneuver == RoadOption.LEFT:            self.current_maneuver = -1
        elif self.current_road_maneuver == RoadOption.RIGHT:           self.current_maneuver = 1
        elif self.current_road_maneuver == RoadOption.STRAIGHT:        self.current_maneuver = 0
        elif self.current_road_maneuver == RoadOption.LANEFOLLOW:      self.current_maneuver = 2
        elif self.current_road_maneuver == RoadOption.CHANGELANELEFT:  self.current_maneuver = 3
        elif self.current_road_maneuver == RoadOption.CHANGELANERIGHT: self.current_maneuver = 4
        else:                                                          self.current_maneuver = 5



    def render(self):


         # Get maneuver name
        if self.current_road_maneuver == RoadOption.LANEFOLLOW: maneuver = "Follow Lane"
        elif self.current_road_maneuver == RoadOption.LEFT:     maneuver = "Left"
        elif self.current_road_maneuver == RoadOption.RIGHT:    maneuver = "Right"
        elif self.current_road_maneuver == RoadOption.STRAIGHT: maneuver = "Straight"
        elif self.current_road_maneuver == RoadOption.VOID:     maneuver = "VOID"
        else:                                                   maneuver = "INVALID(%i)" % self.current_road_maneuver


        # Add metrics to HUD
        self.extra_info.extend([
            "Reward: % 19.2f" % self.last_reward,
            "Subrewards:",
            'Subrewards:',
                'Speed reward: % 19.2f' %self.speed_reward,
                'Cross track:  % 19.2f' %self.cross_track_error_reward,
                'Heading:      % 19.2f' %self.heading_error_reward,
            "Maneuver:        % 11s"       % maneuver,
            "Routes completed:    % 7.2f"  % self.routes_completed,
            "Distance traveled: % 7d m"    % self.distance_traveled,
            "Center deviance:   % 7.2f m"  % self.distance_from_center
        ])
        self._world.render(self.display)
        
        self.hud_no_rendering.add_info('RL', self.extra_info)
        self.hud_no_rendering.render(self.display)
        self.extra_info = []
        self.render_waypoints(self._world.hero_surface)
         # Render HUD
        #self.hud.render(self.display, extra_info=self.extra_info)
        #self.extra_info = [] # Reset extra info list

         # Render to screen
        pygame.display.flip()




        #self.lidar_sensor.render(display)
        #self.hud.render(display)


        #self.rgb_camera.render(display)
 
    def close(self):
        if self._world is not None:
            self.destroy()
        pygame.quit()

    def destroy(self):
        if self.rgb_camera is not None: 
            sensors = [
                self.rgb_camera.sensor,
                self.collision_sensor.sensor,
                self.lane_invasion_sensor.sensor,
                self.lidar_sensor.sensor]
            for sensor in sensors:
                if sensor is not None:
                    sensor.stop()
                    sensor.destroy()
        if self.player is not None:
            self.player.destroy()

    def _get_observation(self):
        while self.observation_buffer is None:
            pass
        obs = self.observation_buffer.copy()
        self.observation_buffer = None
        return obs
    
    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image
    
    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image


    def _draw_path(self, life_time=60.0, skip=0):
        """
            Draw a connected path from start of route to end.
            Green node = start
            Red node   = point along path
            Blue node  = destination
        """
        for i in range(0, len(self.route_waypoints)-1, skip+1):
            w0 = self.route_waypoints[i][0]
            w1 = self.route_waypoints[i+1][0]
            self._world.debug.draw_line(
                w0.transform.location + carla.Location(z=0.25),
                w1.transform.location + carla.Location(z=0.25),
                thickness=0.1, color=carla.Color(255, 0, 0),
                life_time=life_time, persistent_lines=False)
            self._world.debug.draw_point(
                w0.transform.location + carla.Location(z=0.25), 0.1,
                carla.Color(0, 255, 0) if i == 0 else carla.Color(255, 0, 0),
                life_time, False)
        self._world.debug.draw_point(
            self.route_waypoints[-1][0].transform.location + carla.Location(z=0.25), 0.1,
            carla.Color(0, 0, 255),
            life_time, False)

    def process_control(self, action):

        steer, throttle = [float(a) for a in action]
        steer    = steer * self.action_smoothing + steer * (1.0-self.action_smoothing)
        throttle = throttle * self.action_smoothing + throttle * (1.0-self.action_smoothing)

        self._control.steer = steer
        self._control.throttle = throttle

        self.player.apply_control(self._control)

    def run_step(self, action, max_throttle=1.0, max_brake=1.0,
                 max_steering=1.0):
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint
        at a given target_speed.

            :param target_speed: desired vehicle speed
            :param waypoint: target location encoded as a waypoint
            :return: distance (in meters) to the waypoint
        """
        current_steering, target_speed = [float(a) for a in action]
        current_steering = current_steering*self.steering_ratio
        target_speed = target_speed * self.max_target_speed
        acceleration = self._lon_controller.run_step(target_speed)
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        """
        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1
        """
        if current_steering >= 0:
            steering = min(max_steering, current_steering)
        else:
            steering = max(-max_steering, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        self.player.apply_control(control)

    def close(self):
        pygame.quit()
        if self._world is not None:
            self.destroy()
        self.closed = True

    def compute_average_heading_error(self, route_waypoints):
        vehicle_current_yaw = math.radians(self.player.get_transform().rotation.yaw)

        for _ in range(len(route_waypoints[self.current_waypoint_index:])):
            if len(route_waypoints[self.current_waypoint_index:]) >= 4:
                self.current_waypoints, _ = zip(*self.route_waypoints[self.current_waypoint_index:self.current_waypoint_index+4])
                summ = math.radians(sum(item.transform.rotation.yaw for item in self.current_waypoints))
                average_heading_error = summ/4 - vehicle_current_yaw
            else:
                self.current_waypoints, _ = zip(*self.route_waypoints[self.current_waypoint_index:])
                summ = math.radians(sum(item.transform.rotation.yaw for item in self.current_waypoints))
                average_heading_error = summ/len(route_waypoints[self.current_waypoint_index:]) - vehicle_current_yaw
        
        if average_heading_error > np.pi:
            average_heading_error -= 2 * np.pi
        if average_heading_error < - np.pi:
            average_heading_error += 2 * np.pi

        
        return average_heading_error




if __name__ == '__main__':
    env = CarlaEnv()
    action = np.zeros(env.action_space.shape[0])

    while True:
        state = env.reset()
        while True:
            # Process key inputs
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[K_LEFT] or keys[K_a]:
                action[0] = -0.5
            elif keys[K_RIGHT] or keys[K_d]:
                action[0] = 0.5
            else:
                action[0] = 0.0
            action[0] = np.clip(action[0], -1, 1)
            action[1] = 1.0 if keys[K_UP] or keys[K_w] else 0.0
            state, reward, done, info = env.step(action)
            env.render()
            if done:
                env.destroy()
                break
    env.close()

    try:
        env.step()
    finally:
        env.close()
    
    


