

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

import random
import re
import weakref
import time
import cv2
import numpy as np
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_f
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


IM_WIDTH = 640
IM_HEIGHT = 480
PORT = 2000
HOST = "localhost"
SHOW_PREVIEW = True

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def distance_to_line(A, B, p):
    num   = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    if np.isclose(denom, 0):
        return np.linalg.norm(p - A)
    return num / denom


def vector(v):
    """ Turn carla Location/Vector3D/Rotation to np.array """
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])


class CarlaEnv(object):

    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client(HOST, PORT)
        self.client.set_timeout(11.0)
        self.world = self.client.get_world()
        self.settings = self.world.get_settings()
        
        self.world.apply_settings(self.settings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self._control = carla.VehicleControl()
        self._is_on_reverse = False

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        SECONDS_PER_EPISODE = 0.1
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys.

        Return None
        if a new episode was requested.
        """

        if keys[K_LEFT] or keys[K_a]:
            self._control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            self._control.steer = 1.0
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = 1.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = 1.0
        if keys[K_SPACE]:
            self._control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        self._control.reverse = self._is_on_reverse

        self.vehicle.apply_control(self._control)
        self.sensor.listen(lambda data: self.process_img(data))
        return self._control        
    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()
        self.vehicle.destroy()

    @staticmethod
    def emergency_stop():
        """
        Send an emergency stop command to the vehicle

            :return: control for braking
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control



# ==============================================================================
# -- Camera -------------------------------------------------------------
# ==============================================================================


class RGBCamera(object):
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.recording = False
        self.image = None
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        self._camera_transforms = carla.Transform(carla.Location(x=2.5, z=1.2))

        self.rgb_cam = bp_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")
        self.sensor = world.spawn_actor(self.rgb_cam, self._camera_transforms, attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: RGBCamera._parse_image(weak_self, image))
        

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
            cv2.imshow("", self.image)
            cv2.waitKey(1)
        

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        
        self.image = np.array(image.raw_data)
        self.image = self.image.reshape((self.im_height, self.im_width, 4))
        self.image = self.image[:, :, :3]
        


# ==============================================================================
# -- Lidar -------------------------------------------------------------
# ==============================================================================


class Lidar(object):

    def __init__(self, parent_actor):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.recording = False
        world = self._parent.get_world()
        self.width = 640
        self.height = 360
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')

        lidar_bp.set_attribute('upper_fov', '30.0')
        lidar_bp.set_attribute('lower_fov', '-25.0')
        lidar_bp.set_attribute('channels', '64.0')
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('points_per_second', '100000')
        self.lidar_range = 100
        self._lidar_transforms = carla.Transform(carla.Location(x = 1.0, z = 1.8))

        self.sensor = world.spawn_actor(lidar_bp, self._lidar_transforms, attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda lidar: Lidar.lidar_callback(weak_self, lidar))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def lidar_callback(weak_self, lidar):
        self = weak_self()
        if not self:
            return
        points = np.frombuffer(lidar.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(self.width, self.height) / (2.0 * self.lidar_range)
        lidar_data += (0.5 * self.width, 0.5 * self.height)
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (self.width, self.height, 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        self.surface = pygame.surfarray.make_surface(lidar_img)
       
        



# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)



# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]



# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):

    def __init__(self, carla_world):
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.rgb_camera = None
        self.lidar_sensor = None
        self.restart()
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self):

        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        # Get the ego vehicle

        vehicle_blueprint_library = self.world.get_blueprint_library()
        model_3 = vehicle_blueprint_library.filter("model3")[0]
        vehicle_transform = random.choice(self.world.get_map().get_spawn_points())
        self.player = self.world.spawn_actor(model_3, vehicle_transform)
        self.player_name = self.player.type_id

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player)
        self.rgb_camera = RGBCamera(self.player)
        self.lidar_sensor = Lidar(self.player)
        self.world.wait_for_tick()

    def tick(self, clock, wait_for_repetitions):
        if len(self.world.get_actors().filter(self.player_name)) < 1:
            if not wait_for_repetitions:
                return False
            else:
                self.player = None
                self.destroy()
                self.restart()

        return True

    def render(self, display):
        #self.lidar_sensor.render(display)
        self.rgb_camera.render(display)

    def destroy_sensors(self):
        self.rgb_camera.sensor.destroy()
        self.camera_manager.sensor = None

    def destroy(self):
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




class KeyboardControl(object):
    
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    """Class that handles keyboard input."""
    def __init__(self, world):
        self._control = carla.VehicleControl()
        self._lights = carla.VehicleLightState.NONE
        self._steer_cache = 0.0
        world.player.set_light_state(self._lights)
        
    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def parse_events(self, client, world, clock):
        """
        Return a VehicleControl message based on the pressed keys.
        """
        current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
        control = self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
        self._control.reverse = self._control.gear < 0
        # Set automatic control-related vehicle lights
        if self._control.brake:
            current_lights |= carla.VehicleLightState.Brake
        else: # Remove the Brake flag
            current_lights &= ~carla.VehicleLightState.Brake
        if self._control.reverse:
            current_lights |= carla.VehicleLightState.Reverse
        else: # Remove the Reverse flag
            current_lights &= ~carla.VehicleLightState.Reverse
        if current_lights != self._lights: # Change the light state only if necessary
            self._lights = current_lights
            world.player.set_light_state(carla.VehicleLightState(self._lights))
        world.player.apply_control(control)



    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.1, 1.00)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]
        return self._control

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


class AffordanceGetter(object):

    # for stop signs
    PROXIMITY_THRESHOLD = 30.0  # meters
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 1.0  # meters

    def __init__(self, carla_world, vehicle):
        self._world = carla_world
        try:
            self._map = self._world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.affordances = {}
        
        self.affordances["traffic_light"] = None
        self._vehicle = vehicle


        self.affordances["stop_sign"] = None
        self._affected_by_stop = (
            False  # if the ego vehicle is influenced by a stop sign
        )
        self._stop_completed = False  # if the ego vehicle has completed the stop sign
        self._target_stop_sign = None  # the stop sign affecting the ego vehicle

        self.affordances["ego_speed"] = None
        self.affordances["ego_steering"] = None

        self.affordances["speed_limit"] = None
    def _is_stop_sign_hazard(self, stop_sign_list):

        res = []
        if self._affected_by_stop:
            if not self._stop_completed:
                current_speed = self._get_forward_speed()
                if current_speed < self.SPEED_THRESHOLD:
                    self._stop_completed = True
                    return res
                else:
                    return [self._target_stop_sign]
            else:
                # reset if the ego vehicle is outside the influence of the current stop sign
                if not self._is_actor_affected_by_stop(
                    self._vehicle, self._target_stop_sign
                ):
                    self._affected_by_stop = False
                    self._stop_completed = False
                    self._target_stop_sign = None
                return res

        ve_tra = self._vehicle.get_transform()
        ve_dir = ve_tra.get_forward_vector()

        wp = self._map.get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in stop_sign_list:
                if self._is_actor_affected_by_stop(self._vehicle, stop_sign):
                    # this stop sign is affecting the vehicle
                    self._affected_by_stop = True
                    self._target_stop_sign = stop_sign
                    res.append(self._target_stop_sign)

        return res

    def _get_forward_speed(self, transform=None, velocity=None):
        """Convert the vehicle transform directly to forward speed"""
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array(
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)]
        )
        speed = np.dot(vel_np, orientation)
        return speed

    def get_speed(self):

        vel = self._vehicle.get_velocity()

        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    

    def get_acc(self):

        acc = self._vehicle.get_acceleration()

        return math.sqrt(acc.x ** 2 + acc.y ** 2 + acc.z ** 2)

    def distance_to_lane_center(self):

        # 1. calculate lateral error
        
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location(), project_to_road=True)
        
        # .next() meaning next waypoint in 0.3m?         
        next_waypints = self._map.get_waypoint(self._vehicle.get_location(), project_to_road=True).next(0.3)
        
        next_waypint = np.random.choice(next_waypints)

        #print(current_waypoint, next_waypint)

        distance_from_center = distance_to_line(vector(current_waypoint.transform.location),
                                                vector(next_waypint.transform.location),
                                                vector(self._vehicle.get_location()))

        # 2. calculate heading error

        heading_error = self.calculate_heading_error(current_waypoint, next_waypint)


        crosstrack_error, yaw_diff_crosstrack = self.calculate_cross_track_error(current_waypoint)


        return distance_from_center, heading_error, crosstrack_error
    

    def calculate_heading_error(self, current_waypoint, next_waypoint):

        #yaw_path = np.arctan2(current_waypoint.x-next_waypoint.x, current_waypoint.y-next_waypoint.y)

        yaw_path = math.radians(current_waypoint.transform.rotation.yaw)

        yaw = math.radians(self._vehicle.get_transform().rotation.yaw)

        yaw_diff = yaw_path - yaw

        if yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        if yaw_diff < - np.pi:
            yaw_diff += 2 * np.pi
        
        return yaw_diff

    def calculate_cross_track_error(self,current_waypoint):

        k_e = 0.3
        k_v = 10

        x = self._vehicle.get_location().x
        y = self._vehicle.get_location().y

        way_point_x = current_waypoint.transform.location.x
        way_point_y = current_waypoint.transform.location.y

        yaw_path = math.radians(current_waypoint.transform.rotation.yaw)

        current_xy = np.array([x, y])
        way_point_xy = np.array([way_point_x, way_point_y])

        crosstrack_error = np.min(np.sum((current_xy - way_point_xy)**2))

        yaw_cross_track = np.arctan2(y-way_point_y, x-way_point_x)

        yaw_path2ct = yaw_path - yaw_cross_track

        if yaw_path2ct > np.pi:
            yaw_path2ct -= 2 * np.pi
        if yaw_path2ct < - np.pi:
            yaw_path2ct += 2 * np.pi
        if yaw_path2ct > 0:
            crosstrack_error = abs(crosstrack_error)
        else:
            crosstrack_error = - abs(crosstrack_error)

        yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / (k_v + (self.get_speed() / 3.6)))

        return crosstrack_error, yaw_diff_crosstrack


    def calcu_c_1(host_waypoint, route_distance):
        previous_waypoint = host_waypoint.previous(route_distance)[0]
        next_waypoint = host_waypoint.next(route_distance)[0]
        _transform = next_waypoint.transform
        _location, _rotation  = _transform.location, _transform.rotation
        x1, y1 = _location.x, _location.y
        yaw1 = _rotation.yaw

        _transform = previous_waypoint.transform
        _location, _rotation  = _transform.location, _transform.rotation
        x2, y2 = _location.x, _location.y
        yaw2 = _rotation.yaw

        c = 2*math.sin(math.radians((yaw1-yaw2)/2)) / math.sqrt((x1-x2)**2 + (y1-y2)**2)
        return c

    



    def _translate_tl_state(self, state):

        if state == carla.TrafficLightState.Red:
            return 0
        elif state == carla.TrafficLightState.Yellow:
            return 1
        elif state == carla.TrafficLightState.Green:
            return 2
        elif state == carla.TrafficLightState.Off:
            return 3
        elif state == carla.TrafficLightState.Unknown:
            return 4
        else:
            return None
    
    def _find_obstacle(self, obstacle_type="*traffic_light*"):
        """Find all actors of a certain type that are close to the vehicle

        Args:
            obstacle_type (str, optional): [description]. Defaults to '*traffic_light*'.

        Returns:
            [type]: [description]
        """
        obst = list()

        _actors = self._world.get_actors()
        _obstacles = _actors.filter(obstacle_type)

        for _obstacle in _obstacles:
            trigger = _obstacle.trigger_volume

            _obstacle.get_transform().transform(trigger.location)
            distance_to_car = trigger.location.distance(self._vehicle.get_location())

            a = np.sqrt(
                trigger.extent.x**2 + trigger.extent.y**2 + trigger.extent.z**2
            )
            b = np.sqrt(
                self._vehicle.bounding_box.extent.x**2
                + self._vehicle.bounding_box.extent.y**2
                + self._vehicle.bounding_box.extent.z**2
            )

            s = a + b + 10

            if distance_to_car <= s:
                # the actor is affected by this obstacle.
                obst.append(_obstacle)

        return obst
    
    def _point_inside_boundingbox(self, point, bb_center, bb_extent):
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

    def _is_actor_affected_by_stop(self, actor, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
            return affected

        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._map.get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                waypoint = waypoint.next(self.WAYPOINT_STEP)[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self._point_inside_boundingbox(
                actor_location, transformed_tv, stop.trigger_volume.extent
            ):
                affected = True

        return affected


    def get_affordance(self):

        self.affordances["traffic_light"] = self._translate_tl_state(
                        self._vehicle.get_traffic_light_state()
                    )
        #stop_signs = self._find_obstacle("*stop*")
        #self.affordances["stop_sign"] = self._is_actor_affected_by_stop(self._vehicle, stop_signs)

        stop_sign = self._is_stop_sign_hazard(self._find_obstacle("*stop*"))
        self.affordances["stop_sign"] = stop_sign
        
        self.affordances["ego_speed"] = self.get_speed() / 3.6

        self.affordances["ego_acceleration"] = self.get_acc()

        self.affordances["speed_limit"] = self._vehicle.get_speed_limit()

        self.affordances["distance_to_current_lane_center"], self.affordances["heading_error"], self.affordances["crosstrack_error"] = self.distance_to_lane_center()


        return self.affordances







def game_loop():
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(HOST, PORT)
        client.set_timeout(20.0)
        sim_world = client.get_world()
        world = World(sim_world)
        controller = KeyboardControl(world)
        sim_world.wait_for_tick()
        affordancegetter = AffordanceGetter(sim_world, world.player)
        display = pygame.display.set_mode(
            (640, 360),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            if not world.tick(clock, False):
                return
            world.render(display)
            final_affordance = affordancegetter.get_affordance()
            print(final_affordance) 
            pygame.display.flip()

    finally:
        if world is not None:
            world.destroy()

        pygame.quit()

if __name__ == '__main__':
    try:
        game_loop()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as error:
        logging.exception(error)
