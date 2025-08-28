import carla
import cv2
import weakref
import numpy as np
import pygame
import math
from utils import *
import collections
import random

# ==============================================================================
# -- Camera -------------------------------------------------------------
# ==============================================================================

IM_WIDTH = 640
IM_HEIGHT = 480
PORT = 2000
HOST = "localhost"
SHOW_PREVIEW = True

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
        self.array = None
        self.sensor.listen(lambda image: self._parse_image(image))
        

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
            
        

    def _parse_image(self, image):
        self.array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        self.array = np.reshape(self.array, (image.height, image.width, 4))
        self.array = self.array[:, :, :3]
        self.array = self.array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(self.array.swapaxes(0, 1))
        
        self.image = np.array(image.raw_data)
        self.image = self.image.reshape((self.im_height, self.im_width, 4))
        self.image = self.image[:, :, :3]
        return self.array
        


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
        self.impulse = None
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
        self.impulse = event.normal_impulse
        intensity = math.sqrt(self.impulse.x**2 + self.impulse.y**2 + self.impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)
        return event



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



class Vehicle(object):

    def __init__(self, world, vehicle_type = "model3"):
        
        self._world = world
        vehicle_blueprint_library = self._world.get_blueprint_library()
        vehicle_bp = vehicle_blueprint_library.filter(vehicle_type)[0]
        vehicle_transform = random.choice(self._world.get_map().get_spawn_points())

        self.vehicle = self._world.spawn_actor(vehicle_bp, vehicle_transform)

        self.control = carla.VehicleControl()
        self.destroyed = False
    
    def get_speed(self):
        velocity = self.vehicle.get_velocity()
        return  math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def destroy(self):
        if self.destroyed:
            raise Exception("Actor already destroyed.")
        else:
            print("Destroying ", self, "...")
            self.vehicle.destroy()
            self.world.actor_list.remove(self)
            self.destroyed = True



