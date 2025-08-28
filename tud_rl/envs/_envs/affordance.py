import carla
import numpy as np
import math
from utils import *
import glob
import os
import sys



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
        
        return abs(yaw_diff)

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
