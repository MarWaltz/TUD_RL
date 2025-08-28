import numpy as np
import carla
import math


def distance_to_lane_center(current_waypoint, next_waypoint, vehicle):

    distance_from_center = distance_to_line(vector(current_waypoint.transform.location),
                                            vector(next_waypoint.transform.location),
                                            vector(vehicle.get_transform().location))

    # 2. calculate heading error

    heading_error = calculate_heading_error(current_waypoint, next_waypoint, vehicle.get_transform())

    # 3. calculate cross track error

    crosstrack_error, yaw_diff_crosstrack = calculate_cross_track_error(current_waypoint, vehicle)

    return distance_from_center, heading_error, crosstrack_error


def lidar_to_histogram_features(lidar, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 14
        y_meters_max = 28
        xbins = np.linspace(-2*x_meters_max, 2*x_meters_max+1, 2*x_meters_max*pixels_per_meter+1)
        ybins = np.linspace(-y_meters_max, 0, y_meters_max*pixels_per_meter+1)
        hist = np.histogramdd(point_cloud[...,:2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[...,2]<=-2.0]
    above = lidar[lidar[...,2]>-2.0]
    below_features = splat_points(below)
    above_features = splat_points(above)
    total_features = below_features + above_features
    features = np.stack([below_features, above_features, total_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features

def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:,2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    # reset z-coordinate
    out[:,2] = xyz[:,2]

    return out

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

def calculate_heading_error(current_waypoint, next_waypoint, current_location):

    #yaw_path = np.arctan2(current_waypoint.x-next_waypoint.x, current_waypoint.y-next_waypoint.y)

    yaw_path = math.radians(current_waypoint.transform.rotation.yaw)

    yaw = math.radians(current_location.rotation.yaw)

    yaw_diff = yaw_path - yaw

    if yaw_diff > np.pi:
        yaw_diff -= 2 * np.pi
    if yaw_diff < - np.pi:
        yaw_diff += 2 * np.pi
    
    return abs(yaw_diff)

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def calculate_cross_track_error(current_waypoint, vehicle):

    k_e = 0.3
    k_v = 10

    

    x = vehicle.get_transform().location.x
    y = vehicle.get_transform().location.y

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

    yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / (k_v + (get_speed(vehicle) / 3.6)))

    return crosstrack_error, yaw_diff_crosstrack

def tl_affecting_vehicle(vehicle):
    state = translate_tl_state(vehicle.get_traffic_light_state())
    max_desired_speed = 6  # desired speed 30km/h
    vehicle_location = vehicle.get_transform().location

    if state == 0:
        affecting_tl = vehicle.get_traffic_light()
        

        #affecting_tl_transform = affecting_tl.get_transform().location
        
        #distance_to_tl = np.linalg.norm(vector(vehicle_location) - vector(affecting_tl_transform))

        distance_to_tl = min(5, cal_distance_to_tl(affecting_tl, vehicle_location))
        ## according to desired speed relation, closer to red light, desired speed linear decrease to 0 

        desired_speed = math.sqrt(distance_to_tl*(max_desired_speed**2)/5)
        
        #print(affecting_tl_transform, vehicle_location, distance_to_tl)
    else:
        desired_speed = max_desired_speed
        distance_to_tl = 10
    return distance_to_tl, desired_speed


def cal_distance_to_tl(affecting_tl, vehicle_location):

    if affecting_tl is None:
        return 10.0

    nearby_waypoints = affecting_tl.get_stop_waypoints()
    nearby_waypoint = (np.random.choice(nearby_waypoints)).transform.location
    affecting_tl_transform = affecting_tl.get_transform().location

    if abs(nearby_waypoint.x - affecting_tl_transform.x) >  abs(nearby_waypoint.y - affecting_tl_transform.y):
        distance_to_tl = min(10, abs(affecting_tl_transform.x - vehicle_location.x))
    else:
        distance_to_tl = min(10, abs(affecting_tl_transform.y - vehicle_location.y))
    
    return max(0, (distance_to_tl - 5))



def translate_tl_state(state):

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