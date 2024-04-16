import carla
from shapely.geometry import Polygon
import numpy as np
from agents.navigation.controller import VehiclePIDController

from agents.tools.misc import (get_speed, is_within_distance,
                               get_trafficlight_trigger_location,
                               compute_distance)
import math
import weakref

class IDMAgent(object):

    def __init__(self, vehicle, target_speed=100, acceleration_exponent = 4, time_gap = 0.75, jam_distance=1.0, max_accel = 3.0, desired_decel = 2.0, coolness_factor = 0.99): # time_gap = 1.0, jam_distance=1.5
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param target_speed: speed (in Km/h) at which the vehicle will move
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.

        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._debug = self._world.debug

        # Base parameters
        self._destination = None
        self._target_speed = target_speed
        self._base_tlight_threshold = 5.0  # meters
        self._base_vehicle_threshold = 5.0  # meters
        self._speed_ratio = 1
        self._max_throt = 0.8
        self._max_steer = 0.8        
        self._max_brake = 0.3
        self._offset = 0
        self._dt = 1.0 / 10.0
        self._max_road_width = 4.0
        self._vehicle_length = 4.18  # meters
        self._vehicle_width = 1.99  # meters
        self._args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': self._dt}
        self._args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': self._dt}

        # IDM paramters
        self._acceleration_exponent = acceleration_exponent
        self._desired_time_gap = time_gap
        self._jam_distance = jam_distance
        self._max_accel = max_accel
        self._desired_decel = desired_decel
        self._coolness_factor = coolness_factor

        #Sensor
        self.imu_sensor = IMUSensor(self._vehicle)

        self._init_controller()

    def _init_controller(self):
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=self._args_lateral_dict,
                                                        args_longitudinal=self._args_longitudinal_dict,
                                                        offset=self._offset,
                                                        max_throttle=self._max_throt,
                                                        max_brake=self._max_brake,
                                                        max_steering=self._max_steer)

    def set_target_speed(self, speed):
        """
        Changes the target speed of the agent
            :param speed (float): target speed in Km/h
        """
        self._target_speed = speed
        self._local_planner.set_speed(speed)

    def set_destination(self, end_location):
        self._destination = end_location
    


    def run_step(self, debug=False):
        """Execute one step of navigation."""

        # Retrieve all relevant actors
        vehicle_list = self._surrounded_vehicles()
        
        
        cur_wp = self._map.get_waypoint(self._vehicle.get_location())

        # Check for possible vehicle obstacles
        # max_vehicle_distance = self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
        lead_vehicle = self.get_lead_vehicle()
        vehicle_speed = self._vehicle.get_velocity().length() # vehicle_speed = [m/s]
        if lead_vehicle:
            self.draw_bb(lead_vehicle)
        if lead_vehicle:
            delta_velocity = vehicle_speed - lead_vehicle.get_velocity().length()
            s = self._gap_btw_two(lead_vehicle, self._vehicle)
            s_star = self._jam_distance + vehicle_speed*self._desired_time_gap + vehicle_speed*delta_velocity/(2*math.sqrt(self._max_accel*self._desired_decel))
            a_IDM = self._max_accel*(1 - (vehicle_speed/(self._target_speed/3.6))**self._acceleration_exponent - (s_star/s)**2)
        else:
            a_IDM = self._max_accel*(1 - (vehicle_speed/(self._target_speed/3.6))**self._acceleration_exponent)
        a_IDM = max(a_IDM, -self._desired_decel)
        v_IDM = max(vehicle_speed + 2*a_IDM,0) # vehicle_speed + 2*self._dt*a_IDM
        # jerk_IDM = a_IDM - self.imu_sensor.accelerometer[0]
        next_wp = cur_wp.next(max(1.0, self._dt*v_IDM*2))[0]
        # control = self._vehicle_controller.run_step(75.9, next_wp)
        control = self._vehicle_controller.run_step(3.6* v_IDM, next_wp)
        if 3.6*v_IDM < 0.1:
            control.throttle = 0
            control.brake = self._max_brake

        print(round(vehicle_speed*3.6,3), round(v_IDM*3.6,3), round(a_IDM,3), round(self.imu_sensor.accelerometer[0],3), round(control.brake,3))
        # print(control)
        return control

    def _surrounded_vehicles(self, radius = 50.0):
        sur_vehicles = []
        vehicles = self._world.get_actors().filter('vehicle.*')
        for veh in vehicles:
            if veh.id != self._vehicle.id:
                veh_to_ego = veh.get_location().distance(self._vehicle.get_location())
                if veh_to_ego < radius:
                    sur_vehicles.append(veh)
        return sur_vehicles
    
    def get_lead_vehicle(self):
        # detect only forward or right vehicle
        base_length = 50.0
        SVs = self._surrounded_vehicles(radius = 200.0)
        cur_location = self._vehicle.get_location()
        cur_wp = self._map.get_waypoint(cur_location)
        lead_vehicle = None
        cur_speed = self._vehicle.get_velocity().length()
        for SV in SVs:
            cand_location = SV.get_location()
            cand_speed = SV.get_velocity().length()
            gap = self._gap_btw_two(SV, self._vehicle)
            if (gap < 0) or (cur_location.distance(cand_location) > max(3 * cur_speed,base_length)): # 차량의 앞에 없거나 특정 거리 안에 없으면 continue
                continue
            cand_wp = self._map.get_waypoint(cand_location)
            if (cur_wp.lane_id == cand_wp.lane_id) : # 같은 차선일 떄
                pass
            elif (SV.get_light_state() == carla.VehicleLightState.LeftBlinker) and (0 < cand_location.x-cur_location.x < self._max_road_width): # 오른쪽 차선에 깜빡이 상태로 있을 때
                necessary_gap = ((cur_speed-cand_speed)**2)/(2*self._desired_decel)
                # print('speed:',cur_speed,cand_speed,'gap:',round(gap,3), 'need gap:',round(necessary_gap,3), 'Acceleration[m/s^s]:',round(self.imu_sensor.accelerometer[0],3))
                if (cand_location.x-cur_location.x > self._vehicle_width) and (gap < necessary_gap): # 부딪히진 않으나 Gap이 충분하지 않으면
                    continue
            else:
                continue
            if not lead_vehicle or lead_vehicle.get_location().y < cand_location.y:
                lead_vehicle = SV
        return lead_vehicle

    def _gap_btw_two(self, leading_veh, following_veh):
        return (following_veh.get_location().y-following_veh.bounding_box.location.x-following_veh.bounding_box.extent.x) - (leading_veh.get_location().y-leading_veh.bounding_box.location.x+leading_veh.bounding_box.extent.x)

    def draw_bb(self, vehicle):
        vehicle_location = vehicle.get_transform().location
        self._debug.draw_box(carla.BoundingBox(carla.Location(x=vehicle_location.x,y=vehicle_location.y,z=vehicle.bounding_box.location.z), vehicle.bounding_box.extent),vehicle.get_transform().rotation,thickness=0.3,color = carla.Color(255,0,0,0),life_time=0.1)

class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)