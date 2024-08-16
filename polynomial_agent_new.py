# import carla
import sys
import os
from shapely.geometry import Polygon
import glob
from collections import deque
from time import sleep
import time
import numpy as np
import copy
from pprint import pprint
import weakref
import pickle
import json

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/carla')
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/PythonRobotics/PathPlanning')
except IndexError:
    pass
try:
    sys.path.append(glob.glob('./carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from agents.tools.misc import *
from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner

from agents.navigation.controller import VehiclePIDController

from CubicSpline import cubic_spline_planner
import statsmodels.api as sm
import pandas as pd


class QuarticPolynomial:

    # position_start, velocity_start, accel_start, velocity_end, accel_end
    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt



class QuinticPolynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt


class FrenetPath:

    def __init__(self):

        # Frenet Frame
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        # Glabol Frame
        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []
    def to_dict(self):
        return {
            "t": self.t, "d": self.d, "d_d": self.d_d, "d_dd": self.d_dd, "d_ddd": self.d_ddd,
            "s": self.s, "s_d": self.s_d, "s_dd": self.s_dd, "s_ddd": self.s_ddd,
            "cd": self.cd, "cv": self.cv, "cf": self.cf,
            "x": self.x, "y": self.y, "yaw": self.yaw, "ds": self.ds, "c": self.c
        }        

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


class PolynomialAgent(object):

    def __init__(self, vehicle, target_speed=50):
        self.status = 'EXPERIMENT PREPARING'
        self.status_previous = 'EXPERIMENT PREPARING'
        self.status_changed = False
        self.start = False
        self.lead_gap = None
        self.follow_gap = None
        self.vehicle = vehicle
        self._world=vehicle.get_world()
        self._map = self._world.get_map()
        self._target_speed = target_speed / 3.6 # [km/h to m/s]
        self.LV = None
        self.FV = None
        self._lights = carla.VehicleLightState.NONE
        self.SVs = self._surrounded_vehicles()
        self.vehicle_speed = 0.0
        self.blinking_time = None
        self.DATA_TO_SAVE = []
        self.DATA_COUNT = 0
        self.vehicle_length = 2*self.vehicle.bounding_box.extent.x
        self.vehicle_width = 2*self.vehicle.bounding_box.extent.y
        # x,y 좌표로 저장하고 싶으면 _ACTUAL_PATH와 _PLANNED_PATH로 저장
        self._ACTUAL_PATH = {'t':[],'x':[],'y':[]}
        self._PLANNED_PATH = {'t':[],'x':[],'y':[]}
        self._PLANNED_TIME = 0.0


        # Base Parameter
        opt_dict = {}
        opt_dict['target_speed'] = target_speed
        self._dt = 1.0 / 10.0
        self._prev_d, self._cur_d = 0.0, 0.0
        self._args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': self._dt}
        self._args_longitudinal_dict = {'K_P': 0.75, 'K_I': 0.1, 'K_D': 0, 'dt': self._dt}
        self._desired_acceleration = 3.0 # [m/s^2]
        self._max_throt = 0.75
        self._max_brake = 0.75
        self._max_steer = 0.8
        self._offset = 0
        self._pathminlength = 90.0
        self._pathlength = 215.0 # [m]
        self._base_range = 1.0
        self._distance_ratio = 0.5
        
        self._min_waypoint_queue_length = 100
        self._stop_waypoint_creation = False
        self._debug = self._world.debug
        self._destination = None
        self._safetygap = 1
        self._start_emergency = None
        self._ego_wp = None
        self._unitlength = 3.0

        self.TTC = None
        self.P_interaction = None
        self.P_interaction_previous = None
        self._speed_diff_criteria = 15
        self._dist_diff_criteria = 15
        
        # Path Parameter
        self._max_road_width = 3.5
        self._lateral_no = 3
        self._MIN_T = 3.0
        self._MAX_T = 5.0
        self._speed_range = 50.0 / 3.6 # [m/s]
        self._longitudinal_no = 5
        self._max_accel = 10.0 # [m/s^2]
        self._max_curvature = 0.2 # [1/m]
        self.init_agent()
        """
         # Reference point 표시
        s = np.arange(0, self.merging_csp.s[-1], 1)
        for i_s in s:
            ix, iy = self.merging_csp.calc_position(i_s)
            self._debug.draw_point(carla.Location(x=ix,y=iy,z=0.3),0.1,carla.Color(0,0,0),5)
            if i_s !=0:
                self._debug.draw_line(carla.Location(x=ix,y=iy,z=0.3),carla.Location(x=pix,y=piy,z=0.3),0.1,carla.Color(0,0,0),5)
            pix, piy = ix, iy
        s = np.arange(0, self.target_csp.s[-1], 1)
        for i_s in s:
            ix, iy = self.target_csp.calc_position(i_s)
            self._debug.draw_point(carla.Location(x=ix,y=iy,z=0.3),0.1,carla.Color(0,0,0),5)
            if i_s !=0:
                self._debug.draw_line(carla.Location(x=ix,y=iy,z=0.3),carla.Location(x=pix,y=piy,z=0.3),0.1,carla.Color(0,0,0),5)
            pix, piy = ix, iy
        """

        # Path Cost weights
        self._K_J = 0.01
        self._K_T = 0.05
        self._K_Lateral_Target = 0.05
        self._K_Logitudinal_Target = 0.3
        self._K_LAT = 0.3
        self._K_LON = 0.3
        self._K_INTER = 1.0
        self._target_offset = 0.0

        # Frenet Coordinate
        self._si=0.0
        self._si_d=0.0
        self._si_dd=0.0
        self._di=0.0
        self._di_d=0.0
        self._di_dd=0.0
        # _path[i] : [waypoint, 속도[km/h], frenet frame coordinate=['si','si_d','si_dd','di','di_d','di_dd']]
        

        # Initialize controller
        self._init_controller()
        self._arrived = False
        
        # Sensor
        self.imu_sensor = IMUSensor(self.vehicle)

        # Load interaction probability model
        with open('logit_model_new.pkl','rb') as file:
        # with open('logit_model.pkl','rb') as file:
            self._interaction_func = pickle.load(file)


    def _init_controller(self):
        self._vehicle_controller = VehiclePIDController(self.vehicle,
                                                        args_lateral=self._args_lateral_dict,
                                                        args_longitudinal=self._args_longitudinal_dict,
                                                        offset=self._offset,
                                                        max_throttle=self._max_throt,
                                                        max_brake=0.3,
                                                        max_steering=self._max_steer)
    
    def init_agent(self):
        self._start_time = time.time()
        self._path = deque(maxlen=10000)
        self._path.append([self.vehicle.get_location(), 0.0,[0.0, self.vehicle.get_velocity().length(), 0.0, 0.0, 0.0, 0.0]])
        merging_ref_path_wp = self.getforwardwaypoints(self._pathlength, self.vehicle.get_location())
        target_ref_path_wp = self.getforwardwaypoints(self._pathlength+100.0, carla.Location(x=self.vehicle.get_location().x-4.0, y=self.vehicle.get_location().y, z=self.vehicle.get_location().z))
        self.merging_csp = self.waypoints_to_csp(merging_ref_path_wp)
        self.target_csp = self.waypoints_to_csp(target_ref_path_wp)
        self.cur_csp = self.merging_csp
        
        print('Init velocity[m/s] :',self.vehicle.get_velocity().length())


    def run_step(self, debug = False):
        self._is_debug = debug
        # ----------------------------------------------------------------------- Vehicle State
        vehicle_location = self.vehicle.get_location()
        self.vehicle_speed = self.vehicle.get_velocity().length() # [m/s]
        vehicle_speed_kph = self.vehicle_speed*3.6 # [km/h]
        vehicle_acceleration = get_acceleration(self.vehicle)  # [km/(h*s)]
        vehicle_acceleration_mps = vehicle_acceleration / 3.6 # [m/s^2]
        self.SVs = self._surrounded_vehicles()

        # ----------------------------------------------------------------------- Path Recording
        self._ACTUAL_PATH['t'].append(time.time()-self._start_time)
        self._ACTUAL_PATH['x'].append(vehicle_location.x)
        self._ACTUAL_PATH['y'].append(vehicle_location.y)

        if self.status != 'DONE' and vehicle_location.x < 13.3:
            self.status = 'DONE'
            self.status_changed = True
            self.cur_csp = self.target_csp
            self._path = deque([self._path[0]],maxlen=10000)
            self._path[0][2][3] -= self._max_road_width

        # ----------------------------------------------------------------------- When arrived
        if self._arrived or vehicle_location.distance(self._destination) < 1.0 :
            if not self._arrived:
                self._arrived = True
                print("ARRIVE!")
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = self._max_brake
            control.hand_brake = False
            return control
   
        
        # ----------------------------------------------------------------------- Calculate interaction probability
        if self.FV != None:
            if self.P_interaction!= None:
                self.P_interaction_previous = self.P_interaction
            speed_diff = get_speed(self.FV)-vehicle_speed_kph
            remained_distance = vehicle_location.y + 169 - self.vehicle_length/2
            # X = pd.DataFrame({'const':[1],'gap_btw_MV':[self.follow_gap],'speed_diff':[speed_diff]})
            X = pd.DataFrame({'const':[1],
                              'gap_btw_MV':[self.follow_gap],
                              'speed_diff':[speed_diff], 
                              'lateral_offset_MV':[(remained_distance + self.vehicle_length/2)*np.sin(0.25*np.pi/180) + 14.75 - vehicle_location.x],
                              'remained_distance':[remained_distance],
                              'LeftBlinker_MV':[1 if self.status in ['PREPARING', 'NEGOTIATING', 'LANE CHANGING'] else 0]})
            self.P_interaction = self._interaction_func.predict(X)[0]
            if self.status == 'EXPERIMENT PREPARING' and (0.90*self._dist_diff_criteria < self.follow_gap < 1.1*self._dist_diff_criteria):
                self.start = True
        else:
            self.P_interaction = None
            self.P_interaction_previous = None
            self.TTC = None
        # ----------------------------------------------------------------------- Define Status [EXPERIMENT PREPARING, PREPARING, NEGOTIATING, LANE CHANGING, DONE]
        if self.start and self.status != 'DONE':
            if self.P_interaction == None or self.P_interaction_previous == None:
                self.status = 'LANE CHANGING'
            elif self.P_interaction > 0.7:
                self.status = 'LANE CHANGING'
            elif (self.P_interaction > self.P_interaction_previous and self.P_interaction > 0.1) or (0.4 < self.P_interaction):
                self.status = 'NEGOTIATING'
            else:
                self.status = 'PREPARING'
        if self.status != self.status_previous:
            self.status_changed = True
            self.status_previous = self.status
        else:
            self.status_changed = False
        
        # ----------------------------------------------------------------------- Light State
        if self.status in ['PREPARING', 'NEGOTIATING', 'LANE CHANGING']:
            self._lights = carla.VehicleLightState.LeftBlinker
            if self.blinking_time is None:
                self.blinking_time = time.time()
            if time.time()-self.blinking_time<0.5:
                left_blink_location = self.vehicle.get_location()
                left_blink_location.y+=self.vehicle.bounding_box.extent.x
                left_blink_location.x-=(self.vehicle.bounding_box.extent.y-0.2)
                left_blink_location.z+=0.7
                self._debug.draw_point(left_blink_location,0.2,carla.Color(255,255,0,255),0.1)
            elif time.time()-self.blinking_time>1.0:
                self.blinking_time = time.time()
        else:
            self.blinking_time = None
            self._lights = carla.VehicleLightState.NONE
        self.vehicle.set_light_state(self._lights)

        # ----------------------------------------------------------------------- Remove passed waypoint
        num_remove_wp = 0
        for i in range(len(self._path)):
            if self._path[i][0].distance(vehicle_location) < self._base_range+self.vehicle_speed*self._dt:
                num_remove_wp+=1
            else:
                break
        for _ in range(num_remove_wp):
            if len(self._path)<=1:
                break
            temp = self._path.popleft()
            self._PLANNED_PATH['t'].append(self._PLANNED_TIME)
            self._PLANNED_PATH['x'].append(temp[0].x)
            self._PLANNED_PATH['y'].append(temp[0].y)
            self._PLANNED_TIME+=0.1
        if self._is_debug:            
            for i in range(len(self._path)):
                self._debug.draw_point(self._path[i][0],0.1,carla.Color(0,255,0,100),0.1)
        # print('Number of Removed point:',num_remove_wp)

        # ----------------------------------------------------------------------- Path Generation
        if self._risk_assessment() or (len(self._path)<=1) or ((time.time() - self.path_generation_time) > 1.0) or self.status_changed :
            self.path_generation_time = time.time()
            # Path Initialize
            if self._path:
                path_frenet = self._frenet_optimal_planning(self.cur_csp, *self._path[0][2])
                print('Replanning')
            else:
                print('NO PATH')
            if not path_frenet:
                print('NO AVAILABLE PATH')
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.steer = 0.0
                control.brake = self._max_brake
                control.hand_brake = False
                return control
            self._path = deque(maxlen=10000)
            for i in range(len(path_frenet.x)):
                concised_frenet = [path_frenet.s[i], path_frenet.s_d[i], path_frenet.s_dd[i], path_frenet.d[i], path_frenet.d_d[i] ,path_frenet.d_dd[i]]
                self._path.append([carla.Location(x=path_frenet.x[i],y=path_frenet.y[i],z=0.3), path_frenet.ds[i]/self._dt*3.6, concised_frenet])
            self._PLANNED_TIME = time.time()-self._start_time
        # _path[i] : [waypoint, 속도[km/h], frenet frame coordinate=['si','si_d','si_dd','di','di_d','di_dd']]
        if self.status_changed:
            print('State:', self.status, ', Current Vel:',round(self.vehicle_speed*3.6,2),', Final Vel:',round(self._path[-1][1],4))
        # ------------------------------------------------------------------------ Throttle, Steering Controller

        next_wp = carla.Transform(self._path[0][0])
        next_speed = self._path[0][1] #[km/h]
        control = self._vehicle_controller.run_step(next_speed, next_wp)

        # print('Throttle',control.throttle)
        # print((self._path[1][1]-self._path[0][1])/self._dt/3.6,self.imu_sensor.accelerometer[0])
        """
        # ------------------------------------------------------------------------ Red Box when brake, Green Box when accelerate
        if control.brake > 0.0 :
            self._debug.draw_box(carla.BoundingBox(carla.Location(x=vehicle_location.x,y=vehicle_location.y,z=self.vehicle.bounding_box.location.z),self.vehicle.bounding_box.extent),self.vehicle.get_transform().rotation,color = carla.Color(255,0,0,0),life_time=0.1)
            print('Brake', control.brake)
        else:
            self._debug.draw_box(carla.BoundingBox(carla.Location(x=vehicle_location.x,y=vehicle_location.y,z=self.vehicle.bounding_box.location.z),self.vehicle.bounding_box.extent),self.vehicle.get_transform().rotation,color = carla.Color(0,255,0,1),life_time=0.1)
        """

        return control

    def get_lead_follow_vehicles(self, r = 100.0):

        self.LV, self.FV = None, None
        lead_gap_cand = float('inf')
        follow_gap_cand = float('inf')

        for sv in self.SVs:
            lateral_diff = self.vehicle.get_location().x-sv.get_location().x
            longitudinal_diff = self.vehicle.get_location().y-sv.get_location().y
            if -self._max_road_width/2 < lateral_diff < 1.5*self._max_road_width: # 왼쪽 or 같은 차선에 있을 때
                if longitudinal_diff < 0: # Ego가 앞에 있을 때
                    gap = self._gap_btw_two(self.vehicle,sv)
                    if gap < follow_gap_cand:
                        self.FV = sv
                        follow_gap_cand = gap
                else: # Ego가 뒤에 있을 때
                    gap = self._gap_btw_two(sv, self.vehicle)
                    if gap < lead_gap_cand:
                        self.LV = sv
                        lead_gap_cand = gap
        if lead_gap_cand != float('inf'):
            self.lead_gap = lead_gap_cand
        else:
            self.lead_gap = None
        if follow_gap_cand != float('inf'):
            self.follow_gap = follow_gap_cand
        else:
            self.follow_gap = None
        return
    
    def getforwardwaypoints(self, pathlength, location):
        waypoints = []
        start_wp = self._map.get_waypoint(location)
        waypoints.append(start_wp)
        for i in range(int(self._unitlength),int(pathlength),int(self._unitlength)):
            waypoints.append(start_wp.next(i)[0])
        return waypoints
    
    def get_waypoints_until_end(self, location):
        start_wp = self._map.get_waypoint(location)
        return start_wp.next_until_lane_end(self._unitlength)
    
    def waypoints_to_csp(self, wps):
        wpx, wpy = [], []
        for wp in wps:
            wpx.append(wp.transform.location.x)
            wpy.append(wp.transform.location.y)
        return cubic_spline_planner.CubicSpline2D(wpx, wpy)

    def _gap_btw_two(self, leading_veh, following_veh):
        return (following_veh.get_location().y-following_veh.bounding_box.location.x-following_veh.bounding_box.extent.x) - (leading_veh.get_location().y-leading_veh.bounding_box.location.x+leading_veh.bounding_box.extent.x)

    def _get_vehicle_status(self, ref_path):

        curvature = ref_path.calc_curvature(0)
        tan_angle = ref_path.calc_yaw(0)
        norm_angle =  tan_angle+math.pi/2
        x_d, y_d = self.vehicle.get_velocity().x, self.vehicle.get_velocity().y
        x_dd, y_dd = self.vehicle.get_acceleration().x, self.vehicle.get_acceleration().y
        ref_x, ref_y = ref_path.calc_position(0)

        s_d = x_d*math.cos(tan_angle) + y_d * math.sin(tan_angle)
        s_dd = x_dd*math.cos(tan_angle) + y_dd * math.sin(tan_angle)

        d = -(self.vehicle.get_location().x-ref_x)*math.cos(norm_angle)-(self.vehicle.get_location().y-ref_y)*math.sin(norm_angle)
        
        d_d = -x_d*math.cos(norm_angle) - y_d*math.sin(norm_angle)
        d_dd = - x_dd*math.cos(norm_angle) - y_dd*math.sin(norm_angle)
        
        return s_d, s_dd, d, d_d,d_dd


# ===============================================================
# -- Polynomial trajectory planning -----------------------------
# ===============================================================
    def _frenet_optimal_planning(self, ref_path, si, si_d, si_dd, di, di_d, di_dd):
        clock1 = time.time()
        fplist = self._calc_frenet_paths(si, si_d, si_dd, di, di_d, di_dd) # Cost function
        clock2 = time.time()
        fplist = self._calc_global_paths(fplist, ref_path)
        clock3 = time.time()
        fplist = self._check_paths(fplist) # Cost function
        clock4 = time.time()
        fplist = self._interaction_cost(fplist)
        clock5 = time.time()
        # print(round(clock2-clock1,4), round(clock3-clock2,4), round(clock4-clock3,4), round(clock5-clock4,4))


        # find minimum cost path
        min_cost = float("inf")
        best_path = None
        for fp in fplist :
            if min_cost >= fp.cf:
                min_cost = fp.cf
                best_path = fp
        if not best_path:
            return None

        ######## SAVE
        # self._SAVING_PATH_func([best_path], 'SELECTED')
        self.DATA_COUNT+=1

        return best_path

    def _calc_frenet_paths(self, si, si_d, si_dd, di, di_d, di_dd):
        frenet_paths = []
        # generate path to each offset
        RIGHT = 0
        LEFT = 0
        # STATE 정의
        if self.status in ['EXPERIMENT PREPARING','PREPARING', 'DONE']:
            self._lateral_no = 1
            self._MIN_T = 1.5
            self._MAX_T = 3.0
            RIGHT = 0
            LEFT = 0
            self._target_offset = 0.0
            self._K_Logitudinal_Target = 0.3

        elif self.status == 'NEGOTIATING':
            self._MIN_T = 1.5
            self._MAX_T = 3.0
            self._lateral_no = 3
            RIGHT = self._max_road_width/8
            LEFT = self._max_road_width*2/9
            self._target_offset = self._max_road_width*2/9
            self._K_Logitudinal_Target = 0.02

        elif self.status == 'LANE CHANGING':
            self._MIN_T = 3
            self._MAX_T = 5
            self._lateral_no = 3
            RIGHT = (self._max_road_width*2)/3
            LEFT = self._max_road_width
            self._target_offset = self._max_road_width
            self._K_Logitudinal_Target = 0.02
        
        # Time segment
        for Ti in np.linspace(self._MIN_T, self._MAX_T, 3):
            fp_lateral_list = []
            fp_time = FrenetPath()
            fp_time.t = [t for t in np.arange(0.0, Ti, self._dt)] 
            # Lateral segment
            for df in np.linspace(RIGHT, LEFT, self._lateral_no):
                fp = FrenetPath()
                lat_qp = QuinticPolynomial(di, di_d, di_dd, df, 0.0, 0.0, Ti)
                fp.d = [lat_qp.calc_point(t) for t in fp_time.t]
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp_time.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp_time.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp_time.t]
                Jp = sum(np.power(fp.d_ddd, 2))
                fp.cd = self._K_J * Jp + self._K_T * Ti + self._K_Lateral_Target * ((fp.d[-1]-self._target_offset) ** 2)
                # aligning_cost = sum(np.power(tfp.d,2))
                fp_lateral_list.append(fp)

            # Longitudinal (Velocity keeping)
            UPPER_SPEED = si_d + Ti*self._desired_acceleration
            LOWER_SPEED = si_d - Ti*self._desired_acceleration

            if LOWER_SPEED < self._target_speed < UPPER_SPEED:
                UPPER_SPEED = self._target_speed + Ti*self._desired_acceleration
                LOWER_SPEED = self._target_speed - Ti*self._desired_acceleration
            fp_longitudinal_list = []
            for tv in np.linspace(max(LOWER_SPEED,0.1), UPPER_SPEED, self._longitudinal_no):
                fp = FrenetPath()
                lon_qp = QuarticPolynomial(si, si_d, si_dd, tv, 0.0, Ti)

                fp.s = [lon_qp.calc_point(t) for t in fp_time.t]
                fp.s_d = [lon_qp.calc_first_derivative(t) for t in fp_time.t]
                fp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp_time.t]
                fp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp_time.t]
                fp_longitudinal_list.append(fp)

                Js = sum(np.power(fp.s_ddd, 2))  # square of jerk
                # square of diff from target speed
                ds = (self._target_speed - fp.s_d[-1]) ** 2
                fp.cv = self._K_J * Js + self._K_T * Ti + self._K_Logitudinal_Target * ds

            for fp_lateral in fp_lateral_list:
                for fp_longitudinal in fp_longitudinal_list:
                    fp_new = FrenetPath()
                    fp_new = copy.deepcopy(fp_lateral)
                    fp_new.t = fp_time.t
                    fp_new.s = fp_longitudinal.s
                    fp_new.s_d = fp_longitudinal.s_d
                    fp_new.s_dd = fp_longitudinal.s_dd
                    fp_new.s_ddd = fp_longitudinal.s_ddd
                    fp_new.cv = fp_longitudinal.cv
                    fp_new.cf = self._K_LAT * fp_lateral.cd + self._K_LON * fp_new.cv

                    frenet_paths.append(fp_new)
        # print(len(frenet_paths))
        return frenet_paths

    def _calc_global_paths(self, fplist, ref_path):
        for fp in fplist:
            # calc global positions
            for i in range(len(fp.s)):
                ix, iy = ref_path.calc_position(fp.s[i])
                if ix is None:
                    # print(i, fp.s)
                    break
                i_yaw = ref_path.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix - di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy - di * math.sin(i_yaw + math.pi / 2.0)
                fp.x.append(fx)
                fp.y.append(fy)
            # print('s:',fp.s[-1], 'd:',fp.d[-1])
            # calc yaw and ds
            for i in range(len(fp.x) - 1):
                dx = fp.x[i + 1] - fp.x[i]
                dy = fp.y[i + 1] - fp.y[i]
                fp.yaw.append(math.atan2(dy, dx))
                fp.ds.append(math.hypot(dx, dy))
            if fp.yaw:
                fp.yaw.append(fp.yaw[-1])
                fp.ds.append(fp.ds[-1])
            else:
                fp.yaw.append(0.0)
                fp.ds.append(0.0)
            # calc curvature
            for i in range(len(fp.yaw) - 1):
                fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])
        return fplist

    def _check_paths(self, fplist):
        result_paths = []
        SAVING_PATHS = []
        reason = [0,0,0,0]
        MAX_SPEED = self._target_speed + self._speed_range
        for i, PATH in enumerate(fplist): # Each trajectory
            violated = False
            if max(PATH.s_d) > MAX_SPEED: # any([v > MAX_SPEED for v in PATH.s_d]):  # Max speed check
                violated = True
                reason[0]+=1
            elif any([abs(a) > self._max_accel for a in PATH.s_dd]):  # Max accel check
                violated = True
                reason[1]+=1
            elif any([abs(c) > self._max_curvature for c in PATH.c]):  # Max curvature check
                violated = True
                reason[2]+=1
            elif not self._check_collision(PATH):
                reason[3]+=1
                violated = True
            if not violated :
                result_paths.append(PATH)
                color = carla.Color(0,0,255)
            ####### SAVE
            else:
                SAVING_PATHS.append(PATH)
                color = carla.Color(255,0,0)
            if self._is_debug:
                for num in range(len(PATH.x)):
                    self._debug.draw_point(carla.Location(x=PATH.x[num],y=PATH.y[num],z=0.3),0.05, color, 1.0)
        # if not result_paths:
        # print('vel, accel, curvature, Collision',reason)
        
        ####### SAVE
        # self._SAVING_PATH_func(SAVING_PATHS, 'IMPOSSIBLE')
        return result_paths

    def _check_collision(self, fp):
        if self._guard_collision(fp):
            return False
        if self.status == 'NEGOTIATING' and max(fp.d)>1.2:
            print('Nego crossed the line')
            return False
        
        if self.LV is not None:
            safe_lead_gap = max(0,self._dt*self.vehicle_speed + (self.vehicle_speed**2)/(2*self._desired_acceleration) - (self.LV.get_velocity().length()**2)/(2*self._desired_acceleration))
            
            max_y_coordinate = self.LV.get_location().y + self.LV.get_velocity().y*fp.t[-1] + safe_lead_gap + (self.LV.bounding_box.extent.x*2)
            if (fp.y[-1] < max_y_coordinate) and (abs(self.LV.get_location().x-fp.x[-1]) < 2*self.LV.bounding_box.extent.y):
                # print(self.LV.get_location().y,'+', self.LV.get_velocity().y*fp.t[-1], '+',safe_lead_gap,'+',(self.LV.bounding_box.extent.x*2))
                # print('LV collision detected!!', round(fp.y[-1],3), round(max_y_coordinate,3))
                return False
        if self.FV is not None: # Human reaction time : 0.75 sec
            estimated_FV_velocity = max(0,self.FV.get_velocity().length()-self._desired_acceleration*fp.t[-1])
            # safe_follow_gap = max(0,0.75*self.FV.get_velocity().length() + (self.FV.get_velocity().length()**2)/(2*self._desired_acceleration) - (self.vehicle_speed**2)/(2*self._desired_acceleration))
            safe_follow_gap = max(0,0.75*estimated_FV_velocity + (estimated_FV_velocity**2)/(2*self._desired_acceleration) - (self.vehicle_speed**2)/(2*self._desired_acceleration))
            # min_y_coordinate = self.FV.get_location().y + self.FV.get_velocity().y*fp.t[-1] - safe_follow_gap - (self.vehicle_length)
            min_y_coordinate = self.FV.get_location().y + self.FV.get_velocity().y*fp.t[-1] + 0.5*self._desired_acceleration*(fp.t[-1]**2) - safe_follow_gap - (self.FV.bounding_box.extent.x*2)
            if (fp.y[-1] > min_y_coordinate) and (abs(self.FV.get_location().x-fp.x[-1]) < 2*self.FV.bounding_box.extent.y):
                # print(self.FV.get_location().y,'+', self.FV.get_velocity().y*fp.t[-1], '+',0.5*self._desired_acceleration*(fp.t[-1]**2),'-',safe_follow_gap,'-',(self.FV.bounding_box.extent.x*2))
                # print('FV collision detected!!', round(fp.y[-1],3), round(min_y_coordinate,3))
                return False
        return True
    
    def _guard_collision(self,fp):
        for i in range(len(fp.x)):
            if (fp.y[i]<-169) and (fp.x[i]>13.0):
                print('Guard collision!!')
                return True
        return False

    def _interaction_cost(self, fplist):
        if self.FV == None or not self.status in ['NEGOTIATING','LANE CHANGING']: # only when lane changing and negotiating
            ####### SAVE
            # self._SAVING_PATH_func(fplist, 'POSSIBLE')
            return fplist
        V_FV = get_speed(self.FV)
        # Need to be normalized
        for fp in fplist:
            # X_dict = {'const':[],
            # 'gap_btw_MV':[],
            # 'speed_diff':[], 
            # 'lateral_offset_MV':[],
            # 'remained_distance':[],
            # 'LeftBlinker_MV':[]}
            # for i in range(len(fp.x)):
            #     speed_diff = V_FV - fp.ds[i]/self._dt*3.6
            #     predict_y_FV = self.FV.get_location().y + i*self._dt*self.FV.get_velocity().y
            #     remained_distance = fp.y[i] + 169 - self.vehicle_length/2
            #     follow_gap = predict_y_FV - self.FV.bounding_box.location.x - self.FV.bounding_box.extent.x - (fp.y[i]- self.vehicle.bounding_box.location.x - self.vehicle.bounding_box.extent.x)
            #     X_dict['const'].append(1)
            #     X_dict['gap_btw_MV'].append(follow_gap)
            #     X_dict['speed_diff'].append(speed_diff)
            #     X_dict['lateral_offset_MV'].append((remained_distance + self.vehicle_length/2)*np.sin(0.25*np.pi/180) + 14.75 - fp.x[i])
            #     X_dict['remained_distance'].append(remained_distance)
            #     X_dict['LeftBlinker_MV'].append(1)
            predict_y_FV = self.FV.get_location().y + len(fp.x)*self._dt*self.FV.get_velocity().y
            follow_gap = predict_y_FV - self.FV.bounding_box.location.x - self.FV.bounding_box.extent.x - (fp.y[-1]- self.vehicle.bounding_box.location.x - self.vehicle.bounding_box.extent.x)
            speed_diff = V_FV - fp.ds[-1]/self._dt*3.6
            remained_distance = fp.y[-1] + 169 - self.vehicle_length/2
            if remained_distance <0:
                remained_distance = 0
            X_dict = {'const':[1],
            'gap_btw_MV':[follow_gap],
            'speed_diff':[speed_diff], 
            'lateral_offset_MV':[(remained_distance + self.vehicle_length/2)*np.sin(0.25*np.pi/180) + 14.75 - fp.x[-1]],
            'remained_distance':[remained_distance],
            'LeftBlinker_MV':[1]}
            X = pd.DataFrame(X_dict)
            # predicted_values = self._interaction_func.predict(X)
            # interaction_cost = 1 - np.mean(predicted_values)
            interaction_cost = 1 - self._interaction_func.predict(X)[0]
            # print(X)
            # print('Path cost:',1 - interaction_cost)
            # print(round(self._K_INTER*interaction_cost,3), round(fp.ds[-1]/self._dt*3.6,3),'km/h, Cost:',fp.cf)
            fp.cf += self._K_INTER*interaction_cost
        
        ####### SAVE
        # self._SAVING_PATH_func(fplist, 'POSSIBLE')
        return fplist
    

    def _SAVING_PATH_func(self, fplist, available):
        # if self.status in self.DONE_SAVE_STATUS:
        #     return
        # print('SAVING****************', self.status)
        # self.DONE_SAVE_STATUS.append(self.status)
        if self.status == 'EXPERIMENT PREPARING':
            return
        for path in fplist:
            temp_dict = path.to_dict()
            temp_dict['STATUS'] = self.status
            temp_dict['AVAILABLE'] = available
            temp_dict['ORDER'] = self.DATA_COUNT
            self.DATA_TO_SAVE.append(temp_dict)
        return

# ===================================================================================================================================

    def set_destination(self, end_location):
        self._destination = end_location
        return None
    
    def set_distance_criteria(self, distance):
        self._dist_diff_criteria = distance
        return 

    def _surrounded_vehicles(self, radius = 50.0):
        sur_vehicles = []
        vehicles = self._world.get_actors().filter('vehicle.*')
        for veh in vehicles:
            if veh.id != self.vehicle.id:
                veh_to_ego = veh.get_location().distance(self.vehicle.get_location())
                if veh_to_ego < radius:
                    sur_vehicles.append(veh)
        return sur_vehicles
    
    def _risk_assessment(self):
        
        if self._path is None:
            return False
        """
        for sv in self.SVs:
            for i in range(min(len(self._path), 10)):
                pred_sur_location = sv.get_location() + i*self._dt*sv.get_velocity()
                pred_ego_location = self._path[i][0]
                if abs(pred_sur_location.x-pred_ego_location.x) < self.vehicle_width + 0.2 and abs(pred_sur_location.y-pred_ego_location.y) < self.vehicle_length+self._safetygap:
                    print('COLLISION DETECTED!')
                    return True
        """
        if self.LV is not None:
            pred_LV_location = self.LV.get_location() + 1*self.LV.get_velocity()
            pred_ego_location = self.vehicle.get_location() + 1*self.vehicle.get_velocity()
            if abs(pred_LV_location.x-pred_ego_location.x) < self.vehicle_width and abs(pred_LV_location.y-pred_ego_location.y) < self.vehicle_length:
                return True
        if self.FV is not None:
            pred_FV_location = self.FV.get_location() + 1*self.FV.get_velocity()
            pred_ego_location = self.vehicle.get_location() + 1*self.vehicle.get_velocity()
            if abs(pred_FV_location.x-pred_ego_location.x) < self.vehicle_width and abs(pred_FV_location.y-pred_ego_location.y) < self.vehicle_length:
                return True
        return False



class PolynomialAgentBaseLine(object):

    def __init__(self, vehicle, target_speed=50):
        self.status = 'EXPERIMENT PREPARING'
        self.status_previous = 'EXPERIMENT PREPARING'
        self.status_changed = False
        self.start = False
        self.lead_gap = None
        self.follow_gap = None
        self.vehicle = vehicle
        self._world=vehicle.get_world()
        self._map = self._world.get_map()
        self._target_speed = target_speed / 3.6 # [km/h to m/s]
        self.LV = None
        self.FV = None
        self._lights = carla.VehicleLightState.NONE
        self.SVs = self._surrounded_vehicles()
        self.blinking_time = None
        self.vehicle_length = 2*self.vehicle.bounding_box.extent.x
        self.vehicle_width = 2*self.vehicle.bounding_box.extent.y
        self.best_y = None
        self.DATA_TO_SAVE = []
        self.DATA_COUNT = 0
        self._ACTUAL_PATH ={}
        self._PLANNED_PATH ={}

        # Base Parameter
        opt_dict = {}
        opt_dict['target_speed'] = target_speed
        self._dt = 1.0 / 10.0
        self._prev_d, self._cur_d = 0.0, 0.0
        self._args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': self._dt}
        self._args_longitudinal_dict = {'K_P': 0.75, 'K_I': 0.1, 'K_D': 0, 'dt': self._dt}
        self._desired_acceleration = 3.0 # [m/s^2]
        self._max_throt = 0.75
        self._max_brake = 0.75
        self._max_steer = 0.8
        self._offset = 0
        self._pathminlength = 90.0
        self._pathlength = 215.0 # [m]
        self._base_range = 1.0
        self._distance_ratio = 0.5
        self._min_waypoint_queue_length = 100
        self._stop_waypoint_creation = False
        self._debug = self._world.debug
        self._destination = None
        self._safetygap = 1
        self._start_emergency = None
        self._ego_wp = None
        self._unitlength = 3.0

        self.TTC = None
        self.P_interaction = None
        self.P_interaction_previous = None
        self._speed_diff_criteria = 15
        self._dist_diff_criteria = 15
        
        # Path Parameter
        self._max_road_width = 3.5
        self._lateral_no = 3
        self._MIN_T = 3.0
        self._MAX_T = 5.0
        self._speed_range = 50.0 / 3.6 # [m/s]
        self._longitudinal_no = 5
        self._max_accel = 10.0 # [m/s^2]
        self._max_curvature = 1.0 # [1/m]
        self.init_agent()

        # Path Cost weights
        self._K_J = 0.01
        self._K_T = 0.05
        self._K_Lateral_Target = 0.3
        self._K_Logitudinal_Target = 0.02
        # self._K_Logitudinal_Target = 0.3
        self._K_LAT = 0.3
        self._K_LON = 0.3
        self._K_INTER = 0
        self._target_offset = 0.0

        # Frenet Coordinate
        self._si=0.0
        self._si_d=0.0
        self._si_dd=0.0
        self._di=0.0
        self._di_d=0.0
        self._di_dd=0.0

        # Initialize controller
        self._init_controller()
        self._arrived = False
        
        # Sensor
        self.imu_sensor = IMUSensor(self.vehicle)


    def _init_controller(self):
        self._vehicle_controller = VehiclePIDController(self.vehicle,
                                                        args_lateral=self._args_lateral_dict,
                                                        args_longitudinal=self._args_longitudinal_dict,
                                                        offset=self._offset,
                                                        max_throttle=self._max_throt,
                                                        max_brake=0.3,
                                                        max_steering=self._max_steer)
        
    def init_agent(self):
        self._path = deque(maxlen=10000)
        merging_ref_path_wp = self.getforwardwaypoints(self._pathlength, self.vehicle.get_location())
        target_ref_path_wp = self.getforwardwaypoints(self._pathlength+100.0, carla.Location(x=self.vehicle.get_location().x-4.0, y=self.vehicle.get_location().y, z=self.vehicle.get_location().z))
        self.merging_csp = self.waypoints_to_csp(merging_ref_path_wp)
        self.target_csp = self.waypoints_to_csp(target_ref_path_wp)
        self.cur_csp = self.merging_csp
        self._path.append([self.vehicle.get_location(), 0.0,[0.0, self.vehicle.get_velocity().length(), 0.0, 0.0, 0.0, 0.0]])


    def run_step(self, debug = False):
        self._is_debug = debug
        # ----------------------------------------------------------------------- Vehicle State
        vehicle_location = self.vehicle.get_location()
        self.vehicle_speed = self.vehicle.get_velocity().length() # [m/s]
        vehicle_speed_kph = self.vehicle_speed*3.6 # [km/h]
        vehicle_acceleration = get_acceleration(self.vehicle)  # [km/(h*s)]
        vehicle_acceleration_mps = vehicle_acceleration / 3.6 # [m/s^2]
        self.SVs = self._surrounded_vehicles()

        if self.status != 'DONE' and vehicle_location.x < 13.3:
            self.status = 'DONE'
            self.status_changed = True
            self.cur_csp = self.target_csp
            self._path = deque([self._path[0]],maxlen=10000)
            self._path[0][2][3] -= self._max_road_width

        # ----------------------------------------------------------------------- When arrived
        if self._arrived or vehicle_location.distance(self._destination) < 1.0 :
            if not self._arrived:
                self._arrived = True
                print("ARRIVE!")
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = self._max_brake
            control.hand_brake = False
            return control
        
        # ----------------------------------------------------------------------- Define Status [EXPERIMENT PREPARING, LANE CHANGING, DONE]
        if self.FV != None and self.status == 'EXPERIMENT PREPARING' and (0.90*self._dist_diff_criteria < self.follow_gap < 1.1*self._dist_diff_criteria): 
            self.start = True
        if self.start and self.status != 'DONE':
            self.status = 'LANE CHANGING'
        if self.status != self.status_previous: # When status changed
            self.status_changed = True
            self.status_previous = self.status
        else:
            self.status_changed = False

        # ----------------------------------------------------------------------- Light State
        if self.status in ['PREPARING', 'NEGOTIATING', 'LANE CHANGING']:
            self._lights = carla.VehicleLightState.LeftBlinker
            if self.blinking_time is None:
                self.blinking_time = time.time()
            if time.time()-self.blinking_time<0.5:
                left_blink_location = self.vehicle.get_location()
                left_blink_location.y+=self.vehicle.bounding_box.extent.x
                left_blink_location.x-=(self.vehicle.bounding_box.extent.y-0.2)
                left_blink_location.z+=0.7
                self._debug.draw_point(left_blink_location,0.2,carla.Color(255,255,0,10),0.1)
            elif time.time()-self.blinking_time>1.0:
                self.blinking_time = time.time()
        else:
            self.blinking_time = None
            self._lights = carla.VehicleLightState.NONE
        self.vehicle.set_light_state(self._lights)


        # ----------------------------------------------------------------------- Remove Passed Waypoint
        num_remove_wp = 0
        for i in range(len(self._path)):
            if self._path[i][0].distance(vehicle_location) < self._base_range + self.vehicle_speed*self._dt:
                num_remove_wp+=1
            else:
                break
        for _ in range(num_remove_wp):
            if len(self._path)<=1:
                break
            self._path.popleft()
        if self._is_debug:
            for i in range(len(self._path)):
                self._debug.draw_point(self._path[i][0],0.1,carla.Color(0,255,0,100),0.1)

        # print('Number of Removed point:',num_remove_wp)

        # ----------------------------------------------------------------------- Path Generation 
        if self._risk_assessment() or (len(self._path)<=1) or ((time.time() - self.path_generation_time) > 1.0) or self.status_changed :
            self.path_generation_time = time.time()
            # Path Initialize
            if self._path:
                path_frenet = self._frenet_optimal_planning(self.cur_csp, *self._path[0][2])
            else:
                print('NO PATH')
            if not path_frenet:
                print('NO AVAILABLE PATH')
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.steer = 0.0
                control.brake = self._max_brake
                control.hand_brake = False
                return control
            self._path = deque(maxlen=10000)
            for i in range(len(path_frenet.x)):
                try:
                    concised_frenet = [path_frenet.s[i], path_frenet.s_d[i], path_frenet.s_dd[i], path_frenet.d[i], path_frenet.d_d[i] ,path_frenet.d_dd[i]]
                    self._path.append([carla.Location(x=path_frenet.x[i],y=path_frenet.y[i],z=0.3), path_frenet.ds[i]/self._dt*3.6, concised_frenet])
                except IndexError:
                    print(len(path_frenet.x),len(path_frenet.y),len(path_frenet.ds))
        # _path[i] : [waypoint, 속도[km/h], frenet frame coordinate=['si','si_d','si_dd','di','di_d','di_dd']]
        if self.status_changed:
            print('State:', self.status, ', Current Vel:',round(self.vehicle_speed*3.6,2),', Final Vel:',round(self._path[-1][1],4))
        # ------------------------------------------------------------------------ Throttle, Steering Controller
        next_wp = carla.Transform(self._path[0][0])
        next_speed = self._path[0][1] #[km/h]
        control = self._vehicle_controller.run_step(next_speed, next_wp)

        return control

    def get_lead_follow_vehicles(self, r = 100.0):

        self.LV, self.FV = None, None
        lead_gap_cand = float('inf')
        follow_gap_cand = float('inf')

        for sv in self.SVs:
            lateral_diff = self.vehicle.get_location().x-sv.get_location().x
            longitudinal_diff = self.vehicle.get_location().y-sv.get_location().y
            if -self._max_road_width/2 < lateral_diff < 1.5*self._max_road_width: # 왼쪽 or 같은 차선에 있을 때
                if longitudinal_diff < 0: # Ego가 앞에 있을 때
                    gap = self._gap_btw_two(self.vehicle,sv)
                    if gap < follow_gap_cand:
                        self.FV = sv
                        follow_gap_cand = gap
                else: # Ego가 뒤에 있을 때
                    gap = self._gap_btw_two(sv, self.vehicle)
                    if gap < lead_gap_cand:
                        self.LV = sv
                        lead_gap_cand = gap
        if lead_gap_cand != float('inf'):
            self.lead_gap = lead_gap_cand
        else:
            self.lead_gap = None
        if follow_gap_cand != float('inf'):
            self.follow_gap = follow_gap_cand
        else:
            self.follow_gap = None
        return
    
    def getforwardwaypoints(self, pathlength, location):
        waypoints = []
        start_wp = self._map.get_waypoint(location)
        waypoints.append(start_wp)
        for i in range(int(self._unitlength),int(pathlength),int(self._unitlength)):
            waypoints.append(start_wp.next(i)[0])
        return waypoints
    
    def get_waypoints_until_end(self, location):
        start_wp = self._map.get_waypoint(location)
        return start_wp.next_until_lane_end(self._unitlength)
    
    def waypoints_to_csp(self, wps):
        wpx, wpy = [], []
        for wp in wps:
            wpx.append(wp.transform.location.x)
            wpy.append(wp.transform.location.y)
        return cubic_spline_planner.CubicSpline2D(wpx, wpy)

    def _gap_btw_two(self, leading_veh, following_veh):
        return (following_veh.get_location().y-following_veh.bounding_box.location.x-following_veh.bounding_box.extent.x) - (leading_veh.get_location().y-leading_veh.bounding_box.location.x+leading_veh.bounding_box.extent.x)


# ===============================================================
# -- Polynomial trajectory planning -----------------------------
# ===============================================================
    def _frenet_optimal_planning(self, ref_path, si, si_d, si_dd, di, di_d, di_dd):
        clock1 = time.time()
        fplist = self._calc_frenet_paths(si, si_d, si_dd, di, di_d, di_dd) # Cost function
        clock2 = time.time()
        fplist = self._calc_global_paths(fplist, ref_path)
        clock3 = time.time()
        fplist = self._check_paths(fplist) # Cost function
        clock4 = time.time()
        # print(round(clock2-clock1,4), round(clock3-clock2,4), round(clock4-clock3,4))
        
        # find minimum cost path
        min_cost = float("inf")
        best_path = None
        for fp in fplist :
            if min_cost >= fp.cf:
                min_cost = fp.cf
                best_path = fp
        ###########################
        # print(len(best_path.x),len(best_path.t))
        if not best_path:
            return None
        
        ######## SAVE
        # self._SAVING_PATH_func([best_path], 'SELECTED')
        self.DATA_COUNT+=1
        ###########################
        return best_path

    def _calc_frenet_paths(self, si, si_d, si_dd, di, di_d, di_dd):
        frenet_paths = []
        # generate path to each offset
        RIGHT = 0
        LEFT = 0
        # STATE 정의
        if self.status == 'LANE CHANGING':
            self._lateral_no = 3
            RIGHT = 0
            LEFT = self._max_road_width
            self._target_offset = self._max_road_width
        elif self.status == 'DONE':
            self._lateral_no = 1
            RIGHT = 0
            LEFT = 0
            self._target_offset = 0.0
            self._K_Logitudinal_Target = 0.3
            self._longitudinal_no = 7
        else:
            self._lateral_no = 1
            RIGHT = 0
            LEFT = 0
            self._target_offset = 0.0
        
        # Time segment
        for Ti in np.linspace(self._MIN_T, self._MAX_T, 3):
            fp_lateral_list = []
            fp_time = FrenetPath()
            fp_time.t = [t for t in np.arange(0.0, Ti, self._dt)] 
            # Lateral segment
            for df in np.linspace(RIGHT, LEFT, self._lateral_no):
                fp = FrenetPath()
                lat_qp = QuinticPolynomial(di, di_d, di_dd, df, 0.0, 0.0, Ti)
                fp.d = [lat_qp.calc_point(t) for t in fp_time.t]
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp_time.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp_time.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp_time.t]
                Jp = sum(np.power(fp.d_ddd, 2))
                fp.cd = self._K_J * Jp + self._K_T * Ti + self._K_Lateral_Target * ((fp.d[-1]-self._target_offset) ** 2)
                # aligning_cost = sum(np.power(tfp.d,2))
                fp_lateral_list.append(fp)

            # Longitudinal (Velocity keeping)
            UPPER_SPEED = si_d + Ti*self._desired_acceleration
            LOWER_SPEED = si_d - Ti*self._desired_acceleration
            if LOWER_SPEED < self._target_speed < UPPER_SPEED:
                UPPER_SPEED = self._target_speed + Ti*self._desired_acceleration
                LOWER_SPEED = self._target_speed - Ti*self._desired_acceleration
            fp_longitudinal_list = []
            for tv in np.linspace(max(LOWER_SPEED,0.1), UPPER_SPEED, self._longitudinal_no):
                fp = FrenetPath()
                lon_qp = QuarticPolynomial(si, si_d, si_dd, tv, 0.0, Ti)

                fp.s = [lon_qp.calc_point(t) for t in fp_time.t]
                fp.s_d = [lon_qp.calc_first_derivative(t) for t in fp_time.t]
                fp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp_time.t]
                fp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp_time.t]
                fp_longitudinal_list.append(fp)

                Js = sum(np.power(fp.s_ddd, 2))  # square of jerk
                # square of diff from target speed
                ds = (self._target_speed - fp.s_d[-1]) ** 2
                fp.cv = self._K_J * Js + self._K_T * Ti + self._K_Logitudinal_Target * ds

            for fp_lateral in fp_lateral_list:
                for fp_longitudinal in fp_longitudinal_list:
                    fp_new = FrenetPath()
                    fp_new = copy.deepcopy(fp_lateral)
                    fp_new.t = fp_time.t
                    fp_new.s = fp_longitudinal.s
                    fp_new.s_d = fp_longitudinal.s_d
                    fp_new.s_dd = fp_longitudinal.s_dd
                    fp_new.s_ddd = fp_longitudinal.s_ddd
                    fp_new.cv = fp_longitudinal.cv
                    fp_new.cf = self._K_LAT * fp_lateral.cd + self._K_LON * fp_new.cv

                    frenet_paths.append(fp_new)
        # print(len(frenet_paths))
        return frenet_paths

    def _calc_global_paths(self, fplist, ref_path):
        for fp in fplist:
            # calc global positions
            for i in range(len(fp.s)):
                ix, iy = ref_path.calc_position(fp.s[i])
                if ix is None:
                    # print(i, fp.s)
                    break
                i_yaw = ref_path.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix - di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy - di * math.sin(i_yaw + math.pi / 2.0)
                fp.x.append(fx)
                fp.y.append(fy)
            # calc yaw and ds
            for i in range(len(fp.x) - 1):
                dx = fp.x[i + 1] - fp.x[i]
                dy = fp.y[i + 1] - fp.y[i]
                fp.yaw.append(math.atan2(dy, dx))
                fp.ds.append(math.hypot(dx, dy))
            if fp.yaw:
                fp.yaw.append(fp.yaw[-1])
                fp.ds.append(fp.ds[-1])
            else:
                fp.yaw.append(0.0)
                fp.ds.append(0.0)
            # calc curvature
            for i in range(len(fp.yaw) - 1):
                fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])
        return fplist

    def _check_paths(self, fplist):
        result_paths = []
        reason = [0,0,0,0]
        MAX_SPEED = self._target_speed + self._speed_range
        SAVING_PATHS_possible = []
        SAVING_PATHS_impossible = []
        for i, PATH in enumerate(fplist): # Each trajectory
            violated = False
            if max(PATH.s_d) > MAX_SPEED: # any([v > MAX_SPEED for v in PATH.s_d]):  # Max speed check
                violated = True
                reason[0]+=1
            elif any([abs(a) > self._max_accel for a in PATH.s_dd]):  # Max accel check
                violated = True
                reason[1]+=1
            elif any([abs(c) > self._max_curvature for c in PATH.c]):  # Max curvature check
                violated = True
                reason[2]+=1
            elif not self._check_collision(PATH):
                reason[3]+=1
                violated = True
            if not violated :
                SAVING_PATHS_possible.append(PATH)
                result_paths.append(PATH)
                color = carla.Color(0,0,255)
            ####### SAVE
            else:
                SAVING_PATHS_impossible.append(PATH)
                color = carla.Color(255,0,0)
            if self._is_debug:
                for num in range(len(PATH.x)):
                    self._debug.draw_point(carla.Location(x=PATH.x[num],y=PATH.y[num],z=0.3),0.05, color, 0.5)
        # print('vel, accel, curvature, Collision',reason)
        ####### SAVE
        # self._SAVING_PATH_func(SAVING_PATHS_impossible, 'IMPOSSIBLE')
        # self._SAVING_PATH_func(SAVING_PATHS_possible, 'POSSIBLE')                       
        return result_paths

    def _check_collision(self, fp):
        if self._guard_collision(fp):
            return False
        if self.LV is not None:
            safe_lead_gap = max(self._dt*self.vehicle_speed + (self.vehicle_speed**2)/(2*self._desired_acceleration) - (self.LV.get_velocity().length()**2)/(2*self._desired_acceleration),0)
            max_y_coordinate = self.LV.get_location().y + self.LV.get_velocity().y*fp.t[-1] + safe_lead_gap + (self.vehicle_length)
            if fp.y[-1] < max_y_coordinate and abs(self.LV.get_location().x-fp.x[-1]) < self.vehicle_width:
                # print(self.LV.get_location().y,'+', self.LV.get_velocity().y*fp.t[-1], '+',safe_lead_gap,'+',(self.vehicle_length))
                # print('LV collision detected!!', round(fp.y[-1],3), round(max_y_coordinate,3))
                return False
        if self.FV is not None: # Human reaction time : 0.75 sec
            safe_follow_gap = max(0.75*self.FV.get_velocity().length() + (self.FV.get_velocity().length()**2)/(2*self._desired_acceleration) - (self.vehicle_speed**2)/(2*self._desired_acceleration),0)
            min_y_coordinate = self.FV.get_location().y + self.FV.get_velocity().y*fp.t[-1] - safe_follow_gap - (self.vehicle_length) # - 0.5*self._desired_acceleration*fp.t[-1]**2
            if self.status == "DONE":
                min_y_coordinate = self.FV.get_location().y + self.FV.get_velocity().y*fp.t[-1] + 0.5*self._desired_acceleration*(fp.t[-1]**2) - safe_follow_gap - (self.FV.bounding_box.extent.x*2)
            if fp.y[-1] > min_y_coordinate and abs(self.FV.get_location().x-fp.x[-1]) < self.vehicle_width:
                # print(self.FV.get_location().y,'+', self.FV.get_velocity().y*fp.t[-1], '-',safe_follow_gap,'-',(self.vehicle_length))
                # print('FV collision detected!!', round(fp.y[-1],3), round(min_y_coordinate,3))
                return False
        return True
    
    def _guard_collision(self,fp):
        for i in range(len(fp.x)):
            if (fp.y[i]<-169) and (fp.x[i]>13.0):
                print('Guard collision!!')
                return True
        return False


# ===================================================================================================================================

    def set_destination(self, end_location):
        self._destination = end_location
        return None
    def set_distance_criteria(self, distance):
        self._dist_diff_criteria = distance
        return 
    def _surrounded_vehicles(self, radius = 50.0):
        sur_vehicles = []
        vehicles = self._world.get_actors().filter('vehicle.*')
        for veh in vehicles:
            if veh.id != self.vehicle.id:
                veh_to_ego = veh.get_location().distance(self.vehicle.get_location())
                if veh_to_ego < radius:
                    sur_vehicles.append(veh)
        return sur_vehicles
    
    def _risk_assessment(self):
        """
        if self._path is None:
            return False
        for sv in self.SVs:
            for i in range(min(len(self._path), 10)): # 
                pred_sur_location = sv.get_location() + i*self._dt*sv.get_velocity()
                pred_ego_location = self._path[i][0]
                if abs(pred_sur_location.x-pred_ego_location.x) < self.vehicle_width + 0.2 and abs(pred_sur_location.y-pred_ego_location.y) < self.vehicle_length:
                    print('PATH COLLISION DETECTED!') #,'y difference:',pred_sur_location.y-pred_ego_location.y, sv.attributes['color'], 'LV:', self.LV.attributes['color'])
                    return True
        """
        if self.LV is not None:
            pred_LV_location = self.LV.get_location() + 1*self.LV.get_velocity()
            pred_ego_location = self.vehicle.get_location() + 1*self.vehicle.get_velocity()
            if abs(pred_LV_location.x-pred_ego_location.x) < self.vehicle_width and abs(pred_LV_location.y-pred_ego_location.y) < self.vehicle_length:
                return True
        if self.FV is not None:
            pred_FV_location = self.FV.get_location() + 1*self.FV.get_velocity()
            pred_ego_location = self.vehicle.get_location() + 1*self.vehicle.get_velocity()
            if abs(pred_FV_location.x-pred_ego_location.x) < self.vehicle_width and abs(pred_FV_location.y-pred_ego_location.y) < self.vehicle_length:
                return True
        return False
    
    def _SAVING_PATH_func(self, fplist, available):
        # if self.status in self.DONE_SAVE_STATUS:
        #     return
        # print('SAVING****************', self.status)
        # self.DONE_SAVE_STATUS.append(self.status)
        if self.status == 'EXPERIMENT PREPARING':
            return
        for path in fplist:
            temp_dict = path.to_dict()
            temp_dict['STATUS'] = self.status
            temp_dict['AVAILABLE'] = available
            temp_dict['ORDER'] = self.DATA_COUNT
            self.DATA_TO_SAVE.append(temp_dict)
        return    