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


class PolynomialAgent(object):

    def __init__(self, vehicle, target_speed=50):
        self.status = 'STATUS INITIALIZING'
        self.lead_gap = None
        self.follow_gap = None
        self.vehicle = vehicle
        self._world=vehicle.get_world()
        self._map = self._world.get_map()
        self._target_speed = target_speed / 3.6 # [km/h to m/s]
        self.LV = None
        self.FV = None
        
        # Base Parameter
        opt_dict = {}
        opt_dict['target_speed'] = target_speed
        self._dt = 1.0 / 10.0
        self._prev_d, self._cur_d = 0.0, 0.0
        self._args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': self._dt}
        self._args_longitudinal_dict = {'K_P': 0.75, 'K_I': 0.1, 'K_D': 0, 'dt': self._dt}
        self._desired_acceleration = 3.0 # [m/s^2]
        self._max_throt = 0.75
        self._max_brake = 0.3
        self._max_steer = 0.8
        self._offset = 0
        self._pathminlength = 90.0
        self._pathlength = 320.0 # [m]
        self._base_range = 1.0
        self._distance_ratio = 0.5
        self._path = deque(maxlen=10000)
        self._min_waypoint_queue_length = 100
        self._stop_waypoint_creation = False
        self._debug = self._world.debug
        self._destination = None
        self._safetygap = 1.0
        self._start_emergency = None
        self._ego_wp = None
        self._unitlength = 3.0
        self._speed_diff_criteria = 30.0
        self._dist_diff_criteria = 40.0
        

        # Path Parameter
        self._max_road_width = 4.0
        self._lateral_no = 3
        self._MIN_T = 3.0
        self._MAX_T = 5.0
        self._speed_range = 50.0 / 3.6 # [m/s]
        self._longitudinal_no = 3
        self._max_accel = 10.0 # [m/s^2]
        self._max_curvature = 3.0 # [1/m]
        merging_ref_path_wp = self.getforwardwaypoints(self._pathlength, self.vehicle.get_location())
        target_ref_path_wp = self.getforwardwaypoints(self._pathlength, carla.Location(x=self.vehicle.get_location().x-4.0, y=self.vehicle.get_location().y, z=self.vehicle.get_location().z))
        self.merging_csp = self.waypoints_to_csp(merging_ref_path_wp)
        self.target_csp = self.waypoints_to_csp(target_ref_path_wp)
        
        """
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
        self._K_J = 0.1
        self._K_T = 5.0
        self._K_D1 = 1.0
        self._K_D2 = 1.0
        self._K_LAT = 1.0
        self._K_LON = 1.0

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


    def _init_controller(self):
        self._vehicle_controller = VehiclePIDController(self.vehicle,
                                                        args_lateral=self._args_lateral_dict,
                                                        args_longitudinal=self._args_longitudinal_dict,
                                                        offset=self._offset,
                                                        max_throttle=self._max_throt,
                                                        max_brake=self._max_brake,
                                                        max_steering=self._max_steer)


    def run_step(self, debug = False):
        if self._arrived :
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = self._max_brake
            control.hand_brake = False
            return control
        self._isdebug = debug
        
        vehicle_location = self.vehicle.get_location()

        if self.FV != None:
            speed_diff = get_speed(self.FV)-get_speed(self.vehicle)
            if 0.95*self._speed_diff_criteria < speed_diff < 1.05*self._speed_diff_criteria and 0.95*self._dist_diff_criteria < self.follow_gap < 1.05*self._dist_diff_criteria:
                self.status = 'GAP APPROACH'

        
        #################################################

        vehicle_speed = get_speed(self.vehicle) # [Km/h]
        vehicle_speed_mps = vehicle_speed / 3.6
        vehicle_acceleration = get_acceleration(self.vehicle)
        vehicle_acceleration_mps = vehicle_acceleration / 3.6
        print('vehicle speed :',round(vehicle_speed_mps,3),'[m/s]', round(vehicle_speed_mps*3.6,3),'[km/h]')
        print('vehicle acceleration :',vehicle_acceleration_mps,'[m/s^2]')

        self._ego_wp = self._map.get_waypoint(vehicle_location)
        if not self._ego_wp:
            print('No ego waypoint')

        # 근처에 있는 wp 지우기
        num_remove_wp = 0
        for i in range(len(self._path)):
            if self._path[i][0].distance(vehicle_location) < self._base_range+vehicle_speed_mps*self._dt:
                num_remove_wp+=1
            else:
                break
        for _ in range(num_remove_wp):
            self._path.popleft()
        for i in range(len(self._path)):
            self._debug.draw_point(self._path[i][0],0.1,carla.Color(0,255,0,100),0.1)
        print('Number of Removed point:',num_remove_wp)

        # path generation ##############################################################


        if self._risk_assessment() or not self._path or time.time() - self.path_generation_time > 1.5:
            self.path_generation_time = time.time()
            # USING current vehicle status #################################################
            # Path Initialize
            if self._path:
                path_frenet = self._frenet_optimal_planning(self.merging_csp, *self._path[0][2])
            else: # 첫 생성
                path_frenet = self._frenet_optimal_planning(self.merging_csp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self._path = deque(maxlen=10000)
            if not path_frenet:
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.steer = 0.0
                control.brake = self._max_brake
                control.hand_brake = False
                return control
            for i in range(len(path_frenet.x)):
                concised_frenet = [path_frenet.s[i],path_frenet.s_d[i],path_frenet.s_dd[i],path_frenet.d[i],path_frenet.d_d[i],path_frenet.d_dd[i]]
                self._path.append([carla.Location(x=path_frenet.x[i],y=path_frenet.y[i],z=0.3), path_frenet.ds[i]/self._dt*3.6, concised_frenet])
            # _path[i] = [waypoint, 속도, frenet frame coordinate={'si_d','si_dd','di','di_d','di_dd'}]
    



        next_wp = carla.Transform(self._path[0][0])

        #################################################### Throttle, Steering Controller
        next_speed = self._path[0][1] #[km/h]

        print('next speed:', next_speed / 3.6 , next_speed)
        control = self._vehicle_controller.run_step(next_speed, next_wp)

        print('Throttle',control.throttle)
        if control.brake > 0.0 :
            self._debug.draw_box(carla.BoundingBox(carla.Location(x=vehicle_location.x,y=vehicle_location.y,z=self.vehicle.bounding_box.location.z),self.vehicle.bounding_box.extent),self.vehicle.get_transform().rotation,color = carla.Color(255,0,0,0),life_time=0.1)
            print('Brake', control.brake)
        else:
            self._debug.draw_box(carla.BoundingBox(carla.Location(x=vehicle_location.x,y=vehicle_location.y,z=self.vehicle.bounding_box.location.z),self.vehicle.bounding_box.extent),self.vehicle.get_transform().rotation,color = carla.Color(0,255,0,1),life_time=0.1)

        print('',end='\n\n')
        if vehicle_location.distance(self._destination) < 1.0:
            print("ARRIVE!")
            self._arrived = True
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = self._max_brake
            control.hand_brake = False

            return control

        return control

    def get_lead_follow_vehicles(self, r = 100.0):

        SVs = self._surrounded_vehicles(r)
        self.LV, self.FV = None, None
        lead_gap_cand = float('inf')
        follow_gap_cand = float('inf')
        
        for sv in SVs:
            lateral_diff = self.vehicle.get_location().x-sv.get_location().x
            longitudinal_diff = self.vehicle.get_location().y-sv.get_location().y
            if 0 < lateral_diff < self._max_road_width: # 옆 차선에 있을 때
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

        s_d = x_d*math.cos(tan_angle) + y_d*math.sin(tan_angle)
        s_dd = x_dd*math.cos(tan_angle) + y_dd*math.sin(tan_angle)

        d = -(self.vehicle.get_location().x-ref_x)*math.cos(norm_angle)-(self.vehicle.get_location().y-ref_y)*math.sin(norm_angle)
        
        d_d = -x_d*math.cos(norm_angle) - y_d*math.sin(norm_angle)
        d_dd = - x_dd*math.cos(norm_angle) - y_dd*math.sin(norm_angle)
        
        return s_d, s_dd, d, d_d,d_dd



    def _frenet_optimal_planning(self, ref_path, si, si_d, si_dd, di, di_d, di_dd):
        clock1 = time.time()
        fplist = self._calc_frenet_paths(si, si_d, si_dd, di, di_d, di_dd)
        clock2 = time.time()
        fplist = self._calc_global_paths(fplist, ref_path)
        clock3 = time.time()
        fplist = self._check_paths(fplist)
        clock4 = time.time()
        print(clock2-clock1, clock3-clock2, clock4-clock3)
        # print(len(fplist))

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
        # print(best_path.t[-1], best_path.s_d)
###########################
        return best_path



    def _calc_frenet_paths(self, si, si_d, si_dd, di, di_d, di_dd):
        frenet_paths = []
        # generate path to each offset
        LEFT = self._max_road_width
        RIGHT = -self._max_road_width
        print('Lane_change:',self._ego_wp.lane_change)
        if str(self._ego_wp.lane_change)=='Both':
            pass
        elif str(self._ego_wp.lane_change)=='Left':
            RIGHT = -0.1
        elif str(self._ego_wp.lane_change)=='Right':
            LEFT = 0.1
        else:
            LEFT = 0.1
            RIGHT = -0.1
        
        # Lateral segment
        for df in np.linspace(RIGHT, LEFT, self._lateral_no):
            # Time segment
            for Ti in np.linspace(self._MIN_T, self._MAX_T, 3):
                fp = FrenetPath()
                lat_qp = QuinticPolynomial(di, di_d, di_dd, df, 0.0, 0.0, Ti)

                fp.t = [t for t in np.arange(0.0, Ti, self._dt)]
                fp.d = [lat_qp.calc_point(t) for t in fp.t]
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                # Longitudinal motion planning (Velocity keeping)

                # target speed 바꾸기!
                UPPER_SPEED = si_d + Ti*self._desired_acceleration
                LOWER_SPEED = si_d - Ti*self._desired_acceleration
                if LOWER_SPEED < self._target_speed < UPPER_SPEED:
                    n_level = (UPPER_SPEED-self._target_speed)//((2*Ti*self._desired_acceleration)/(self._longitudinal_no+1))
                    UPPER_SPEED = self._target_speed + n_level*(2*Ti*self._desired_acceleration)/(self._longitudinal_no+1)
                    LOWER_SPEED = self._target_speed - (self._longitudinal_no-n_level-1)*(2*Ti*self._desired_acceleration)/(self._longitudinal_no+1)
                for tv in np.linspace(max(LOWER_SPEED,0.1),
                                    UPPER_SPEED, self._longitudinal_no):
                    tfp = copy.deepcopy(fp)
                    lon_qp = QuarticPolynomial(si, si_d, si_dd, tv, 0.0, Ti)

                    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                    Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                    Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                    # square of diff from target speed
                    ds = (self._target_speed - tfp.s_d[-1]) ** 2

                    # aligning_cost = sum(np.power(tfp.d,2))

                    tfp.cd = self._K_J * Jp + self._K_T * Ti + self._K_D1 * (tfp.d[-1] ** 2)
                    tfp.cv = self._K_J * Js + self._K_T * Ti + self._K_D2 * ds
                    tfp.cf = self._K_LAT * tfp.cd + self._K_LON * tfp.cv

                    frenet_paths.append(tfp)
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

            fp.yaw.append(fp.yaw[-1])
            fp.ds.append(fp.ds[-1])

            # calc curvature
            for i in range(len(fp.yaw) - 1):
                fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])
        return fplist



    def _check_paths(self, fplist):
        ok_ind = []
        reason = [0,0,0,0]
        MAX_SPEED = self._target_speed + self._speed_range
        for i, _ in enumerate(fplist): # Each trajectory
            violated = False
            if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
                violated = True
                reason[0]+=1
            elif any([abs(a) > self._max_accel for a in fplist[i].s_dd]):  # Max accel check
                violated = True
                reason[1]+=1
            elif any([abs(c) > self._max_curvature for c in fplist[i].c]):  # Max curvature check
                violated = True
                reason[2]+=1
            elif not self._check_collision(fplist[i]):
                reason[3]+=1
                violated = True
            if not violated :
                ok_ind.append(i)
                for num in range(len(fplist[i].x)):
                    self._debug.draw_point(carla.Location(x=fplist[i].x[num],y=fplist[i].y[num],z=0.3),0.1,carla.Color(0,0,255),1.0)
            else:
                for num in range(len(fplist[i].x)):
                    self._debug.draw_point(carla.Location(x=fplist[i].x[num],y=fplist[i].y[num],z=0.3),0.1,carla.Color(255,0,0),1.0)
        if not ok_ind:
            print('vel, accel, curvature, Collision',reason)
        return [fplist[i] for i in ok_ind]


    def _check_collision(self, fp):
        SVs = self._surrounded_vehicles()
        for sv in SVs:
            for i in range(len(fp.x)):
                pred_sur_location = sv.get_location() + i*self._dt*sv.get_velocity() + 0.5*sv.get_acceleration()*(i*self._dt)**2
                pred_ego_location = carla.Location(x = fp.x[i], y = fp.y[i])
                if pred_ego_location.distance(pred_sur_location) < self._safetygap:
                    # print('Collision detected!!')
                    return False
        return True

    def set_destination(self, end_location):
        self._destination = end_location
        return None

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
        SVs = self._surrounded_vehicles()
        for sv in SVs:
            for i in range(len(self._path)):
                pred_sur_location = sv.get_location() + i*self._dt*sv.get_velocity() + 0.5*sv.get_acceleration()*(i*self._dt)**2
                pred_ego_location = self._path[i][0]
                if pred_ego_location.distance(pred_sur_location) < self._safetygap:
                    print('Collision detected!!!!!!!!!!!!!')
                    return True
        return False