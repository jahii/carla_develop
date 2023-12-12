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
        self._vehicle = vehicle
        self._world=vehicle.get_world()
        self._map = self._world.get_map()
        self._target_speed = target_speed / 3.6 # [m/s]
        
        # Base Parameter
        opt_dict = {}
        opt_dict['target_speed'] = target_speed
        self._dt = 1.0 / 10.0
        self._prev_d, self._cur_d = 0.0, 0.0
        self._args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': self._dt}
        self._args_longitudinal_dict = {'K_P': 0.75, 'K_I': 0.1, 'K_D': 0, 'dt': self._dt}
        self._desired_acceleration = 5.0 # [m/s^2]
        self._max_throt = 0.75
        self._max_brake = 0.3
        self._max_steer = 0.8
        self._offset = 0
        self._pathminlength = 90.0
        self._pathlength = 100.0 # [m]
        self._base_range = 3.0
        self._distance_ratio = 0.5
        self._waypoints_queue = deque(maxlen=10000)
        self._min_waypoint_queue_length = 100
        self._stop_waypoint_creation = False
        self._debug = self._world.debug
        self._destination = None
        self._safetygap = 5.0
        self._start_emergency = None
        self._ego_wp = None

        # Path Parameter
        self._max_road_width = 3.0
        self._lateral_no = 3
        self._MIN_T = 1.0
        self._MAX_T = 3.0
        self._speed_range = 50.0 / 3.6 # [m/s]
        self._longitudinal_no = 5
        self._max_accel = 15.0 # [m/s^2]
        self._max_curvature = 5.0 # [1/m]

        # Path Cost weights
        self._K_J = 0.1
        self._K_T = 2.0
        self._K_D = 3.0
        self._K_LAT = 1.0
        self._K_LON = 1.0

        # Frenet Coordinate
        self._si_d=0.0
        self._si_dd=0.0
        self._di=0.0
        self._di_d=0.0
        self._di_dd=0.0

        # Initialize controller
        self._init_controller()
        self._arrived = False

    def _init_controller(self):
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=self._args_lateral_dict,
                                                        args_longitudinal=self._args_longitudinal_dict,
                                                        offset=self._offset,
                                                        max_throttle=self._max_throt,
                                                        max_brake=self._max_brake,
                                                        max_steering=self._max_steer)
        # self.cur_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # self._waypoints_queue.append((self.cur_waypoint))


    def run_step(self, debug = False):

        if self._arrived :
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = self._max_brake
            control.hand_brake = False
            return control
        self._isdebug = debug
        
        vehicle_location = self._vehicle.get_location()


        #################################################
        start= time.time()



        vehicle_speed = get_speed(self._vehicle) # [Km/h]
        vehicle_speed_mps = vehicle_speed / 3.6
        vehicle_acceleration = get_acceleration(self._vehicle)
        vehicle_acceleration_mps = vehicle_acceleration / 3.6
        waypoint_distance = self._base_range + vehicle_speed*self._distance_ratio
        self._ego_wp = self._map.get_waypoint(vehicle_location)
        waypoint_distance = max(min(self._pathminlength,waypoint_distance),self._pathlength)

        cur_wps = self.getforwardwaypoints(waypoint_distance,vehicle_speed_mps)
        
        wpx, wpy = [], []
        for wp in cur_wps:
            wpx.append(wp.transform.location.x)
            wpy.append(wp.transform.location.y)

        # Generate Reference Path
        csp = cubic_spline_planner.CubicSpline2D(wpx, wpy)

        s = np.arange(0, csp.s[-1], 1)


        ################################################## VEHICLE STATUS
        si_d, si_dd, di, di_d, di_dd = self._get_vehicle_status(csp)
        
        if si_dd > self._max_accel:
            self._max_accel = math.ceil(si_dd)


        rx, ry, ryaw, rk = [], [], [], []
        
        for i_s in s:
            ix, iy = csp.calc_position(i_s) 
            rx.append(ix)
            ry.append(iy)
            ryaw.append(csp.calc_yaw(i_s))
            rk.append(csp.calc_curvature(i_s))
            if self._isdebug:
                self._debug.draw_point(carla.Location(x=ix,y=iy,z=0.3),0.1,carla.Color(0,0,0),0.05)
                if i_s !=0:
                    self._debug.draw_line(carla.Location(x=ix,y=iy,z=0.3),carla.Location(x=pix,y=piy,z=0.3),0.1,carla.Color(0,0,0),0.05)
                pix, piy = ix, iy
        
        print('vehicle speed :',vehicle_speed_mps,'[m/s]')
        print('vehicle acceleration :',vehicle_acceleration_mps,'[m/s^2]')
        ################################################### path generation
        ################################################### USING current vehicle status
        path = self._frenet_optimal_planning(csp, si_d, si_dd, di, di_d, di_dd)
        ################################################### USING trajectory planning
        # path = self._frenet_optimal_planning(csp, self._si_d, self._si_dd, self._di, self._di_d, self._di_dd)
        # self._si_d, self._si_dd, self._di, self._di_d, self._di_dd = path.s_d[1], path.s_dd[1], path.d[1],path.d_d[1],path.d_dd[1]
        inter1 = time.time()-start

        ################################################### TEMP
        # next_wp = ego_wp.next(10.0)[0]
        print('s_d',si_d,'s_dd', si_dd,'d', di,'d_d', di_d,'d_dd', di_dd)
        cur_yaw = path.yaw[0]
        next_wp = carla.Transform(carla.Location(x=path.x[1],y=path.y[1]), carla.Rotation(yaw = path.yaw[1]*180/math.pi))

        if not any([abs(yaw - cur_yaw) > math.pi/36 for yaw in path.yaw]):
            next_wp = carla.Transform(carla.Location(x=path.x[-1],y=path.y[-1]), carla.Rotation(yaw = path.yaw[-1]*180/math.pi))
        else:
            print('TURNING!!!!!')
        print("WHERE are u going? s:", path.s[1],'d:',path.d[1])

        #################################################### Throttle, Steering Controller
        next_speed = path.ds[-1]/self._dt*3.6 #[km/h]
        con_speed = max(next_speed,10.0) #[km/h]
        print('next speed:',next_speed)
        control = self._vehicle_controller.run_step(next_speed, next_wp)
        if vehicle_acceleration_mps > self._max_accel-2 :
            control.throttle = 0.0

        inter2 = time.time()-(inter1+start)
        #################################################### Ackermann Controller
        # con_steer = path.yaw[1]-self._vehicle.get_transform().rotation.yaw*math.pi/180
        # con_speed = max(path.ds[1],15.0)
        # con_accel = max((path.ds[1]-path.ds[0])/self._dt,40.0)
        # con_jerk = 10
        # control = carla.VehicleAckermannControl(steer=con_steer, speed=con_speed, acceleration=con_accel,jerk = con_jerk )
        # print('yaw:',con_steer,'speed:',con_speed,'accel:', con_accel)
        # self._debug.draw_point(carla.Location(x=path.x[-1],y=path.y[-1],z=0.3),0.5, color = carla.Color(r=1,g=0,b=1),life_time=0.1)
        
        
        # print('sqrt(s_dd^2+di_dd^2)',math.hypot(si_dd,di_dd))
        
        print('Throttle',control.throttle)
        if control.brake > 0.0 :
            self._debug.draw_box(carla.BoundingBox(carla.Location(x=vehicle_location.x,y=vehicle_location.y,z=self._vehicle.bounding_box.location.z),self._vehicle.bounding_box.extent),self._vehicle.get_transform().rotation,color = carla.Color(1,0,0),life_time=0.1)
            print('Brake', control.brake)
            print('speeds:', path.ds)
        else:
            self._debug.draw_box(carla.BoundingBox(carla.Location(x=vehicle_location.x,y=vehicle_location.y,z=self._vehicle.bounding_box.location.z),self._vehicle.bounding_box.extent),self._vehicle.get_transform().rotation,color = carla.Color(0,1,0),life_time=0.1)
        print(inter1, inter2)
        ############### Emergency Stop

        """
        sur_vehicles = self._surrounded_vehicles(self._vehicle)

        collision_checking_T = 2.0
        # 2초
        cur_time = time.time()
        if self._start_emergency is None:
            if vehicle_speed_mps>0:
                for sv in sur_vehicles:
                    for i in range(1,int(collision_checking_T/self._dt)):
                        t = i*self._dt
                        pred_ego_location=ego_wp.next(t*vehicle_speed_mps)[0].transform.location
                        pred_sur_location=sv.get_location()+t*sv.get_velocity()
                        # print("pred_ego_location:", pred_ego_location,'/ pred_sur_location:',pred_sur_location)
                        if pred_ego_location.distance(pred_sur_location)<self._safetygap:
                            self._start_emergency = time.time()
                            # print("Distance:",pred_ego_location.distance(pred_sur_location), t)
                            print("\nBRAKE!\n")
                            # control.throttle = 0.0
                            # control.brake = self._max_brake
                            # control.hand_brake = False
                            control.speed = 0.0
                            control.steer = 0.0

                            if self._isdebug:
                                self._debug.draw_point(pred_ego_location,0.1,carla.Color(255,0,0,255),0.1)
                            break

        elif cur_time - self._start_emergency < 2.5: # still emergency
            # control.throttle = 0.0
            # control.brake = self._max_brake
            # control.hand_brake = False
            control.speed = 0.0
            control.steer = 0.0         
        else:
            self._start_emergency = None
        """
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



    def getforwardwaypoints(self, pathlength, vel_mps):
        waypoints = []
        start_wp = self._map.get_waypoint(self._vehicle.get_location())
        from_veh = max(0.5,vel_mps*self._dt+0.1)
        start_wp=start_wp.next(from_veh)[0]
        waypoints.append(start_wp)
        for i in range(10):
            waypoints.append(start_wp.next(pathlength*(i+1)/10.0)[0])
        # if self._isdebug:
        #     for _ in range(len(waypoints)):
        #         self._debug.draw_point(waypoints[_].transform.location,0.1,carla.Color(0,0,255),0.1)
        #         if _ != 0:
        #             self._debug.draw_line(waypoints[_-1].transform.location,waypoints[_].transform.location,0.1,carla.Color(0,0,255),0.1)
        return waypoints


    def _get_vehicle_status(self, ref_path):
        curvature = ref_path.calc_curvature(0)
        tan_angle = ref_path.calc_yaw(0)
        norm_angle =  tan_angle+math.pi/2
        x_d, y_d = self._vehicle.get_velocity().x, self._vehicle.get_velocity().y
        x_dd, y_dd = self._vehicle.get_acceleration().x, self._vehicle.get_acceleration().y
        ref_x, ref_y = ref_path.calc_position(0)

        s_d = x_d*math.cos(tan_angle) + y_d*math.sin(tan_angle)
        s_dd = x_dd*math.cos(tan_angle) + y_dd*math.sin(tan_angle)

        d = -(self._vehicle.get_location().x-ref_x)*math.cos(norm_angle)-(self._vehicle.get_location().y-ref_y)*math.sin(norm_angle)
        
        d_d = -x_d*math.cos(norm_angle) - y_d*math.sin(norm_angle)
        d_dd = - x_dd*math.cos(norm_angle) - y_dd*math.sin(norm_angle)
        
        return s_d, s_dd, d, d_d,d_dd



    def _frenet_optimal_planning(self, ref_path, si_d, si_dd, di, di_d, di_dd):
        fplist = self._calc_frenet_paths(si_d, si_dd, di, di_d, di_dd)
        fplist = self._calc_global_paths(fplist, ref_path)
        fplist = self._check_paths(fplist)
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
        for num in range(len(best_path.x)):
            self._debug.draw_point(carla.Location(x=best_path.x[num],y=best_path.y[num],z=0.4),0.1,carla.Color(0,1,0),0.1)
###########################
        return best_path



    def _calc_frenet_paths(self, si_d, si_dd, di, di_d, di_dd):
        frenet_paths = []
        # generate path to each offset goal
        if not self._ego_wp.get_left_lane().get_left_lane().get_left_lane():
            LEFT = 0.1
        else : 
            self._debug.draw_point(carla.Location(x=self._ego_wp.get_left_lane().get_left_lane().get_left_lane().transform.location.x,y=self._ego_wp.get_left_lane().get_left_lane().get_left_lane().transform.location.y,z=0.4),0.1,carla.Color(0,1,0),0.1)
            LEFT = self._max_road_width
        if not self._ego_wp.get_right_lane().get_right_lane().get_right_lane():
            RIGHT = -0.1
        else :
            self._debug.draw_point(carla.Location(x=self._ego_wp.get_right_lane().get_right_lane().get_right_lane().transform.location.x,y=self._ego_wp.get_right_lane().get_right_lane().get_right_lane().transform.location.y,z=0.4),0.1,carla.Color(0,1,0),0.1)
            RIGHT = -self._max_road_width
        for df in np.linspace(RIGHT, LEFT, self._lateral_no):
            # Lateral motion planning
            # if si_d < 30/3.6 : # Low speed
            #     Min_T = self._MIN_T
            #     Max_T = self._MAX_T
            # else:
            #     Min_T = 15/(si_d)
            #     Max_T = 30/(si_d)
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
                
                for tv in np.linspace(max(si_d - Ti*self._desired_acceleration,0.1),
                                    si_d + Ti*self._desired_acceleration, self._longitudinal_no):
                    tfp = copy.deepcopy(fp)
                    # lon_qp = QuarticPolynomial(0, si_d, si_dd, tv, 0.0, Ti)
                    lon_qp = QuarticPolynomial(0, si_d, si_dd, tv, 0.0, Ti) 

                    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                    Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                    Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                    # square of diff from target speed
                    ds = (self._target_speed - tfp.s_d[-1]) ** 2

                    aligning_cost = sum(np.power(tfp.d,2))

                    tfp.cd = self._K_J * Jp + self._K_T * Ti + self._K_D * (tfp.d[-1] ** 2 + aligning_cost)
                    tfp.cv = self._K_J * Js + self._K_T * Ti + self._K_D * ds
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
                    print(i, fp.s)
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
        violated = False
        reason = [0,0,0,0]
        MAX_SPEED = self._target_speed + self._speed_range
        for i, _ in enumerate(fplist):
            if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
                # print('speed violated')
                violated = True
                reason[0]+=1
            elif any([abs(a) > self._max_accel for a in
                    fplist[i].s_dd]):  # Max accel check
                # print('accel violated',fplist[i].s_dd)
                violated = True
                reason[1]+=1
            elif any([abs(c) > self._max_curvature for c in
                    fplist[i].c]):  # Max curvature check
                # print('curvature violated')
                violated = True
                reason[2]+=1
            elif not self._check_collision(fplist[i]):
                reason[3]+=1
                violated = True
            if not violated :
                ok_ind.append(i)
                for num in range(len(fplist[i].x)):
                    self._debug.draw_point(carla.Location(x=fplist[i].x[num],y=fplist[i].y[num],z=0.3),0.1,carla.Color(0,0,255),0.1)
            else:
                for num in range(len(fplist[i].x)):
                    self._debug.draw_point(carla.Location(x=fplist[i].x[num],y=fplist[i].y[num],z=0.3),0.1,carla.Color(255,0,0),0.1)
                
            violated = False
        if not ok_ind:
            print('vel, accel, curvature, Collision',reason)
            ############################################ DEBUGGING Acceleration
            if reason[0]==0 and reason[2]==0:
                print('WHY ACCELERATION Violated')
                for i in fplist:
                    print(i.s_dd)

        return [fplist[i] for i in ok_ind]


    def _check_collision(self, fp):
        SVs = self._surrounded_vehicles()
        for sv in SVs:
            for i in range(len(fp.x)):
                pred_sur_location=sv.get_location() + i*self._dt*sv.get_velocity() + 0.5*sv.get_acceleration()*(self._dt)**2
                pred_ego_location=carla.Location(x = fp.x[i], y = fp.y[i])
                if pred_ego_location.distance(pred_sur_location)<self._safetygap:
                    print('Collision detected!!')
                    return False
        return True


    def set_destination(self, end_location):
        self._destination = end_location
        return None

    def _surrounded_vehicles(self, radius = 50.0):
        sur_vehicles = []
        world = self._vehicle.get_world()
        vehicles = world.get_actors().filter('vehicle.*')
        for veh in vehicles:
            if veh.id != self._vehicle.id:
                veh_to_ego = veh_to_ego = veh.get_location().distance(self._vehicle.get_location())
                if veh_to_ego < radius:
                    sur_vehicles.append(veh)
        return sur_vehicles