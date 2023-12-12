# import carla
import random
from time import sleep
import numpy as np
import glob
import os
import sys
import argparse
# import pygame
import time
import math

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

from agents.navigation.controller import VehiclePIDController
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error
from QuinticPolynomialsPlanner.quintic_polynomials_planner import QuinticPolynomial
from CubicSpline import cubic_spline_planner
from agents.tools.misc import *
from polynomial_agent import PolynomialAgent

def main():
    try:
        # Parameters
        pathminlength = 7.0
        pathlength = 20.0 # [m]
        base_range = 3.0
        distance_ratio = 0.5



        # Connect to the client and retrieve the world object
        client = carla.Client('127.0.0.1', 2000)
        world = client.get_world()
        spectator = world.get_spectator()
        blueprint_library = world.get_blueprint_library()
        debug = world.debug

        # Initial Setting(vehicles, weather, model...)
        vehicles = blueprint_library.filter('vehicle.*')
        
        benz_model = vehicles.filter('vehicle.mercedes.coupe')[0]
        benz_length = 3.289
        world.set_weather(carla.WeatherParameters.ClearNoon)
        car_model = random.choice(vehicles)

        # Spawning vehicles
        spawn_point = random.choice(world.get_map().get_spawn_points())
        benz_spawn_point = carla.Transform(carla.Location(x=183.000000, y=52.500000, z=0.300000))

        benz = world.spawn_actor(benz_model, spawn_point)
        print('Mercedes Spawn point : ', spawn_point)
        sleep(0.1)

        # Camera Setting
        Cameara_back_dist = 7.0
        Camera_pos_x = spawn_point.location.x - Cameara_back_dist*math.cos(math.pi/180*spawn_point.rotation.yaw)
        Camera_pos_y = spawn_point.location.y - Cameara_back_dist*math.sin(math.pi/180*spawn_point.rotation.yaw)
        Camera_pos_z = 5.0
        Camera_pos = carla.Transform(carla.Location(x=Camera_pos_x, y=Camera_pos_y, z=Camera_pos_z),carla.Rotation(pitch=-20.0,yaw = spawn_point.rotation.yaw))
        # camera_transform = carla.Transform(carla.Location(x=10, z=10))
        # camera = world.spawn_actor(camera_bp, camera_transform, attach_to=benz)
        # spectator.set_transform(camera.get_transform())
        spectator.set_transform(Camera_pos)
        sleep(0.1)


        # Mark vehicle spawning point 
        world_snapshot = world.get_snapshot()
        for actor_snapshot in world_snapshot:
            actual_actor = world.get_actor(actor_snapshot.id)
            if 'vehicle' in actual_actor.type_id :
                debug_temp_loc = actor_snapshot.get_transform().location
                debug.draw_point(carla.Location(x=debug_temp_loc.x,y=debug_temp_loc.y,z=debug_temp_loc.z+2),0.1, carla.Color(1,1,0,0),2)

        benz_agent = PolynomialAgent(benz, 60)
        # benz_agent = BasicAgent(benz, 45)
        
        start_wp = world.get_map().get_waypoint(benz.get_location())
        end_wp = start_wp.next(400.0)[0]
        benz_agent.set_destination(end_wp.transform.location)

        
        start = time.time()


        tick1 = 0
        while True:
            waypoint = world.get_map().get_waypoint(benz.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving))
            Camera_pos_x = benz.get_location().x - Cameara_back_dist*math.cos(math.pi/180*benz.get_transform().rotation.yaw)
            Camera_pos_y = benz.get_location().y - Cameara_back_dist*math.sin(math.pi/180*benz.get_transform().rotation.yaw)
            Camera_pos = carla.Transform(carla.Location(x=Camera_pos_x, y=Camera_pos_y, z=Camera_pos_z),carla.Rotation(pitch=-20.0,yaw = benz.get_transform().rotation.yaw))
            spectator.set_transform(Camera_pos)

            end = time.time()

            if end-start>0.1:
                vehicle_location = benz.get_location()
                ego_wp = world.get_map().get_waypoint(vehicle_location)

                

                control_benz = benz_agent.run_step(debug=True)
                control_benz.manual_gear_shift = False
                benz.apply_control(control_benz)
                # benz.apply_ackermann_control(control_benz)
                vehicle_accleration = get_acceleration(benz)/3.6
                if vehicle_accleration>10.0:
                    print('Acceleration :',vehicle_accleration)
                tick1 += 1

                if tick1 > 4:
                    # print(benz.get_location(),', Velocity:',benz.get_velocity(),end='\n\n')
                    v_vec = benz.get_velocity()
                    forward_vec = benz.get_transform().get_forward_vector()
                    right_vec = benz.get_transform().get_right_vector()
                    vx = forward_vec.dot(v_vec)
                    vy = right_vec.dot(v_vec)
                    # print('vx:',vx,'\nvy:',vy)
                    # print('angular_velocity :',benz.get_angular_velocity().z*math.pi/180)
                    tick1 = 0
                start = end
            
            # sleep(0.1)

            
    finally :
        vehicle_list = world.get_actors().filter('*vehicle*')
        for vehicle in vehicle_list:
            print('DESTROY',vehicle)
            vehicle.destroy()
        




"""Yaw Rate

v_vec = benz.get_velocity()
forward_vec = benz.get_transform().get_forward_vector()
slip_angle = v_vec.get_vector_angle(forward_vec)
# front_angle = (benz.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)+benz.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel))/2
front_angle = benz.get_wheel_steer_angle(carla.VehicleWheelLocation.Front_Wheel)
yaw_rate = vehicle_speed*math.tan(math.pi/180*front_angle)*math.cos(slip_angle)/benz_length                
print('yaw_rate:',yaw_rate)

"""

def getforwardwaypoints(veh, pathlength):
        waypoints = []
        start_wp = veh.get_world().get_map().get_waypoint(veh.get_location())
        waypoints.append(start_wp)
        waypoints.append(start_wp.next(pathlength/3.0)[0])
        waypoints.append(start_wp.next(pathlength*2.0/3.0)[0])
        waypoints.append(start_wp.next(pathlength)[0])
        debug = veh.get_world().debug
        for _ in range(len(waypoints)):
            debug.draw_point(waypoints[_].transform.location,0.1,carla.Color(0,0,255),0.1)
            if _ != 0:
                debug.draw_line(waypoints[_-1].transform.location,waypoints[_].transform.location,0.1,carla.Color(0,0,255),0.1)
        return waypoints

def get_vehicle_status(veh, ref_path):
    tan_angle = ref_path.calc_yaw(0)
    norm_angle = math.pi/2 + tan_angle
    x_d, y_d = veh.get_velocity().x, veh.get_velocity().y
    x_dd, y_dd = veh.get_acceleration().x, veh.get_acceleration().y
    ref_x, ref_y = ref_path.calc_position(0)

    s_d = x_d*math.cos(tan_angle) + y_d*math.sin(tan_angle)
    s_dd = x_dd*math.cos(tan_angle) + y_dd*math.sin(tan_angle)

    d = (veh.get_location().x-ref_x)*math.cos(norm_angle)+(veh.get_location().y-ref_y)*math.sin(norm_angle)
    d_d = x_d*math.cos(norm_angle) + y_d*math.sin(norm_angle)
    d_dd = x_dd*math.cos(norm_angle) + y_dd*math.sin(norm_angle)
    
    return s_d, s_dd, d, d_d,d_dd




if __name__ == '__main__':
    main()
