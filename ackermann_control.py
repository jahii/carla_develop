import carla
import random
from time import sleep
import numpy as np
import glob
import os
import sys
import argparse
import pygame
import time
import math
from polynomial_agent import PolynomialAgent

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
    print('EGG not found')
    pass
from agents.navigation.controller import VehiclePIDController
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error
from QuinticPolynomialsPlanner.quintic_polynomials_planner import QuinticPolynomial
from CubicSpline import cubic_spline_planner
from agents.tools.misc import get_acceleration, get_speed
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
        spectator.set_transform(Camera_pos)
        sleep(0.1)


        # Mark vehicle spawning point 
        world_snapshot = world.get_snapshot()
        for actor_snapshot in world_snapshot:
            actual_actor = world.get_actor(actor_snapshot.id)
            if 'vehicle' in actual_actor.type_id :
                debug_temp_loc = actor_snapshot.get_transform().location
                debug.draw_point(carla.Location(x=debug_temp_loc.x,y=debug_temp_loc.y,z=debug_temp_loc.z+2),0.1, carla.Color(1,1,0,0),2)

        control_benz =carla.VehicleAckermannControl(steer=0,steer_speed = 0.0, speed=50.0, acceleration=50.0)
        benz.apply_ackermann_control(control_benz)
        print(benz.get_ackermann_controller_settings())

        start = time.time()

        while True:
            # Camera_pos_x = benz.get_location().x - Cameara_back_dist*math.cos(math.pi/180*benz.get_transform().rotation.yaw)
            # Camera_pos_y = benz.get_location().y - Cameara_back_dist*math.sin(math.pi/180*benz.get_transform().rotation.yaw)
            # Camera_pos = carla.Transform(carla.Location(x=Camera_pos_x, y=Camera_pos_y, z=Camera_pos_z),carla.Rotation(pitch=-20.0,yaw = benz.get_transform().rotation.yaw))
            # spectator.set_transform(Camera_pos)

            end = time.time()
            # benz.apply_ackermann_control(control_benz)
            if end-start>0.1:
                vehicle_location = benz.get_location()

                vehicle_accleration = get_acceleration(benz)/3.6
                vehicle_speed = get_speed(benz)/3.6
                print('Speed :',vehicle_speed)
                print('Acceleration :',vehicle_accleration)

                start = end
            
            sleep(0.1)

            
    finally :
        vehicle_list = world.get_actors().filter('*vehicle*')
        for vehicle in vehicle_list:
            print('DESTROY',vehicle)
            vehicle.destroy()        
        
if __name__ == '__main__':
    main()