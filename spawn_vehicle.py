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
# from polynomial_agent import PolynomialAgent
import weakref

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
    print('EGG founded')
except IndexError:
    print('EGG not found')
    pass
from carla import ColorConverter as cc
import carla

try:
    # Connect to the client and retrieve the world object
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    # Retrieve the spectator object
    spectator = world.get_spectator()
    blueprint_library = world.get_blueprint_library()
    debug=world.debug
    # Set the spectator with an empty transform
    # spectator.set_transform(carla.Transform())
    # This will set the spectator at the origin of the map, with 0 degrees
    # pitch, yaw and roll - a good way to orient yourself in the map

    car_model = random.choice(blueprint_library.filter('vehicle.carlamotors.firetruck'))#'vehicle.nissan.patrol'

    print(car_model)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    # Town12
    # spawn_point = carla.Transform(carla.Location(x=4070, y=4140, z=370)) 

    vehicle = world.spawn_actor(car_model, spawn_point)
    sleep(0.5)
    vehicle_transform = vehicle.get_transform()

    Camera_pos = carla.Transform(spawn_point.location,spawn_point.rotation) #,carla.Rotation(yaw=vehicle.bounding_box.rotation.yaw)
    # world.spawn_actor(['sensor.camera.rgb', cc.Raw, 'Camera RGB'])
    f_vec =spawn_point.get_forward_vector()
    r_vec =spawn_point.get_right_vector()
    print(f_vec)
    print('Spawn point :',spawn_point)
    bound_x = vehicle.bounding_box.location.x*math.cos(math.pi/180*vehicle_transform.rotation.yaw) - vehicle.bounding_box.location.y*math.sin(math.pi/180*vehicle_transform.rotation.yaw)
    bound_y = vehicle.bounding_box.location.x*math.sin(math.pi/180*vehicle_transform.rotation.yaw) + vehicle.bounding_box.location.y*math.cos(math.pi/180*vehicle_transform.rotation.yaw)
    bound_z = vehicle.bounding_box.location.z
    crossed_vec = vehicle.bounding_box.location.cross(f_vec)
    print('Vehicle location:',vehicle.get_location())
    print("Bound location",vehicle.bounding_box.location.x,vehicle.bounding_box.location.y,bound_z)
    print("Bound extent :",vehicle.bounding_box.extent)
    # print(math.sqrt(r_vec.x**2+r_vec.y**2+r_vec.z**2))
    spectator.set_transform(Camera_pos)
    debug.draw_box(carla.BoundingBox(carla.Location(x=vehicle.get_location().x+bound_x,y=vehicle.get_location().y+bound_y,z=vehicle.get_location().z+bound_z),vehicle.bounding_box.extent),spawn_point.rotation,life_time=10.0)
    while True:
        sleep(2)
finally:
    vehicle_list = world.get_actors().filter('*vehicle*')
    for vehicle in vehicle_list:
        print('DESTROY',vehicle)
        vehicle.destroy()

