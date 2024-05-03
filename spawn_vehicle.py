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
    all_car_model = blueprint_library.filter('vehicle.*')
    car_model = random.choice(blueprint_library.filter('vehicle.audi.tt'))
    # print('colors:',car_model.get_attribute('color').recommended_values)
    car_model.set_attribute('color', random.choice(car_model.get_attribute('color').recommended_values))
    print(car_model)

    spawn_point = random.choice(world.get_map().get_spawn_points())
    spawn_point = carla.Transform(carla.Location(x=16.17, y=-163.0, z=0.300000), carla.Rotation(yaw=-90.29)) 
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
    # print('Spawn point :',spawn_point)
    bound_x = vehicle.bounding_box.location.x*math.cos(math.pi/180*vehicle_transform.rotation.yaw) - vehicle.bounding_box.location.y*math.sin(math.pi/180*vehicle_transform.rotation.yaw)
    bound_y = vehicle.bounding_box.location.x*math.sin(math.pi/180*vehicle_transform.rotation.yaw) + vehicle.bounding_box.location.y*math.cos(math.pi/180*vehicle_transform.rotation.yaw)
    bound_z = vehicle.bounding_box.location.z
    crossed_vec = vehicle.bounding_box.location.cross(f_vec)
    # print('Vehicle location:',vehicle.get_location())
    print("Bound location",vehicle.bounding_box.location.x,vehicle.bounding_box.location.y,bound_z)
    print("Bound extent :",vehicle.bounding_box.extent)
    # print(math.sqrt(r_vec.x**2+r_vec.y**2+r_vec.z**2))
    spectator.set_transform(carla.Transform(carla.Location(x=13.1, y=-160, z=0.3),spawn_point.rotation))
    debug.draw_box(carla.BoundingBox(carla.Location(x=vehicle.get_location().x+bound_x,y=vehicle.get_location().y+bound_y,z=vehicle.get_location().z+bound_z),vehicle.bounding_box.extent),spawn_point.rotation,life_time=2.0)
    debug.draw_line(carla.Location(x=13.3, y=-30, z=0.3),carla.Location(x=13.3, y=-169, z=0.3),life_time = 2)
    debug.draw_line(carla.Location(x=9.9, y=-30, z=0.3),carla.Location(x=9.9, y=-169, z=0.3),life_time = 2)
    debug.draw_line(carla.Location(x=16.8, y=-30, z=0.3),carla.Location(x=16.8, y=-169, z=0.3),life_time = 2)
    debug.draw_line(carla.Location(x=13.3, y=-175, z=0.3),carla.Location(x=16.8, y=-169, z=0.3),life_time = 2)
    print(vehicle.type_id, 'vehicle length:',2*vehicle.bounding_box.extent.x)
    print(vehicle.type_id, 'vehicle width:', 2*vehicle.bounding_box.extent.y)
    while True:
        sleep(2)
finally:
    vehicle_list = world.get_actors().filter('*vehicle*')
    for vehicle in vehicle_list:
        print('DESTROY',vehicle)
        vehicle.destroy()

