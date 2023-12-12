import carla
import random
import pygame
import sys
import os
from pygame.locals import KMOD_CTRL
from pygame.locals import K_ESCAPE
from pygame.locals import K_q
import logging
import math
from time import sleep

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

    car_model = random.choice(blueprint_library.filter('vehicle.*'))#'vehicle.nissan.patrol'

    print(car_model)

    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(car_model, spawn_point)
    print('Spawn point :',spawn_point)
    bound_x = vehicle.bounding_box.location.x
    bound_y = vehicle.bounding_box.location.y
    bound_z = vehicle.bounding_box.location.z
    print("Bound location",bound_x,bound_y,bound_z)
    print("Bound extent :",vehicle.bounding_box.extent)
    Camera_pos = carla.Transform(spawn_point.location,spawn_point.rotation) #,carla.Rotation(yaw=vehicle.bounding_box.rotation.yaw)
    # world.spawn_actor(['sensor.camera.rgb', cc.Raw, 'Camera RGB'])
    r_vec =spawn_point.get_right_vector()
    print(r_vec)
    # print(math.sqrt(r_vec.x**2+r_vec.y**2+r_vec.z**2))
    spectator.set_transform(Camera_pos)
    debug.draw_box(carla.BoundingBox(carla.Location(x=spawn_point.location.x,y=spawn_point.location.y,z=bound_z),vehicle.bounding_box.extent),spawn_point.rotation,life_time=10.0)
    while True:
        sleep(2)
finally:
    vehicle_list = world.get_actors().filter('*vehicle*')
    for vehicle in vehicle_list:
        print('DESTROY',vehicle)
        vehicle.destroy()

