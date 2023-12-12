import carla
import random
import pygame
import sys
import os
from pygame.locals import KMOD_CTRL
from pygame.locals import K_ESCAPE
from pygame.locals import K_q
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/carla')

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error


pygame.init()
pygame.font.init()
display = pygame.display.set_mode(
            (1280, 720),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()
# Retrieve the spectator object
spectator = world.get_spectator()
blueprint_library = world.get_blueprint_library()

# Set the spectator with an empty transform
# spectator.set_transform(carla.Transform())
# This will set the spectator at the origin of the map, with 0 degrees
# pitch, yaw and roll - a good way to orient yourself in the map


car_model = random.choice(blueprint_library.filter('vehicle.*'))

print(car_model)
spawn_point = carla.Transform(carla.Location(x=183.236069, y=86.662941, z=0.300000), carla.Rotation(pitch=0.000000, yaw=-36.711231, roll=0.000000))
vehicle = world.spawn_actor(car_model, spawn_point)
print('Spawn point :',spawn_point)
bound_x = vehicle.bounding_box.location.x
bound_y = vehicle.bounding_box.location.y
bound_z = vehicle.bounding_box.location.z+0.5

print(bound_x,bound_y,bound_z)
Camera_pos = carla.Transform(carla.Location(x=bound_x+spawn_point.location.x-3.0, y=+0.0*bound_y+spawn_point.location.y, z=2.0*bound_z+spawn_point.location.z),spawn_point.rotation) #,carla.Rotation(yaw=vehicle.bounding_box.rotation.yaw)
# world.spawn_actor(['sensor.camera.rgb', cc.Raw, 'Camera RGB'])

spectator.set_transform(Camera_pos)

clock = pygame.time.Clock()
while True:
    clock.tick()
