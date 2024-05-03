from time import sleep
import numpy as np
import glob
import os
import sys
import argparse
import pygame
import time
import math

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

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
client.set_timeout(3.0)
world = client.get_world()
vehicle_list = world.get_actors().filter('vehicle*')
sensor_list = world.get_actors().filter('sensor*')
for actor in vehicle_list:
    print('DESTROY',actor)
    actor.destroy()
for actor in sensor_list:
    print('DESTROY',actor)
    actor.destroy()



