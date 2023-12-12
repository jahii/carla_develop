import glob
import os
import sys
import argparse
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
import random
from time import sleep
import carla


def main():

    # Connect to the client and retrieve the world object
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(60.0)
    world = client.load_world('Town06_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles) 
    # actors = world.get_actors()
    # spectator = world.get_spectator()
    # blueprint_library = world.get_blueprint_library()
    world.unload_map_layer(carla.MapLayer.Buildings)
    world.unload_map_layer(carla.MapLayer.Foliage)

if __name__ == '__main__':
    main()
