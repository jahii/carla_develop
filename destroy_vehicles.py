import carla
import random
from time import sleep

import glob
import os
import sys
import argparse
def main():

    # Connect to the client and retrieve the world object
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    actors = world.get_actors()
    spectator = world.get_spectator()
    blueprint_library = world.get_blueprint_library()
    vehicle_list = world.get_actors().filter('*vehicle*')

    for vehicle in vehicle_list:
        print('DESTROY',vehicle)
        vehicle.destroy()

if __name__ == '__main__':
    main()
