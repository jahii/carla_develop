
import glob
import os
import sys


try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

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
import carla
from carla import ColorConverter as cc

import argparse
import time
import math
import weakref
import random
from time import sleep
import numpy as np

from agents.navigation.controller import VehiclePIDController
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error
from QuinticPolynomialsPlanner.quintic_polynomials_planner import QuinticPolynomial
from CubicSpline import cubic_spline_planner
from agents.tools.misc import *
from polynomial_agent import PolynomialAgent



class DisplayManager(object):

    def __init__(self,veh,SIZE):
        self.surface = None
        self.world = veh.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        self.spawn_point = veh.get_transform()
        self.Cameara_back_dist = 7.0

        
        Attach_Camera_pos_x = -self.Cameara_back_dist # *math.cos(math.pi/180*self.spawn_point.rotation.yaw)
        Attach_Camera_pos_y = 0 # -self.Cameara_back_dist*math.sin(math.pi/180*self.spawn_point.rotation.yaw)
        Camera_pos_z = 5.0
        Attach_Camera_pos = carla.Transform(carla.Location(x=Attach_Camera_pos_x, y=Attach_Camera_pos_y, z=Camera_pos_z),carla.Rotation(pitch=-20.0))
        attachment = carla.AttachmentType.Rigid
        camera_bp = self.bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x',str(SIZE[0]))
        camera_bp.set_attribute('image_size_y',str(SIZE[1]))
        self.camera_actor = self.world.spawn_actor(camera_bp, Attach_Camera_pos, attach_to=veh,attachment_type=attachment)
        weak_self = weakref.ref(self)
        self.camera_actor.listen(lambda image: DisplayManager._parse_image(weak_self, image))
        
    def spectator_to_vehicle(self):
        spectator = self.world.get_spectator()
        Camera_pos_z = 5.0
        Camera_pos_x = self.spawn_point.location.x - self.Cameara_back_dist*math.cos(math.pi/180*self.spawn_point.rotation.yaw)
        Camera_pos_y = self.spawn_point.location.y - self.Cameara_back_dist*math.sin(math.pi/180*self.spawn_point.rotation.yaw)
        Camera_pos = carla.Transform(carla.Location(x=Camera_pos_x, y=Camera_pos_y, z=Camera_pos_z),carla.Rotation(pitch=-20.0,yaw = self.spawn_point.rotation.yaw))
        spectator.set_transform(Camera_pos)

    def render(self,display):
        if self.surface is not None:
            display.blit(self.surface,(0,0))
            
    def destroy_display(self):
        self.camera_actor.destroy()

    @staticmethod
    def _parse_image(weak_self,image):
        self = weak_self()
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0,1))


def main():
    
    try:
        # Pygame Setting
        pygame.init()
        pygame.font.init()
        WIDTH = 1600
        HEIGHT = 900
        display = pygame.display.set_mode(
            (WIDTH, HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        clock = pygame.time.Clock()


        # Connect to the client and retrieve the world object
        client = carla.Client('localhost', 2000) # 2번 컴 : '143.248.221.198'
        client.set_timeout(10.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        debug = world.debug

        # Initial Setting(vehicles, weather, model...)
        vehicles = blueprint_library.filter('vehicle.*')
        ego_model = vehicles.filter('vehicle.dodge.charger_2020')[0]

        benz_length = 3.289
        world.set_weather(carla.WeatherParameters(sun_azimuth_angle=-1.000000, sun_altitude_angle=45.000000,wind_intensity=0.0,fog_density=0.0))
        car_model = random.choice(vehicles)

        # Spawning vehicles
        # spawn_point = random.choice(world.get_map().get_spawn_points())
        spawn_point = carla.Transform(carla.Location(x=15.2, y=-4.7, z=0.3), carla.Rotation(pitch=0.000000, yaw=-90.0, roll=0.000000)) 
        # Town 4 : carla.Transform(carla.Location(x=15.2, y=-4.7, z=0.3), carla.Rotation(pitch=0.000000, yaw=-90.0, roll=0.000000))
        # Town 6 : carla.Transform(carla.Location(x=183.236069, y=86.662941, z=0.300000), carla.Rotation(pitch=0.000000, yaw=-36.711231, roll=0.000000)) 
        # Town 12: carla.Transform(carla.Location(x=2230, y=3080, z=370), carla.Rotation(pitch=0.000000, yaw=-36.711231, roll=0.000000))
        Ego_actor = world.spawn_actor(ego_model, spawn_point)
        print('Mercedes Spawn point : ', spawn_point)
        sleep(0.1)

        # Display Setting
        display_manager = DisplayManager(Ego_actor,(WIDTH,HEIGHT))
        display_manager.spectator_to_vehicle()
        
        # camera_transform = carla.Transform(carla.Location(x=10, z=10))
        # camera = world.spawn_actor(camera_bp, camera_transform, attach_to=benz)
        # spectator.set_transform(camera.get_transform())
        
        sleep(0.1)


        # Mark vehicle spawning point 
        world_snapshot = world.get_snapshot()
        for actor_snapshot in world_snapshot:
            actual_actor = world.get_actor(actor_snapshot.id)
            if 'vehicle' in actual_actor.type_id :
                debug_temp_loc = actor_snapshot.get_transform().location
                debug.draw_point(carla.Location(x=debug_temp_loc.x,y=debug_temp_loc.y,z=debug_temp_loc.z+2),0.1, carla.Color(1,1,0,0),2)

        benz_agent = PolynomialAgent(Ego_actor, 50)
        # benz_agent = BasicAgent(benz, 45)
        
        start_wp = world.get_map().get_waypoint(Ego_actor.get_location())
        end_wp = start_wp.next(400.0)[0]
        benz_agent.set_destination(end_wp.transform.location)

        
        start = time.time()

        
        tick1 = 0
        while True:
            clock.tick()

            display_manager.render(display)
            end = time.time()
            pygame.display.flip()

            if end-start>0.1:
                control_benz = benz_agent.run_step(debug=True)
                control_benz.manual_gear_shift = False
                Ego_actor.apply_control(control_benz)
                vehicle_accleration = get_acceleration(Ego_actor)/3.6

                if tick1 > 4:
                    # print(benz.get_location(),', Velocity:',benz.get_velocity(),end='\n\n')
                    # v_vec = benz.get_velocity()
                    # forward_vec = benz.get_transform().get_forward_vector()
                    # right_vec = benz.get_transform().get_right_vector()
                    # vx = forward_vec.dot(v_vec)
                    # vy = right_vec.dot(v_vec)
                    # print('vx:',vx,'\nvy:',vy)
                    # print('angular_velocity :',benz.get_angular_velocity().z*math.pi/180)
                    tick1 = 0
                start = end
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                
    finally :
        vehicle_list = world.get_actors().filter('*vehicle*')
        for vehicle in vehicle_list:
            print('DESTROY',vehicle)
            vehicle.destroy()
        display_manager.destroy_display()
        pygame.quit()





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



if __name__ == '__main__':
    main()

