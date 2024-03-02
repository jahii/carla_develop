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
from agents.navigation.controller import VehiclePIDController
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error
from QuinticPolynomialsPlanner.quintic_polynomials_planner import QuinticPolynomial
from CubicSpline import cubic_spline_planner
from carla import ColorConverter as cc
import carla

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
    client = carla.Client('127.0.0.1', 2000)
    world = client.get_world()
    try:
        # Pygame Setting
        pygame.init()
        pygame.font.init()
        WIDTH = 1280
        HEIGHT = 720
        display = pygame.display.set_mode(
            (WIDTH, HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        clock = pygame.time.Clock()

        # Connect to the client and retrieve the world object
        spectator = world.get_spectator()
        blueprint_library = world.get_blueprint_library()
        debug = world.debug

        # Initial Setting(vehicles, weather, model...)
        vehicles = blueprint_library.filter('vehicle.*')
        nissan_model = vehicles.filter('vehicle.nissan.patrol')[0]
        dodge_model = vehicles.filter('vehicle.dodge.charger_2020')[0]
        benz_model = vehicles.filter('vehicle.mercedes.coupe')[0]
        world.set_weather(carla.WeatherParameters(sun_azimuth_angle=-1.000000, sun_altitude_angle=45.000000,wind_intensity=0.0,fog_density=0.0))
        car_model = random.choice(vehicles)


        # Spawning vehicles
        ego_spawn_point = carla.Transform(carla.Location(x=183.236069, y=86.662941, z=0.300000), carla.Rotation(pitch=0.000000, yaw=-36.711231, roll=0.000000)) 
        benz_spawn_point = carla.Transform(carla.Location(x=183.000000, y=52.500000, z=0.300000))
        ego_veh = world.spawn_actor(dodge_model, ego_spawn_point)
        human_veh = world.spawn_actor(benz_model, benz_spawn_point)
        print('Ego Spawn point :',ego_spawn_point)
        print('Mercedes Spawn point : ',benz_spawn_point)

        # Camera Setting
        Camera_pos_x = ego_spawn_point.location.x+60
        Camera_pos_y = ego_spawn_point.location.y-30
        Camera_pos_z = 100
        Camera_pos = carla.Transform(carla.Location(x=Camera_pos_x, y=Camera_pos_y, z=Camera_pos_z),carla.Rotation(yaw=90.0,pitch=-89.0))
        spectator.set_transform(Camera_pos)
        sleep(1.0)

        # Display Setting
        display_manager = DisplayManager(ego_veh,(WIDTH,HEIGHT))

        # Mark vehicle spawning point 
        world_snapshot = world.get_snapshot()
        for actor_snapshot in world_snapshot:
            actual_actor = world.get_actor(actor_snapshot.id)
            if 'vehicle' in actual_actor.type_id :
                debug_temp_loc = actor_snapshot.get_transform().location
                debug.draw_point(carla.Location(x=debug_temp_loc.x,y=debug_temp_loc.y,z=debug_temp_loc.z+2),0.1, carla.Color(255,0,0,0),2)

        
        ego_agent = PolynomialAgent(ego_veh, 50)
        hum_agent = BasicAgent(human_veh, 45)
        
        start_wp = world.get_map().get_waypoint(ego_veh.get_location())
        end_wp = start_wp.next(250.0)[0]
        ego_agent.set_destination(end_wp.transform.location)
        hum_agent.set_destination(carla.Location(x=330.000000, y=52.500000, z=0.300000))

        
        start = time.time()

        while True:
            waypoint = world.get_map().get_waypoint(ego_veh.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving))

            end = time.time()

            clock.tick()
            display_manager.render(display)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if end-start>0.1:
                control_hum = hum_agent.run_step(debug=False)
                control_hum.manual_gear_shift = False
                human_veh.apply_control(control_hum)
                control_ego = ego_agent.run_step(debug=True)
                # control_ego.manual_gear_shift = False
                if isinstance(control_ego,carla.VehicleControl):
                    ego_veh.apply_control(control_ego)
                elif isinstance(control_ego, carla.VehicleAckermannControl):   
                    ego_veh.apply_ackermann_control(control_ego)
                # ego_veh.apply_ackermann_control(control_ego)
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
        





if __name__ == '__main__':
    main()
