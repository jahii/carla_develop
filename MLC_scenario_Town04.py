import random
import datetime
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
from IDM_agent import IDMAgent
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
from agents.tools.misc import *
from pygame.locals import K_ESCAPE

# ==============================================================================
# -- DisplayManager ------------------------------------------------------------
# ==============================================================================

class DisplayManager(object):

    def __init__(self, veh, SIZE):
        self.surface = None
        self.world = veh.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        self.spawn_point = veh.get_transform()
        self.Cameara_back_dist = 7.0

        Attach_Camera_pos_x = -self.Cameara_back_dist # *math.cos(math.pi/180*self.spawn_point.rotation.yaw)
        Attach_Camera_pos_y = 0 # -self.Cameara_back_dist*math.sin(math.pi/180*self.spawn_point.rotation.yaw)
        Camera_pos_z = 5.0
        # Attach_Camera_pos = carla.Transform(carla.Location(x=Attach_Camera_pos_x, y=Attach_Camera_pos_y, z=Camera_pos_z),carla.Rotation(pitch=-20.0))
        Attach_Camera_pos = carla.Transform(carla.Location(x=0, y=Attach_Camera_pos_y-50, z=Camera_pos_z+60.0),carla.Rotation(yaw = 90.0, pitch=-60.0))
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

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    def __init__(self, width, height, Agent):
        self.dim = (width, height)
        self.ego_agent = Agent

        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width//4, 40), (220, 0))
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self.start_time = float('inf')
        self.follow_gap = float('inf')
        self.leading_gap = float('inf')
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self._show_ackermann_info = False
        self._ackermann_control = carla.VehicleAckermannControl()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds
        if self.start_time==float('inf'):
            self.start_time = timestamp.elapsed_seconds
        
    def tick(self, clock, c):
        self._notifications.tick(clock)
        if not self._show_info:
            return
        # t = world.player.get_transform()
        # v = world.player.get_velocity()
        # c = world.player.get_control()

        # colhist = world.collision_sensor.get_collision_history()
        # collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        # max_col = max(1.0, max(collision))
        # collision = [x / max_col for x in collision]
        # vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time-self.start_time)),
            '',
            'Ego Veh Speed: % 9.2f km/h' % get_speed(self.ego_agent.vehicle),
            '']
        if self.ego_agent.lead_gap != None:
            self._info_text+=[
                'LV Speed   : % 11.2f km/h' % get_speed(self.ego_agent.LV),
                'Lead gap   : % 11.2f m' % self.ego_agent.lead_gap
            ]
        else:
            self._info_text+=[
                'LV Speed   :        Not Found',
                'Lead gap   :        Not Found'
            ]
        if self.ego_agent.follow_gap != None:
            self._info_text+=[
                'FV Speed   : % 11.2f km/h' % get_speed(self.ego_agent.FV),
                'Follow gap : % 11.2f m' % self.ego_agent.follow_gap
            ]
        else:
            self._info_text+=[
                'FV Speed   :        Not Found',
                'Follow gap :        Not Found'
            ]
        # if self.ego_agent.P_interaction!=None:
        self._info_text+=[
            '',
            'P(Interaction) :        %2.3f' % self.ego_agent.P_interaction if self.ego_agent.P_interaction != None else 'P(Interaction) :    Not Found'
        ]
        self._info_text+=[
            '',
            'Current lane id : %4.1f'% self.ego_agent._map.get_waypoint(self.ego_agent.vehicle.get_location()).lane_id,
            ''
        ]


        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
            if self._show_ackermann_info:
                self._info_text += [
                    '',
                    'Ackermann Controller:',
                    '  Target speed: % 8.0f km/h' % (3.6*self._ackermann_control.speed),
                ]

    def notification(self, text, color=(255, 255, 255), seconds=2.0):
        self._notifications.set_text(text, color, seconds=seconds)

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)

def main():
    client = carla.Client('127.0.0.1', 2000)
    world = client.get_world()
    try:
        # Pygame Setting
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (900, 100) #
        pygame.init()
        pygame.font.init()
        WIDTH = 1600
        HEIGHT = 900
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
        audi_model = vehicles.filter('vehicle.audi.tt')[0]
        audi_colors = audi_model.get_attribute('color').recommended_values
        print(audi_colors)
        world.set_weather(carla.WeatherParameters(sun_azimuth_angle=-90.0000, sun_altitude_angle=90.0,wind_intensity=0.0,fog_density=0.0))
        print(world.get_weather())
        # car_model = random.choice(vehicles)


        TARGET_SPEED_DIFF = 30.0

        # After 3 secs, condition should be satisfied

        # Spawning vehicles
        ego_spawn_point = carla.Transform(carla.Location(x=16.17, y=80.0, z=0.3), carla.Rotation(pitch=0.000000, yaw=-90.29, roll=0.000000))
        ego_spawn_point = world.get_map().get_waypoint(ego_spawn_point.location).transform
        ego_spawn_point.location.z+=0.3

        FV_spawn_point = carla.Transform(carla.Location(x=12.87, y=120.0, z=0.3), carla.Rotation(pitch=0.000000, yaw=-90.29, roll=0.000000))
        # FV_spawn_point = carla.Transform(carla.Location(x=8.8, y=30.0, z=0.3), carla.Rotation(pitch=0.000000, yaw=-90.29, roll=0.000000))
        FV_spawn_point = world.get_map().get_waypoint(FV_spawn_point.location).transform
        FV_spawn_point.location.z+=0.3
        
        LV_spawn_point = carla.Transform(carla.Location(x=12.8, y=70.0, z=0.3), carla.Rotation(pitch=0.000000, yaw=-90.29, roll=0.000000))
        LV_spawn_point = world.get_map().get_waypoint(LV_spawn_point.location).transform
        LV_spawn_point.location.z+=0.3

        audi_model.set_attribute('color', audi_colors[0])
        ego_veh = world.spawn_actor(audi_model, ego_spawn_point)
        sleep(0.01)
        start_wp = world.get_map().get_waypoint(ego_veh.get_location())
        end_wp = start_wp.next(250.0)[0]
        ego_agent = PolynomialAgent(ego_veh, 60)
        ego_agent.set_destination(end_wp.transform.location)

        audi_model.set_attribute('color', '255,255,255')
        FV_veh = world.spawn_actor(audi_model, FV_spawn_point)
        sleep(0.01)
        FV_agent = IDMAgent(FV_veh, 90)
        FV_agent.set_destination(carla.Location(x=12.9, y=-180.0, z=0.3))

        audi_model.set_attribute('color', '0,100,255')
        LV_veh = world.spawn_actor(audi_model, LV_spawn_point)
        sleep(0.01)
        LV_agent = BasicAgent(LV_veh, 70)
        LV_agent.set_destination(carla.Location(x=12.8, y=-250.0, z=0.3))
        
        print('Ego Spawn point :',ego_spawn_point)
        print('FV Spawn point : ',FV_spawn_point)
        print('LV Spawn point :',LV_spawn_point)

        # Camera Setting
        # 항공뷰
        Camera_pos_x = FV_spawn_point.location.x - 20*math.cos(math.pi/180*ego_spawn_point.rotation.yaw)
        Camera_pos_y = FV_spawn_point.location.y - 20*math.sin(math.pi/180*ego_spawn_point.rotation.yaw)
        Camera_pos_z = 90
        Camera_pos = carla.Transform(carla.Location(x=Camera_pos_x, y=Camera_pos_y, z=Camera_pos_z),carla.Rotation(pitch=-50,yaw = FV_spawn_point.rotation.yaw))
        # Behind Ego
        """
        Camera_pos_z = 5.0
        Camera_pos_x = ego_spawn_point.location.x - 7*math.cos(math.pi/180*ego_spawn_point.rotation.yaw)
        Camera_pos_y = ego_spawn_point.location.y - 7*math.sin(math.pi/180*ego_spawn_point.rotation.yaw)
        Camera_pos = carla.Transform(carla.Location(x=Camera_pos_x, y=Camera_pos_y, z=Camera_pos_z),carla.Rotation(pitch=-20.0,yaw = ego_spawn_point.rotation.yaw))
        """
        spectator.set_transform(Camera_pos)


        # Display Setting
        display_manager = DisplayManager(ego_veh,(WIDTH,HEIGHT))
        

        sleep(0.1)

        # Mark vehicle spawning point 
        world_snapshot = world.get_snapshot()
        for actor_snapshot in world_snapshot:
            actual_actor = world.get_actor(actor_snapshot.id)
            if 'vehicle' in actual_actor.type_id :
                debug_temp_loc = actor_snapshot.get_transform().location
                debug.draw_point(carla.Location(x=debug_temp_loc.x,y=debug_temp_loc.y,z=debug_temp_loc.z+2),0.1, carla.Color(255,0,0,0),2)

        start = time.time()
        hud = HUD(WIDTH,HEIGHT,ego_agent)
        world.on_tick(hud.on_world_tick)
        sleep(0.1)
        control_ego = carla.VehicleControl()
        color = (255, 255, 255)
        lane_count_time = time.time()
        lane_start = False
        lane_done = False
        while True:
            ego_agent.get_lead_follow_vehicles()
            
            display_manager.render(display)
            clock.tick()
            hud.tick(clock, control_ego)
            
            if ego_agent.status in ['NEGOTIATING', 'PREPARING']:
                color = (255, 255, 0)
            elif ego_agent.status == 'LANE CHANGING':
                color = (255, 153, 0)
            elif ego_agent.status == 'DONE':
                color = (0, 255, 0)
            hud.notification(ego_agent.status, color = color)
            hud.render(display)
            
            pygame.display.flip()
            ego_agent.get_lead_follow_vehicles()
            end = time.time()
            if end-start>0.1:
                start = end

                # FV control
                control_FV = FV_agent.run_step(debug=False)
                if control_FV.brake > 0:
                    control_FV.brake *= 0.5
                FV_veh.apply_control(control_FV)

                # LV control
                control_LV = LV_agent.run_step(debug=False)
                LV_veh.apply_control(control_LV)
                

                # Ego control
                control_ego = ego_agent.run_step(debug=True)
                if isinstance(control_ego, carla.VehicleControl):
                    ego_veh.apply_control(control_ego)
                elif isinstance(control_ego, carla.VehicleAckermannControl):   
                    ego_veh.apply_ackermann_control(control_ego)
            
            for event in pygame.event.get():
                if event.type==pygame.QUIT or (event.type == pygame.KEYUP and event.key == K_ESCAPE):
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
