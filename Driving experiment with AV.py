#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control with steering wheel Logitech G29.

To drive start by preshing the brake pedal.
Change your wheel_config.ini according to your steering wheel.

To find out the values of your steering wheel use jstest-gtk in Ubuntu.

"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('./carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
from copy import deepcopy
import time
from numba import jit
import csv
from polynomial_agent import PolynomialAgent, PolynomialAgentBaseLine
import json



if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_RETURN
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, actor_filter, ID, spawn_location):
        self.ID = ID
        self.start_point = spawn_location
        
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self.imu_sensor = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        # blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint = self.world.get_blueprint_library().filter('vehicle.audi.tt')[0]
        blueprint.set_attribute('role_name', 'hero')
        blueprint.set_attribute('color', '255,255,255')
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            if self.ID == 0:
                print('FV Spawn point:',self.start_point)
            else:
                print('MV Spawn point:',self.start_point)
            self.player = self.world.try_spawn_actor(blueprint, self.start_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        if self.ID == 1:
            self.camera_manager.set_sidecamera()
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock, contoller):
        self.hud.tick(self, clock, contoller)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
        if self.camera_manager.sidecamera is not None:
            self.camera_manager.sidecamera.destroy()
        

# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================


class DualControl(object):
    def __init__(self, world, start_in_autopilot):
        self.numactor = len(world)
        self.START=False
        self.EXPERIMENT_START=False
        self._autopilot_enabled = start_in_autopilot
        self._current_control = 0

        self.steerCmd = None
        self.BrakePedal_MV = None
        self.AccelPedal_MV = None
        self.BrakePedal_FV = None
        self.AccelPedal_FV = None
        self.MV_leftlight = False

        self._control = [carla.VehicleControl() for _ in range(self.numactor)]
        self._lights = [carla.VehicleLightState.NONE for _ in range(self.numactor)]
        for i in range(self.numactor):
            world[i].player.set_autopilot(self._autopilot_enabled)
            world[i].player.set_light_state(self._lights[i])
        self._steer_cache = 0.0
        # world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        # initialize steering wheel
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()

        print('joystick count:',joystick_count)
        if joystick_count == 0:
            self._joystick_enabled = False
        else:
            self._joystick_enabled = True
            self._joystick = [pygame.joystick.Joystick(i) for i in range(len(world))]
            for i in range(self.numactor):
                self._joystick[i].init()

        self._parser = ConfigParser()
        self._parser.read('wheel_config.ini')
        self._steer_idx = int(self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._restart_idx = int(self._parser.get('G29 Racing Wheel', 'restart'))
        self._hud_idx = int(self._parser.get('G29 Racing Wheel', 'hud'))
        self._leftblink_idx = int(self._parser.get('G29 Racing Wheel', 'leftblink'))
        self._rightblink_idx = int(self._parser.get('G29 Racing Wheel', 'rightblink'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(self._parser.get('G29 Racing Wheel', 'handbrake'))
        self._camera_idx = int(self._parser.get('G29 Racing Wheel', 'camera'))
    def parse_events(self, world, clock):
        current_lights = deepcopy(self._lights)
        for event in pygame.event.get():
            if event.type == pygame.QUIT :
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                i = event.joy
                if event.button == self._restart_idx:
                    world[i].restart()
                elif event.button == self._hud_idx:
                    world[i].hud.toggle_info()
                elif event.button == self._camera_idx:
                    world[i].camera_manager.toggle_camera()
                elif event.button == self._reverse_idx:
                    self._control[i].gear = 1 if self._control[i].reverse else -1
                elif event.button == self._leftblink_idx:
                    current_lights[i] ^= carla.VehicleLightState.LeftBlinker
                    if i == 1:
                        self.MV_leftlight = not self.MV_leftlight
                elif event.button == self._rightblink_idx:
                    current_lights[i]  ^= carla.VehicleLightState.RightBlinker
            elif event.type == pygame.KEYUP:
                i = self._current_control
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world[i].restart()
                elif event.key == K_F1:
                    world[i].hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world[i].hud.help.toggle()
                elif event.key == K_TAB:
                    world[i].camera_manager.toggle_camera()
                elif event.key == K_TAB and pygame.key.get_mods() & KMOD_CTRL:
                    self._current_control = (self._current_control+1)%self.numactor
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world[i].next_weather(reverse=True)
                elif event.key == K_c:
                    world[i].next_weather()
                elif event.key == K_BACKQUOTE:
                    world[i].camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world[i].camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world[i].camera_manager.toggle_recording()
                elif event.key == K_RETURN:
                    self.START = True
                if isinstance(self._control[i], carla.VehicleControl):
                    if event.key == K_q:
                        self._control[i].gear = 1 if self._control[i].reverse else -1
                    elif event.key == K_m:
                        self._control[i].manual_gear_shift = not self._control[i].manual_gear_shift
                        self._control[i].gear = world[i].player.get_control().gear
                        world[i].hud.notification('%s Transmission' %
                                               ('Manual' if self._control[i].manual_gear_shift else 'Automatic'))
                    elif self._control[i].manual_gear_shift and event.key == K_COMMA:
                        self._control[i].gear = max(-1, self._control[i].gear - 1)
                    elif self._control[i].manual_gear_shift and event.key == K_PERIOD:
                        self._control[i].gear = self._control[i].gear + 1
                    elif event.key == K_p:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world[i].player.set_autopilot(self._autopilot_enabled)
                        world[i].hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
            
        if self.EXPERIMENT_START:
            self._parse_vehicle_wheel(pygame.key.get_pressed(),clock.get_time())
            for i in range(self.numactor):
                self._control[i].reverse = self._control[i].gear < 0
                if self._control[i].brake:
                    current_lights[i] |= carla.VehicleLightState.Brake
                else:
                    current_lights[i] &= ~carla.VehicleLightState.Brake
                if self._control[i].reverse:
                    current_lights[i] |= carla.VehicleLightState.Reverse
                else:
                    current_lights[i] &= ~carla.VehicleLightState.Reverse
                if current_lights[i] != self._lights[i]:
                    self._lights[i] = current_lights[i]
                world[i].player.set_light_state(carla.VehicleLightState(self._lights[i]))
                world[i].player.apply_control(self._control[i])


    def _parse_vehicle_wheel(self,keys,milliseconds):
        if self._joystick_enabled is False:
            if keys[K_UP] or keys[K_w]:
                
                self._control[0].throttle = min(self._control[0].throttle + 0.1, 1.00)
            else:
                self._control[0].throttle = 0.0

            if keys[K_DOWN] or keys[K_s]:
                self._control[0].brake = min(self._control[0].brake + 0.2, 1)
            else:
                self._control[0].brake = 0
            steer_increment = 5e-4 * milliseconds
            if keys[K_LEFT] or keys[K_a]:
                if self._steer_cache > 0:
                    self._steer_cache = 0
                else:
                    self._steer_cache -= steer_increment
            elif keys[K_RIGHT] or keys[K_d]:
                if self._steer_cache < 0:
                    self._steer_cache = 0
                else:
                    self._steer_cache += steer_increment
            else:
                self._steer_cache = 0.0
            self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
            self._control[0].steer = round(self._steer_cache, 1)
            self._control[0].hand_brake = keys[K_SPACE]
        
        else:
            for i in range(self.numactor):
                numAxes = self._joystick[i].get_numaxes()
                jsInputs = [float(self._joystick[i].get_axis(j)) for j in range(numAxes)]
                # print (jsInputs)
                jsButtons = [float(self._joystick[i].get_button(j)) for j in
                            range(self._joystick[i].get_numbuttons())]
                # print(jsButtons)

                # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
                # For the steering, it seems fine as it is
                K1 = 1.0  # 0.55
                steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])
                if i == 1:
                    self.steerCmd = steerCmd

                K2 = -0.48  # 1.6
                throttleCmd = K2 - (2.05 * math.log10(
                    0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
                if throttleCmd <= 0:
                    throttleCmd = 0
                elif throttleCmd > 0.75:
                    throttleCmd = 0.75
                if i==0:
                    self.AccelPedal_FV = jsInputs[self._throttle_idx]
                else:
                    self.AccelPedal_MV = jsInputs[self._throttle_idx]

                brakeCmd = -0.6 - (2.05 * math.log10(
                    0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
                if brakeCmd <= 0:
                    brakeCmd = 0
                elif brakeCmd > 1:
                    brakeCmd = 1
                if i==0:
                    self.BrakePedal_FV = jsInputs[self._brake_idx]
                else:
                    self.BrakePedal_MV = jsInputs[self._brake_idx]

                self._control[i].steer = steerCmd
                self._control[i].brake = brakeCmd
                self._control[i].throttle = throttleCmd

                #toggle = jsButtons[self._reverse_idx]

                self._control[i].hand_brake = bool(jsButtons[self._handbrake_idx])


    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height, ID):
        self.ID = ID
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (ID*width, height - 40))
        font = pygame.font.Font(pygame.font.get_default_font(), 70)
        self._speedindicator = SpeedIndicator(font,(200, 100), (ID*width+900, height - 200))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = False
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock, controller):
        self._notifications.tick(world, clock)
        self._speedindicator.tick(world, controller.EXPERIMENT_START, controller.MV_leftlight)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        # vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.world.get_map().name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',]
            # 'Number of vehicles: % 8d' % len(vehicles)]
        # if len(vehicles) > 1:
        #     self._info_text += ['Nearby vehicles:']
        #     distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
        #     vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
        #     for d, vehicle in sorted(vehicles):
        #         if d > 200.0:
        #             break
        #         vehicle_type = get_actor_display_name(vehicle, truncate=22)
        #         self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            if self.ID == 1:
                display.blit(info_surface, (self.dim[0], 0))
            else:
                display.blit(info_surface, (0, 0))
            v_offset = 4
            if self.ID == 1:
                bar_h_offset=100+self.dim[0]
            else:
                bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x+self.dim[0]*self.ID + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
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
                    if self.ID ==1:
                        display.blit(surface, (self.dim[0]+8, v_offset))
                    else:
                        display.blit(surface, (8, v_offset))
                v_offset += 18
        
        # self._notifications.render(display)
        self._speedindicator.render(display)
        self.help.render(display)


# ==============================================================================
# -- SpeedIndicator  -----------------------------------------------------------
# ==============================================================================


class SpeedIndicator(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.surface = pygame.Surface(self.dim)

    def tick(self, world, EXPERIMENT_START, left_blink):
        v = world.player.get_velocity()
        color = (255,255,255)
        if not EXPERIMENT_START:
            color = (255,255,0)
        elif 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2) > 120.0 or world.imu_sensor.accelerometer[0]<-5 or world.imu_sensor.accelerometer[0]>3:
            color = (255,0,0)
        text_texture = self.font.render(str(round(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))), True, color)
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (2*self.dim[0]//5, self.dim[1]//6))
        if left_blink and world.ID == 1:
            # Define the size and position of the indicator
            blink_indicator_size = (20, 20)  # Size of the indicator (width, height)
            blink_indicator_position = (0, self.dim[1]//2 - blink_indicator_size[1]//2)  # Position it at left center
            
            # Draw a green rectangle for the left blink indicator
            blink_indicator_rect = pygame.Rect(blink_indicator_position[0], blink_indicator_position[1], blink_indicator_size[0], blink_indicator_size[1])
            pygame.draw.rect(self.surface, (0, 255, 0), blink_indicator_rect)
        
    def render(self, display):
        display.blit(self.surface, self.pos)


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

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self.sidesurface = None
        self._parent = parent_actor
        self.hud = hud
        self.sidesize = [self.hud.dim[0]//5, self.hud.dim[1]//5]
        self.recording = False
        self.sidecamera = None
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            carla.Transform(carla.Location(x=+0.0*bound_x, y=-0.2*bound_y, z=1.00*bound_z), carla.Rotation(pitch=15.0)),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                bp.set_attribute('fov','110')
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None
        self.sidebp = bp_library.find('sensor.camera.rgb')
        self.sidebp.set_attribute('image_size_x', str(self.sidesize[0]))
        self.sidebp.set_attribute('image_size_y', str(self.sidesize[1]))
        self.sidebp.set_attribute('fov','50')

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index
    
    def set_sidecamera(self):
        Attachment = carla.AttachmentType
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        if self.sidecamera is not None:
            self.sidecamera.destroy()
            self.sidesurface = None
        self.sidecamera = self._parent.get_world().spawn_actor(self.sidebp, carla.Transform(carla.Location(x=0.4, y=-1*bound_y, z=0.8*bound_z),carla.Rotation(pitch = 0.0, yaw=180.0)),attach_to=self._parent, attachment_type =  Attachment.Rigid)
        weak_self = weakref.ref(self)
        self.sidecamera.listen(lambda image: CameraManager._parse_image_side(weak_self, image)) 
    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            if self.hud.ID == 1:
                display.blit(self.surface, (self.hud.dim[0],0))
                if self.sidesurface is not None:
                    display.blit(self.sidesurface, (self.hud.dim[0],self.hud.dim[1]-self.sidesize[1]))
            else:
                display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data) # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

    @staticmethod
    def _parse_image_side(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        array = array[:, ::-1]
        self.sidesurface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (-1920, 0) #-args.width//2
    pygame.init()
    pygame.font.init()
    DISTANCE = random.choice([10.0, 20.0, 30.0])
    V_DIFF = random.choice([10.0, 20.0, 30.0])
    EGO_SPEED = random.choice([40.0, 50.0, 60.0])
    # DISTANCE = 20.0
    # V_DIFF = 10.0
    # EGO_SPEED = 40.0
    CRITERIA_DISTANCE = DISTANCE ##
    FV_SPEED = EGO_SPEED + V_DIFF
    LV_SPEED = EGO_SPEED+5
    EGO_y = -40.0+EGO_SPEED/3.6*3
    VEHICLE_LENGTH = 2.090605*2
    INITIAL_DISTANCE = CRITERIA_DISTANCE + 3*V_DIFF/3.6 + VEHICLE_LENGTH 
    
    if 40 <= FV_SPEED < 55:
        FV_GEAR = 3
    elif 55 <= FV_SPEED < 78:
        FV_GEAR = 4
    elif 78 <= FV_SPEED < 100:
        FV_GEAR = 5
    elif 100 <= FV_SPEED:
        FV_GEAR = 6
    
    if 25 <= EGO_SPEED < 40:
        EGO_GEAR = 2
    elif 40 <= EGO_SPEED < 55:
        EGO_GEAR = 3
    elif 55 <= EGO_SPEED < 78:
        EGO_GEAR = 4
    elif 78 <= EGO_SPEED < 100:
        EGO_GEAR = 5
    print('Distance:', DISTANCE,', Speed difference:',V_DIFF,', EGO speed:', EGO_SPEED)
    
    data = []

    world = None
    FV_location = carla.Transform(carla.Location(x=12.8, y=EGO_y+INITIAL_DISTANCE, z=0.300000), carla.Rotation(yaw=-90.29))
    EGO_location = carla.Transform(carla.Location(x=16.17, y=EGO_y, z=0.300000), carla.Rotation(yaw=-90.29))
    LV_location = carla.Transform(carla.Location(x=12.8, y=EGO_y-VEHICLE_LENGTH-10, z=0.300000), carla.Rotation(yaw=-90.29))

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(3.0)

        # Obtain the world object
        world = client.get_world()
        spectator = world.get_spectator()
        settings = world.get_settings()
        world.set_weather(carla.WeatherParameters(sun_azimuth_angle=-90, sun_altitude_angle=90.0,wind_intensity=0.0,fog_density=0.0))
        audi_model = world.get_blueprint_library().filter('vehicle.audi.tt')[0]
        audi_colors = audi_model.get_attribute('color').recommended_values

        FV_location = world.get_map().get_waypoint(FV_location.location).transform
        EGO_location = world.get_map().get_waypoint(EGO_location.location).transform
        LV_location = world.get_map().get_waypoint(LV_location.location).transform
        FV_location.location.z += 0.3
        EGO_location.location.z += 0.3
        LV_location.location.z += 0.3

        # ------------------------------------------------------  Camera Setting
        Camera_pos_x = EGO_location.location.x - 20*math.cos(math.pi/180*EGO_location.rotation.yaw)
        Camera_pos_y = EGO_location.location.y - 20*math.sin(math.pi/180*EGO_location.rotation.yaw)
        Camera_pos_z = 90
        Camera_pos = carla.Transform(carla.Location(x=Camera_pos_x, y=Camera_pos_y, z=Camera_pos_z),carla.Rotation(pitch=-50,yaw = EGO_location.rotation.yaw))
        spectator.set_transform(Camera_pos)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height, 0)

        # ----------------------------------------------------- SPAWN EGO VEHICLE
        audi_model.set_attribute('color', audi_colors[0])
        EGO_actor = world.spawn_actor(audi_model, EGO_location)
        EGO_agent = random.choice([PolynomialAgent(EGO_actor, EGO_SPEED),PolynomialAgentBaseLine(EGO_actor, EGO_SPEED)]) 
        # EGO_agent = PolynomialAgent(EGO_actor, EGO_SPEED)
        # EGO_agent = PolynomialAgentBaseLine(EGO_actor, EGO_SPEED)
        EGO_agent._pathlength = 125.0
        EGO_agent.set_destination(end_location = carla.Location(x=15.460343, y=-168.618637, z=0.006978))
        EGO_agent.set_distance_criteria(CRITERIA_DISTANCE)
        EGO_actor.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=EGO_GEAR))
        if isinstance(EGO_agent, PolynomialAgent):
            MODEL = "Proposed"
        else:
            MODEL = "BaseLine"
        print()
        print(MODEL)
        print()
        # ----------------------------------------------------- SPAWN FOLLOWING VEHICLE and SETTING
        world_FV = World(world, hud, args.filter, 0, FV_location)
        
        controller = DualControl([world_FV], args.autopilot)
        clock = pygame.time.Clock()
        FV_actor = world_FV.player
        FV_actor.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=FV_GEAR))
        
        # ----------------------------------------------------- SPAWN LEADING VEHICLE
        audi_model.set_attribute('color', '0,100,255')
        LV_actor = world.spawn_actor(audi_model, LV_location)
        print('LV Spawn point:',LV_location)

        # ----------------------------------------------------- World synchronous setting
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1/20
        world.apply_settings(settings)
        
        while True:
            clock.tick_busy_loop(20)
            if controller.parse_events([world_FV], clock):
                return
            
            world_FV.tick(clock, controller)
            world.tick()
            

            world_FV.render(display)
            gap_btw_MV = (FV_actor.get_location().y-FV_actor.bounding_box.extent.x) - (EGO_actor.get_location().y + EGO_actor.bounding_box.extent.x)
            gap_btw_LV = (FV_actor.get_location().y-FV_actor.bounding_box.extent.x) - (LV_actor.get_location().y + LV_actor.bounding_box.extent.x)


            if controller.START and not controller.EXPERIMENT_START:
                
                FV_actor.enable_constant_velocity(carla.Vector3D(FV_SPEED/3.6, 0, 0))
                EGO_actor.enable_constant_velocity(carla.Vector3D(EGO_SPEED/3.6, 0, 0))
                LV_actor.enable_constant_velocity(carla.Vector3D(LV_SPEED/3.6, 0, 0))
                

                if CRITERIA_DISTANCE >= gap_btw_MV:
                    # FV_actor.apply_control(carla.VehicleControl(gear=FV_GEAR))
                    # EGO_actor.apply_control(carla.VehicleControl(gear=EGO_GEAR))
                    FV_actor.disable_constant_velocity()
                    EGO_actor.disable_constant_velocity()
                    FV_actor.apply_control(carla.VehicleControl(gear=FV_GEAR))
                    EGO_actor.apply_control(carla.VehicleControl(gear=EGO_GEAR))
                    controller.EXPERIMENT_START = True
                    EGO_agent.init_agent()
                    


            if controller.EXPERIMENT_START:
                EGO_agent.get_lead_follow_vehicles()
                control_ego = EGO_agent.run_step(debug=True)
                EGO_actor.apply_control(control_ego)
                simulation_time = world.get_snapshot().timestamp.elapsed_seconds
                speed_FV = FV_actor.get_velocity().length() * 3.6 # Convert m/s to km/h
                speed_EGO = EGO_actor.get_velocity().length() * 3.6 # Convert m/s to km/h

                interation_probability = EGO_agent.P_interaction

                data.append(
                    (round(simulation_time,4),
                    FV_actor.get_location().x, FV_actor.get_location().y,
                    EGO_actor.get_location().x, EGO_actor.get_location().y,
                    LV_actor.get_location().x, LV_actor.get_location().y,
                    round(speed_FV,3), round(speed_EGO,3), 
                    world_FV.imu_sensor.accelerometer[0], EGO_agent.imu_sensor.accelerometer[0],
                    FV_actor.get_transform().rotation.yaw, EGO_actor.get_transform().rotation.yaw,
                    gap_btw_MV,
                    gap_btw_LV,
                    interation_probability,
                    EGO_agent.status,
                    controller.AccelPedal_FV, controller.BrakePedal_FV
                    ))
            
            # 두 차량 간의 거리, 차량의 좌표, IMU 센서값, 깜빡이, Front center 좌표, interaction probability
            pygame.display.flip()
            # i = (i+1)%2
    finally:
        if EGO_actor.get_location().y < FV_actor.get_location().y:
            result = "success"
        else:
            result = "failure"
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = 0.0
        world.apply_settings(settings)
        LV_actor.destroy()
        if world_FV is not None:
            world_FV.destroy()
        if EGO_actor is not None:
            EGO_actor.destroy()

        if data:
            base_path = "./DS_experiment_data/"
            base_name = MODEL+'_'+str(int(DISTANCE))+'_'+str(int(V_DIFF))+'_'+str(int(EGO_SPEED))+'_'+result+'.csv'
            file_path = create_unique_file(base_path, base_name)
            
            with open(file_path,'w',newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time",
                                 "x_FV","y_FV", "x_MV","y_MV", "x_LV","y_LV",
                                 "speed_FV","speed_MV",
                                 "Acc_x_FV","Acc_x_MV",
                                 "Yaw_FV","Yaw_MV",
                                 "gap_btw_MV","gap_btw_LV",
                                 "Interacting possibility",
                                 "Process state",
                                 "AccelPedal_FV", "BrakePedal_FV"
                                 ])
                writer.writerows(data)

        # date = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
        # if EGO_agent.DATA_TO_SAVE:
        #     with open('Path_data/'+str(date)+'.json', 'w') as file:
        #         json.dump(EGO_agent.DATA_TO_SAVE, file, indent=4)
        #     print(str(date)+'.json SAVED!')
        pygame.quit()


# ==============================================================================
# -- Creating Data -------------------------------------------------------------
# ==============================================================================

def create_unique_file(base_path, base_name):
    # 파일 경로 생성
    file_path = os.path.join(base_path, base_name)
    
    # 파일 이름과 확장자 분리
    name_part, extension = os.path.splitext(base_name)
    
    counter = 1
    # 파일이 존재하면 번호를 증가시켜가며 새 경로 생성
    while os.path.exists(file_path):
        file_path = os.path.join(base_path, f"{name_part}_{counter}{extension}")
        counter += 1
    
    return file_path


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1920x1080', # 5120x1440 
        help='window resolution (default: 1920)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
