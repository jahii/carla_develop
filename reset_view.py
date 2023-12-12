import carla
import random

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()
# Retrieve the spectator object
spectator = world.get_spectator()
blueprint_library = world.get_blueprint_library()


spectator.set_transform(carla.Transform(carla.Location(x=100,z=400), carla.Rotation(yaw=-90)))
