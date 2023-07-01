import rasterio
from stable_baselines3.common.env_checker import check_env
from connectivity_env import Landscape

from connectivity_env import Landscape

RASTERPATH = './data/restoptr_ex_small.tif'

if __name__ == '__main__':
    # Load raster landscape.
    rast = rasterio.open(RASTERPATH)

    # Instantiate the environment.
    env = Landscape(raster = rast)

    # If the environment doesn't follow the standard interface, an error will be thrown.
    check_env(env, warn=True)

