import rasterio
import os
import time

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

from connectivity_env import Landscape

RASTERPATH = './data/restoptr_ex_small.tif'
LOGGINGPATH = "./model"

if __name__ == '__main__':
    # Load raster landscape.
    rast = rasterio.open(RASTERPATH)

    # Logging.
    os.makedirs(LOGGINGPATH, exist_ok=True)

    # Instantiate the environment.
    env = Landscape(raster = rast)

    # wrap it with a monitor.
    env = Monitor(env, LOGGINGPATH)

    # To periodically evaluate the model and save the best version.
    eval_callback = EvalCallback(env, best_model_save_path='./model/', log_path='./model/', eval_freq=1000,
                                 deterministic=False, render=False)
    # Train the agent.
    max_total_step_num = 150000

    PPO_model_args = {
    "learning_rate": 0.001, # Can constant or e.g. (linearly) decreasing over the timesteps.
    "gamma": 0.99, # Discount factor for future rewards, between 0 (only immediate reward matters) and 1 (future reward equivalent to immediate).
    "verbose": 0, # More info on training steps.
    "seed": 7, # To fix the random seed.
    "ent_coef": 0.01, # Entropy coefficient, to encourage exploration.
    "clip_range": 0.2 # Very roughly: probability of an action can not change by more than a factor 1+clip_range.
    }

    starttime = time.time()
    model = PPO('MlpPolicy', env,**PPO_model_args)
    # Load previous best model parameters if available.
    if os.path.exists(LOGGINGPATH + "best_model.zip"):
        model.set_parameters("model/best_model.zip")
    model.learn(total_timesteps=max_total_step_num, callback=eval_callback)
    dt = time.time()-starttime
    model.save("model/best_model.zip")
    print("Calculation took %g hr %g min %g s"%(dt//3600, (dt//60)%60, dt%60) )
