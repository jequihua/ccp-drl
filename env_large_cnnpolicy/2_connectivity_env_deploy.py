import rasterio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from connectivity_env import Landscape

RASTERPATH = './data/usuev250sVII_cropv4_roads.tif'

if __name__ == '__main__':

    # Load raster landscape.
    rast = rasterio.open(RASTERPATH)
    transform = rast.transform
    crs = rast.crs
    width = rast.width
    height = rast.height

    # Instantiate the environment.
    env = Landscape(raster=rast)

    # Load trained model.
    model = PPO.load("./model/best_model")

    # Reset the environment and deploy the model on it.
    obs, _ = env.reset()

    # Framework to save animated gif of agent deployment.
    fig, ax = plt.subplots(figsize=(6,6))
    plt.axis('off')
    frames = []
    fps=18

    max_steps = 10000
    tot_reward = 0
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        tot_reward += reward
        print("Step {}".format(step + 1),"Action: ", action, 'Tot. Reward: %g'%(tot_reward))

        color_array, final_landscape = env.render(mode='rgb_array')
        frames.append([ax.imshow(color_array, animated=True)])
        # Save raster of final spatial solution.
        if terminated:
            new_dataset = rasterio.open('agent_optimal_full.tif', 'w', driver='GTiff',
                                    height=height, width=width,
                                    count=1, dtype=str(final_landscape.dtype),
                                    crs=crs,
                                    transform=transform)
            new_dataset.write(final_landscape, 1)
            new_dataset.close()
            break

    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None) # To remove white bounding box.
    anim = animation.ArtistAnimation(fig, frames, interval=int(1000/fps), blit=True,repeat_delay=1000)
    anim.save("agent_optimal.gif",dpi=150)

