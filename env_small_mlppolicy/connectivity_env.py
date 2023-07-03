import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import geospatial as gs

class Landscape(gym.Env):
    """
    Custom Environment for habitat connectivity optimization (Gymnasium and Stable-Baseline3)
    """
    metadata = {'render.modes': ['console', 'rgb_array']}

    FULLPATH = "./data/"

    ### Constants.
    # Grid classes
    NONNATURAL = 0
    NATURAL = 1
    AGENT = 2

    def __init__(self, raster, budget=6):
        super(Landscape, self).__init__()

        # Initialize the landscape grid (from geotiff raster) + metadata.
        self.budget = budget
        self.budget_init = budget
        self.raster = raster
        self.transform = raster.transform
        self.crs = raster.crs
        self.dimx = raster.width
        self.dimy = raster.height
        self.grid = raster.read(1).astype(np.uint8).copy()
        self.grid_init = raster.read(1).astype(np.uint8).copy()
        self.grid_idx = np.arange(np.size(self.grid)).reshape((self.dimy, self.dimx))

        actions_warp = self.grid_idx[self.grid != self.NATURAL]

        self.actions_warp = actions_warp.flatten()

        # Number of possible indices to which to warp to.
        self.n_actions = len(self.actions_warp)

        # Memory of played (position) values.
        self.taken_actions = np.ones((self.budget), dtype=np.uint8)*-1

        # Initialize agent at the last possible position.
        self.lastposition = self.n_actions-1
        self.step_n = 0

        x, y = gs.numpy_coordinates(self.grid_idx, self.actions_warp[self.n_actions-1])
        self.coordx = int(x)
        self.coordy = int(y)

        # Calculate initial (weighted) Integral Index of Connectivity.
        polys = gs.clump(self.grid, self.transform, connectivity = 4)
        polys_gp = gs.createGDFfromShapes(polys)
        extentarea = gs.extent_area(polys_gp, crs=self.crs)
        self.extentarea = extentarea
        area_matrix = gs.area_matrix(polys_gp)
        np.savetxt(self.FULLPATH + 'nodes.txt', area_matrix, fmt='%d')
        distances = gs.distance_polys_table(polys_gp)
        distances.to_csv(self.FULLPATH + 'distances.txt', header=None, index=None, sep=' ')
        iic = (gs.conefor(area_matrix, path=self.FULLPATH)/self.extentarea**2)/len(polys_gp)
        self.iic_init = iic
        print("Initial weighted IIC.")
        print(self.iic_init)

        # The observation space, it's an (ordered) vector of the positions played up to a certain time step.
        self.observation_space = gym.spaces.Box(low=-1, high=self.n_actions,
                                       shape=(self.budget_init,),
                                       dtype=np.float64)

        # The action space.
        self.action_space = gym.spaces.Discrete(self.n_actions)

    def reset(self, seed = None):
        super().reset(seed=seed)

        # Reset to initial values for next training episode.
        self.budget = self.budget_init
        self.grid = self.grid_init.copy()
        self.lastposition = self.n_actions-1
        self.taken_actions = np.ones((self.budget))*-1
        self.step_n = 0

        observation = self._get_obs()
        info = {}

        return observation, info


    def _get_obs(self):
        # Return observation in the format of self.observation_space.
        return np.sort(self.taken_actions)[::-1]

    def step(self, action):
        # Agents movement.
        step = action

        if self.step_n < self.budget_init:
            self.taken_actions[self.step_n] = step
            self.step_n += 1

        self.lastposition = step

        selected_idx = self.actions_warp[step]

        x, y = gs.numpy_coordinates(self.grid_idx, selected_idx)

        self.coordx = int(x)
        self.coordy = int(y)

        terminated = False
        truncated = False
        reward = 0
        info = {}

        # Add vegetation patch.
        if self.grid[self.coordx, self.coordy] == self.NONNATURAL and self.budget > 0:
            self.grid[self.coordx, self.coordy] = self.NATURAL

            self.budget -= 1

        elif self.grid[self.coordx, self.coordy] != self.NONNATURAL and self.budget > 0:

            polys = gs.clump(self.grid, self.transform, connectivity = 4)
            polys_gp = gs.createGDFfromShapes(polys)
            area_matrix = gs.area_matrix(polys_gp)
            np.savetxt(self.FULLPATH + 'nodes.txt', area_matrix, fmt='%d')
            distances = gs.distance_polys_table(polys_gp)
            distances.to_csv(self.FULLPATH + 'distances.txt', header=None, index=None, sep=' ')
            iic = (gs.conefor(area_matrix, path=self.FULLPATH)/self.extentarea**2)/len(polys_gp)
            reward = iic.item()

            terminated = True

        else:
            polys = gs.clump(self.grid, self.transform, connectivity = 4)
            polys_gp = gs.createGDFfromShapes(polys)
            area_matrix = gs.area_matrix(polys_gp)
            np.savetxt(self.FULLPATH + 'nodes.txt', area_matrix, fmt='%d')
            distances = gs.distance_polys_table(polys_gp)
            distances.to_csv(self.FULLPATH + 'distances.txt', header=None, index=None, sep=' ')
            iic = (gs.conefor(area_matrix, path=self.FULLPATH)/self.extentarea**2)/len(polys_gp)
            reward = iic.item()

            terminated = True

        observation = self._get_obs()

        return observation, reward, terminated, truncated, info

    def render(self, mode='rgb_array'):
        if mode == 'console':
            print(self.grid)
        elif mode == 'rgb_array':
            return self.landscape_plot()
        else:
            raise NotImplementedError()

    def close(self):
        pass

    def landscape_plot(self, plot_inline=False):
        veg_ind = (self.grid == self.NATURAL)

        # Create color array for landscape plotting.
        Color_array = np.zeros((self.dimx, self.dimy, 3), dtype=np.uint8) # Default black.
        Color_array[veg_ind, :] = np.array([0, 255, 0])  # Green vegetation.
        Color_array[self.coordx, self.coordy, :] = np.array([0, 0, 255])  # Blue agent.
        # Plot.
        if plot_inline:
            fig = plt.figure()
            plt.axis('off')
            plt.imshow(Color_array, interpolation='nearest')
            plt.show()

        current_grid = (self.grid != self.grid_init) * 1
        return Color_array, current_grid

