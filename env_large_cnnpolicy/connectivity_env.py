import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import geospatial as gs

def scale(m,minv,maxv,minlim,maxlim, toint=True):
    scaled = ((m-minv)/(maxv-minv)*(maxlim-minlim))+minlim
    if toint:
        scaled = int(np.around(scaled,0))
    return scaled

class Landscape(gym.Env):
    """
    Custom Environment for Stable Baseline 3 for the connectivity optimization
    """
    metadata = {'render.modes': ['console', 'rgb_array']}

    FULLPATH = "./data/"

    def __init__(self, raster, budget = 20000):
        super(Landscape, self).__init__()

        # Set constants.
        self.maxvsize = 6
        self.minvsize = 2

        # Initialize the landscape grid (from geotiff raster) + metadata.
        self.budget = budget
        self.budget_init = budget
        self.raster = raster
        self.transform = raster.transform
        self.crs = raster.crs
        self.ncolumns = raster.width
        self.nrows = raster.height

        band = self.raster.read(1).astype(np.uint8)
        # Map land-cover classes to their restoration costs.
        # Natural vegetation has value 4.
        band[band == 4] = 0  # Cost of transforming NV to NV is 0.
        band[band == 8] = 10  # Secondary veg to NV
        band[band == 6] = 20  # Grassland to NV
        band[band == 5] = 30  # Planted forest to NV
        band[band == 7] = 40  # Bare ground to NV
        band[band == 9] = 50  # Roads to NV
        band[band == 1] = 60  # Agriculture to NV
        band[band == 3] = 250  # Urban to NV
        band[band == 2] = 255  # Water to NV

        self.init_band = band
        self.grid = self.init_band.copy()

        self.observation_img = np.tile(self.grid.copy(), (3, 1, 1))
        self.observation_img[1,:,:] = np.zeros((self.nrows, self.ncolumns), dtype=np.uint8)
        self.observation_img[2,:,:] = (self.grid != 0)*255

        self.grid_idx = np.arange(np.size(self.grid)).reshape((self.nrows, self.ncolumns))

        # Initialize agent on top right corner.
        self.coordy = self.nrows
        self.coordx = self.ncolumns
        self.movement = (self.nrows-1)*(self.ncolumns-1)

        # Actions vector (possible cells to move to).
        actions_warp = self.grid_idx[self.grid != 0]
        self.actions_warp = actions_warp.flatten()
        # Number of possible indices to which to warp.
        self.n_actions = len(self.actions_warp)

        # Initialize vegetation patch size choice.
        self.vsize = 0

        # Calculate the initial Probability of Connectivity value.
        polys = gs.clump((self.grid==0)*1, self.transform)
        polys_gp = gs.createGDFfromShapes(polys)
        self.extentarea = gs.extent_area(polys_gp, crs=self.crs)
        area_matrix = gs.area_matrix(polys_gp)
        np.savetxt(self.FULLPATH + 'nodes.txt', area_matrix, fmt='%d')
        distances = gs.distance_polys_table(polys_gp)
        distances.to_csv(self.FULLPATH + 'distances.txt', header=None, index=None, sep=' ')
        pc = gs.conefor(area_matrix, path=self.FULLPATH) / self.extentarea
        self.pc_init = pc.copy()
        self.pc_current = self.pc_init.copy()
        self.pc_change = 0
        print("Initial PC index.")
        print(self.pc_init)

        # The action space is a 2D array.
        # One entry selects the location to move to and the other the size of the placed patch.
        self.action_space = gym.spaces.MultiDiscrete(np.array([self.n_actions, self.maxvsize]))

        # The observation space here is an RGB image,
        # Channel 0: the initial landscape grid of pixel costs.
        # Channel 1: the vegetation patches placed by the agent up to a time-step.
        # Channel 2: the remaining budget only on pixels that can still be transformed (non-natural).
        # Everything is normalized between 0 and 255, the standard for RGB images in SB3.
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(3,
                                                       self.nrows,
                                                       self.ncolumns),
                                                dtype=np.uint8)


    def reset(self, seed = None):
        super().reset(seed=seed)

        # Reset to initial values for next training episode.
        self.budget = self.budget_init
        self.grid = self.init_band.copy()
        self.observation_img = np.tile(self.grid.copy(), (3, 1, 1))
        self.observation_img[1, :, :] = np.zeros((self.nrows, self.ncolumns), dtype=np.uint8)
        self.observation_img[2, :, :] = (self.grid != 0)*255
        self.coordy = self.nrows
        self.coordx = self.ncolumns
        self.pc_current = self.pc_init.copy()
        self.pc_change = 0

        observation = self._get_obs()
        info = {}

        return observation, info

    def _get_obs(self):
        # Return observation in the format of self.observation_space.
        return self.observation_img

    def step(self, action):
        # Agents movement
        step = action
        self.movement = step[0]
        self.vsize = step[1]

        selected_idx = self.actions_warp[self.movement]

        y, x = gs.numpy_coordinates(self.grid_idx, selected_idx)

        self.coordy = int(y)
        self.coordx = int(x)

        max_y = int(np.min(np.array([y + self.minvsize + self.vsize, self.nrows])))
        max_x = int(np.min(np.array([x + self.minvsize + self.vsize, self.ncolumns])))

        cost = np.sum(self.grid[y:(max_y), x:(max_x)])

        terminated = False
        truncated = False
        reward = 0
        info = {}

        # Add vegetation patch.
        if cost > 0 and self.budget > 0:

            self.grid[y:max_y, x:max_x] = 0

            self.budget = self.budget - cost

            # Set second band as restored habitat patches (here none).
            self.observation_img[1, :, :] = (self.grid != self.init_band)*255

            # Set third band as remaining budget.
            self.observation_img[2, :, :] = (self.grid != 0)*scale(self.budget, 0, self.budget_init, 0, 255)

            polys = gs.clump((self.grid==0)*1, self.transform)
            polys_gp = gs.createGDFfromShapes(polys)
            area_matrix = gs.area_matrix(polys_gp)
            np.savetxt(self.FULLPATH + 'nodes.txt', area_matrix, fmt='%d')
            distances = gs.distance_polys_table(polys_gp)
            distances.to_csv(self.FULLPATH + 'distances.txt', header=None, index=None, sep=' ')
            pc = gs.conefor(area_matrix, path=self.FULLPATH)/self.extentarea

            self.pc_change = np.around(((pc - self.pc_current)/self.pc_current)*100,4)

            self.pc_current = pc.copy()

        elif cost == 0 and self.budget > 0:

            self.pc_change = 0

        else:
            self.pc_change = np.around(((self.pc_current - self.pc_init) / self.pc_init) * 100, 4)
            terminated = True

        reward = self.pc_change

        return self._get_obs(), reward, terminated, truncated, {}

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

        veg_ind = (self.grid == 0)
        secondary_ind = (self.grid == 10)
        grassland_ind = (self.grid == 20)
        planted_ind = (self.grid == 30)
        bareground_ind = (self.grid == 40)
        roads_ind = (self.grid == 50)
        agriculture_ind = (self.grid == 60)
        urban_ind = (self.grid == 250)
        water_ind = (self.grid == 255)

        # Create color array for landscape plotting.
        Color_array = np.zeros((self.nrows, self.ncolumns, 3), dtype=np.uint8)

        y, x = gs.numpy_coordinates(self.grid_idx, self.movement)

        self.coordy = int(y)
        self.coordx = int(x)

        max_y = int(np.min(np.array([y + self.minvsize + self.vsize, self.nrows])))
        max_x = int(np.min(np.array([x + self.minvsize + self.vsize, self.ncolumns])))

        Color_array[y:max_y, x:max_x] = np.array([16, 255, 202])

        # Assign colors of lc classes.
        Color_array[veg_ind, :] = np.array([6, 100, 26])  # dark green forest
        Color_array[secondary_ind, :] = np.array([145, 171, 19])  # olive green
        Color_array[grassland_ind, :] = np.array([26, 220, 16])  # bright green
        Color_array[planted_ind, :] = np.array([131, 198, 64])  # bright green
        Color_array[bareground_ind, :] = np.array([77, 44, 0])  # brown
        Color_array[roads_ind, :] = np.array([0, 0, 0])  # black
        Color_array[agriculture_ind, :] = np.array([ 234, 156, 29])  # orange
        Color_array[urban_ind, :] = np.array([ 252, 0, 0])  # red
        Color_array[water_ind, :] = np.array([16, 25, 202])  # blue

        # Plot.
        if plot_inline:
            fig = plt.figure()
            plt.axis('off')
            plt.imshow(Color_array, interpolation='nearest')
            plt.show()

        current_grid = (self.grid != self.init_band)*1
        return Color_array, current_grid

