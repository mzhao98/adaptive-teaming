import unittest
import matplotlib.pyplot as plt
from adaptive_teaming.env import GridWorld
from adaptive_teaming.utils.object import Fork, Mug
from minigrid.core.grid import Grid

class TestGridBeliefEstimator(unittest.TestCase):

    def test_map_init(self):
        return True

class TestGridWorld(unittest.TestCase):

    def test_map_init(self):
        env = GridWorld(render_mode="human")
        obs, _ = env.reset()
        # grid, _ = Grid.decode(obs["image"])
        # plt.imshow(grid)
        # plt.show()
        __import__('ipdb').set_trace()

        return True

if __name__ == '__main__':
    unittest.main()
