import pygame
import unittest

import matplotlib.pyplot as plt
from adaptive_teaming.env import GridWorld
from adaptive_teaming.utils.object import Fork, Mug
from minigrid.core.grid import Grid
from minigrid.manual_control import ManualControl


class TestGridBeliefEstimator(unittest.TestCase):

    def test_map_init(self):
        return True


class TestGridWorld(unittest.TestCase):

    def test_map_init(self):
        print("Testing map init")
        env = GridWorld(render_mode="human")
        obs, _ = env.reset()
        # grid, _ = Grid.decode(obs["image"])
        # plt.imshow(grid)
        # plt.show()

        return True

    def test_manual_control(self):
        print("Testing manual agent control")
        print("Commands:")
        print(
            """ 'left': Actions.left,
            'right': Actions.right,
            'up': Actions.forward,
            'space': Actions.toggle,
            'pageup': Actions.pickup,
            'pagedown': Actions.drop,
            'tab': Actions.pickup,
            'left shift': Actions.drop,
            'enter': Actions.done,
             """
        )

        env = GridWorld(render_mode="human")
        obs, _ = env.reset()
        manual_control = ManualControl(env)
        try:
            manual_control.start()
        except:
            pass

    def test_env_reset_to_state(self):
        print("Testing reset to env state")
        env = GridWorld(render_mode="human")
        state = {"obj_type": "Key", 
                 "obj_color": "red"}
        obs, _ = env.reset_to_state(state)
        print("You should see a red key in the grid")
        for _ in range(20):
            env.render()
        state = {"obj_type": "Box", 
                 "obj_color": "blue"}
        obs, _ = env.reset_to_state(state)
        print("You should see a blue box in the grid")
        for _ in range(20):
            env.render()
        return True


if __name__ == "__main__":
    unittest.main()
