import unittest
from adaptive_teaming.planner import GridWorldBeliefEstimator


def make_env():
    from adaptive_teaming.env import GridWorld
    from adaptive_teaming.utils.object import Fork, Mug

    map_config = {
        "agent_position": (0, 0),  # The agent's start position
        "dimensions": [20, 20],
        "possible_goals": {"G1": (4, 10), "G2": (4, 0)},
    }

    mug = Mug()
    fork = Fork()
    object_list = [mug, fork]
    reward = {
        "target_object": fork,
        "target_goal": "g1",
    }

    env = GridWorld(map_config=map_config,
                    objects=object_list, reward_dict=reward)

    return env

class TestGridBeliefEstimator(unittest.TestCase):

    def test_belief_update(self):
        env = make_env()
        # tasks
        task_seq = [
            {"agent_posisiont": (0, 0), "dimensionsshape": "s", "color": "red", "size": 300, "pos": [0, 0]},
            {"shape": "s", "color": "blue", "size": 300, "pos": [0, 1]},
            {"shape": "o", "color": "red", "size": 300, "pos": [1, 0]},
            {"shape": "o", "color": "blue", "size": 300, "pos": [1, 0.5]},
            {"shape": "o", "color": "green", "size": 300, "pos": [1.5, 0.5]},
        ]

        pref_params = {
            "G1": {"shape": "s", "color": "gray", "size": 2000, "pos": [-1, 1]},
            "G2": {"shape": "s", "color": "gray", "size": 2000, "pos": [1, -1]},
        }


        # squares together and circles together
        hum_prefs = ["G1", "G1", "G2", "G2", "G2"]
        belief_estimator = GridWorldBeliefEstimator(env, task_seq)
        print("Prior belief: ", belief_estimator.beliefs)
        belief_estimator.update_beliefs(0, {"pref": "G1"})
        print("Posterior belief: ", belief_estimator.beliefs)
        return True

if __name__ == '__main__':
    unittest.main()

