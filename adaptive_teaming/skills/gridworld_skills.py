from minigrid.core.actions import Actions


class PickPlaceSkill:
    def __init__(self, plans):
        # one plan for each goal
        self.plans = plans

    def step(self, env, pref_params, obs):
        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }

        # TODO: choose a plan based on the most likely goal

        total_rew = 0
        gamma, discount = 1.0, 1.0
        for action in self.plan:
            if (action not in key_to_action):
                break
            action = key_to_action[action]
            obs, rew, term, trunc, info = env.step(action)
            total_rew += discount * rew
            discount *= gamma
            done = term or trunc
            if done: break

        return obs, total_rew, done, info
