import logging
import os
from os.path import join

from adaptive_teaming.env.interaction_env import InteractionEnv
from adaptive_teaming.utils.collect_demos import collect_demo_in_gridworld
from adaptive_teaming.utils.utils import pkl_dump, pkl_load
import hydra
from hydra.utils import to_absolute_path
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_env(env_name, **kwargs):
    if env_name == "gridworld":
        from adaptive_teaming.env import GridWorld
        from adaptive_teaming.utils.object import Fork, Mug

        # map_config = {
            # "agent_position": (0, 0),  # The agent's start position
            # "dimensions": [20, 20],
            # "possible_goals": {"G1": (4, 10), "G2": (4, 0)},
        # }

        # mug = Mug()
        # fork = Fork()
        # object_list = [mug, fork]
        # reward = {
            # "target_object": fork,
            # "target_goal": "g1",
        # }

        # env = GridWorld(map_config=map_config,
                        # objects=object_list, reward_dict=reward)
        env = GridWorld(render_mode=kwargs["render_mode"])

    else:
        raise ValueError(f"Unknown environment: {env_name}")

    return env


def make_belief_estimator(cfg, env, task_seq):
    if cfg.env == "gridworld":
        from adaptive_teaming.planner import GridWorldBeliefEstimator

        return GridWorldBeliefEstimator(env, task_seq)


def make_planner(cfg):
    planner_cfg = cfg[cfg.planner]
    if cfg.planner == "fc_mip_planner":
        from adaptive_teaming.planner import FacilityLocationPlanner

        planner = FacilityLocationPlanner(cfg)
    elif cfg.planner == "fc_greedy_planner":
        planner = FacilityLocationGreedyPlanner(cfg)
    elif cfg.planner == "confidence_based_planner":
        from adaptive_teaming.planner import ConfidenceBasedPlanner

        planner = ConfidenceBasedPlanner(planner_cfg, cfg.cost_cfg)
    else:
        raise ValueError(f"Unknown planner: {cfg.planner}")

    return planner


@hydra.main(
    config_path="../cfg", config_name="run_interaction_planner", version_base="1.1"
)
def main(cfg):
    logger.info(f"Output directory: {os.getcwd()}")

    env = make_env(cfg.env, **cfg[cfg.env], render_mode="human" if cfg.render else "none")
    env.reset()
    if cfg.collect_demo:
        demo = collect_demo_in_gridworld(env)
        pkl_dump(demo, f"{cfg.env}_demo.pkl")
    else:
        demo = pkl_load(join(cfg.data_dir, f"{cfg.env}_demo.pkl"))

    from adaptive_teaming.skills.gridworld_skills import PickPlaceSkill
    skill = PickPlaceSkill(demo)

    __import__('ipdb').set_trace()
    interaction_env = InteractionEnv(cfg.human_model, env)

    # tasks
    task_seq = [
        {"obj_type": "Key", "obj_color": "red", "obj_scale": 1,},
        {"obj_type": "Key", "obj_color": "blue", "obj_scale": 1,},
        {"obj_type": "Box", "obj_color": "red", "obj_scale": 1,},
        {"obj_type": "Key", "obj_color": "green", "obj_scale": 1,},
        {"obj_type": "Ball", "obj_color": "yellow", "obj_scale": 1,},
    ]

    pref_params = {
        "G1": {"shape": "s", "color": "gray", "size": 2000, "pos": [-1, 1]},
        "G2": {"shape": "s", "color": "gray", "size": 2000, "pos": [1, -1]},
    }

    for task in task_seq:
        env.reset_to_state(task)
        for _ in range(10): env.render()

    # squares together and circles together
    hum_prefs = ["G1", "G1", "G2", "G2", "G2"]
    belief_estimator = make_belief_estimator(cfg, env, task_seq)
    planner = make_planner(cfg)

    __import__("ipdb").set_trace()

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(1, 2, 1)

    fig1, ax1 = vis_world({"tasks": task_seq, "prefs": pref_params})

    pref_beliefs = [
        np.ones(len(pref_params)) / len(pref_params) for _ in range(len(task_seq))
    ]

    fig2, ax2 = vis_pref_beliefs(pref_beliefs)

    # plt.show()

    # init with a base skill
    skills = [
        {
            "policy": None,
            "train_task": {"shape": "s", "color": "red", "size": 300, "pos": [-1, -1]},
        }
    ]

    # GREEDY Planner
    # ---------------
    if planner_type == "greedy":
        total_cost = 0
        for i in range(len(task_seq)):
            logger.debug(f"Planning for task-{i}")
            tasks = task_seq[i:]

            # options
            # hum
            # -----
            c_hum = cost_cfg["hum"]
            logger.debug(f"  Total HUM cost: {c_hum}")

            # demo
            # -----
            c_demo = cost_cfg["demo"]

            logger.debug(f"  Total DEMO cost: {c_demo}")

            # rob
            # -----
            # pref_belief = pref_beliefs[i]
            # safety cost
            prob_fail = 1 - skill_library_precond(skills, tasks[0])
            c_safe = prob_fail * cost_cfg["fail_cost"]
            c_pref = np.mean(
                [
                    (1 - prob_pref) * cost_cfg["pref_cost"]
                    for prob_pref in pref_beliefs[i]
                ]
            )
            c_rob = cost_cfg["rob"] + c_safe + c_pref

            logger.debug(
                f"  Total ROB cost: {c_rob}, c_safe: {c_safe}, c_pref: {c_pref}"
            )

            # pref
            # -----
            # TODO

            best_action = "hum"
            best_cost = c_hum
            if c_demo < best_cost:
                best_cost = c_demo
                best_action = "demo"

            if c_rob < best_cost:
                best_cost = c_rob
                best_action = "rob"

            logger.debug(
                f"  Best action: {best_action}, Best cost: {best_cost}")

            # simulate the action and get the *actual* cost
            if best_action == "hum":
                total_cost += cost_cfg["hum"]
            elif best_action == "rob":
                prob_fail = 1 - skill_library_precond(skills, tasks[0])
                c_safe = prob_fail * cost_cfg["fail_cost"]
                c_pref = np.mean(
                    [
                        (1 - prob_pref) * cost_cfg["pref_cost"]
                        for prob_pref in pref_beliefs[i]
                    ]
                )
                c_rob = cost_cfg["rob"] + c_safe + c_pref
            elif best_action == "demo":
                skills.append({"policy": None, "train_task": tasks[0]})
            else:
                raise ValueError(f"Unknown action: {best_action}")

        logger.info(f"Total cost: {total_cost}")

    elif planner_type == "informed_greedy":
        # average over all future tasks
        total_cost = 0
        for i in range(len(task_seq)):
            logger.debug(f"Planning for task-{i}")
            tasks = task_seq[i:]

            # options
            # hum
            # -----
            c_hum = cost_cfg["hum"]
            logger.debug(f"  Total HUM cost: {c_hum}")

            # demo
            # -----
            c_demo = cost_cfg["demo"]
            # reduction in future rob costs
            if len(tasks) > 1:
                curr_total_fail_cost = (
                    np.array(
                        [1 - skill_library_precond(skills, task)
                         for task in tasks[1:]]
                    )
                    * cost_cfg["fail_cost"]
                )
                pot_skills = skills + \
                    [{"policy": None, "train_task": tasks[0]}]
                pot_total_fail_cost = (
                    np.array(
                        [
                            1 - skill_library_precond(pot_skills, task)
                            for task in tasks[1:]
                        ]
                    )
                    * cost_cfg["fail_cost"]
                )
                improvement = curr_total_fail_cost - pot_total_fail_cost
                logger.debug(
                    f"  Improvement in future fail costs: {improvement}")
                improvement = np.mean(improvement)
                c_demo -= improvement

            logger.debug(f"  Total DEMO cost: {c_demo}")

            # rob
            # -----
            # pref_belief = pref_beliefs[i]
            # safety cost
            prob_fail = 1 - skill_library_precond(skills, tasks[0])
            c_safe = prob_fail * cost_cfg["fail_cost"]
            c_pref = np.mean(
                [
                    (1 - prob_pref) * cost_cfg["pref_cost"]
                    for prob_pref in pref_beliefs[i]
                ]
            )
            c_rob = cost_cfg["rob"] + c_safe + c_pref

            logger.debug(
                f"  Total ROB cost: {c_rob}, c_safe: {c_safe}, c_pref: {c_pref}"
            )

            # pref
            # -----
            # TODO

            best_action = "hum"
            best_cost = c_hum
            if c_demo < best_cost:
                best_cost = c_demo
                best_action = "demo"

            if c_rob < best_cost:
                best_cost = c_rob
                best_action = "rob"

            logger.debug(
                f"  Best action: {best_action}, Best cost: {best_cost}")

            # simulate the action and get the *actual* cost
            if best_action == "hum":
                total_cost += cost_cfg["hum"]
                # update pref belief
                # rob observes hum pref
                hum_pref = hum_prefs[i]
                # update pref beliefs
                for j, pref_belief in enumerate(pref_beliefs[i:]):
                    # belief estimator assumes shape is important
                    if tasks[j]["shape"] == tasks[0]["shape"]:
                        if hum_pref == "G1":
                            pref_belief[0] = 1
                        elif hum_pref == "G2":
                            pref_belief[1] = 1
                        else:
                            raise ValueError(f"Unknown preference: {hum_pref}")
                vis_pref_beliefs(pref_beliefs)
                plt.show()

            elif best_action == "rob":
                prob_fail = 1 - skill_library_precond(skills, tasks[0])
                c_safe = prob_fail * cost_cfg["fail_cost"]
                c_pref = np.mean(
                    [
                        (1 - prob_pref) * cost_cfg["pref_cost"]
                        for prob_pref in pref_beliefs[i]
                    ]
                )
                c_rob = cost_cfg["rob"] + c_safe + c_pref
            elif best_action == "demo":
                skills.append({"policy": None, "train_task": tasks[0]})
                # TODO update pref belief
            else:
                raise ValueError(f"Unknown action: {best_action}")

        logger.info(f"Total cost: {total_cost}")

    elif planner_type == "facility_location":
        # raise NotImplementedError

        # __import__('ipdb').set_trace()

        # preference parameters

        planner = FacilityLocationPlanner(cost_cfg)
        # task_seq = np.random.rand(10, 2)
        task_seq = np.random.normal(0, 1, (10, 2))

        def task_similarity_fn(x, y):
            return np.exp(-np.linalg.norm(x - y))

        pref_similarity_fn = task_similarity_fn
        planner.plan(task_seq, task_similarity_fn, pref_similarity_fn)


if __name__ == "__main__":
    main()
