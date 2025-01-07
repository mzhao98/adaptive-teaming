import logging
import os
from os.path import join

import hydra
import numpy as np
from adaptive_teaming.utils.collect_demos import collect_demo_in_gridworld
from adaptive_teaming.utils.utils import pkl_dump, pkl_load
from hydra.utils import to_absolute_path
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_env(env_name, cfg):
    if env_name == "gridworld":
        from adaptive_teaming.envs import GridWorld

        env = GridWorld(render_mode="human" if cfg.render else "none")

    elif env_name == "pick_place":

        from adaptive_teaming.envs.pick_place import PickPlaceEnv

        # env_args = dict(
            # env_name=env_name,
            # robots="panda",
            # has_renderer=cfg.render,
            # has_offscreen_renderer=True,
            # ignore_done=True,
            # use_camera_obs=False,
            # control_freq=20,
            # controller_configs=load_controller_config(default_controller="OSC_POSE"),
        # )
        env = PickPlaceEnv(has_renderer=cfg.render)

    else:
        raise ValueError(f"Unknown environment: {env_name}")

    return env


def make_belief_estimator(cfg, env, task_seq):
    if cfg.env == "gridworld":
        from adaptive_teaming.planner import GridWorldBeliefEstimator

        return GridWorldBeliefEstimator(env, task_seq)


def make_planner(interaction_env, belief_estimator, cfg):
    if cfg.planner == "fc_mip_planner":
        from adaptive_teaming.planner import FacilityLocationPlanner

        planner_cfg = cfg[cfg.planner]
        planner = FacilityLocationPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif cfg.planner == "fc_pref_planner":
        from adaptive_teaming.planner import FacilityLocationPrefPlanner

        planner_cfg = cfg[cfg.planner]
        planner = FacilityLocationPrefPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif cfg.planner == "confidence_based_planner":
        from adaptive_teaming.planner import \
            ConfidenceBasedFacilityLocationPlanner

        planner_cfg = cfg[cfg.planner]
        planner = ConfidenceBasedFacilityLocationPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif cfg.planner == "always_human":
        from adaptive_teaming.planner import AlwaysHuman

        planner = AlwaysHuman(
            interaction_env, belief_estimator, None, cfg.cost_cfg)
    elif cfg.planner == "always_learn":
        from adaptive_teaming.planner import AlwaysLearn

        planner = AlwaysLearn(
            interaction_env, belief_estimator, None, cfg.cost_cfg)
    elif cfg.planner == "learn_then_robot":
        from adaptive_teaming.planner import LearnThenRobot

        planner = LearnThenRobot(
            interaction_env, belief_estimator, None, cfg.cost_cfg)
    elif cfg.planner == "fixed_planner":
        from adaptive_teaming.planner import FixedPlanner

        planner_cfg = cfg[cfg.planner]
        planner = FixedPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    else:
        raise ValueError(f"Unknown planner: {cfg.planner}")

    return planner


def init_domain(cfg):
    if cfg.collect_demo:
        env = make_env(cfg.env, cfg)
        env.reset()
        if cfg.env == "gridworld":
            demo_tasks = [
                {
                    "obj_type": "Key",
                    "obj_color": "red",
                    "obj_scale": 1,
                    "position": (3, 1),
                },
                {
                    "obj_type": "Key",
                    "obj_color": "green",
                    "obj_scale": 1,
                    "position": (3, 1),
                },
            ]

        else:
            raise ValueError(f"Unknown environment: {cfg.env}")

        demos = collect_demo_in_gridworld(env, demo_tasks)
        pkl_dump(demos, f"{cfg.env}_demos.pkl")
    else:
        if cfg.env == "gridworld":
            data_files = join(cfg.data_dir, f"{cfg.env}_demos.pkl")
            data_path = (
                os.path.dirname(os.path.realpath(__file__))
                + "/../"
                + cfg.data_dir
            )

            if os.path.isdir(data_path):
                try:
                    demos = pkl_load(data_files)
                except FileNotFoundError:
                    logger.error(f"There's no data in {cfg.data_dir}")
                    exit(f"\nThere's no data in {cfg.data_dir}. Terminating.\n")
            else:
                logger.error(f"The path to the data ({data_path}) is erroneous.")
                exit(f"See if {cfg.data_dir} exists.")
        else:
            logger.warning("Not loading demos from file.")

    env = make_env(cfg.env, cfg)
    env.reset()
    # __import__('ipdb').set_trace()
    if cfg.env == "gridworld":
        from adaptive_teaming.envs import GridWorldInteractionEnv

        interaction_env = GridWorldInteractionEnv(
            env, cfg.human_model, cfg.cost_cfg)
    else:
        raise ValueError(f"Unknown environment: {cfg.env}")

    interaction_env.load_human_demos(demos)

    return env, interaction_env


def generate_tasks(cfg):
    if cfg.env == "gridworld":
        task_seq = [
            {
                "obj_type": "Key",
                "obj_color": "red",
                "obj_scale": 1,
                "position": (3, 1),
            },
            {
                "obj_type": "Key",
                "obj_color": "red",
                "obj_scale": 1,
                "position": (3, 1),
            },
            {
                "obj_type": "Key",
                "obj_color": "blue",
                "obj_scale": 1,
                "position": (3, 1),
            },
            {
                "obj_type": "Box",
                "obj_color": "red",
                "obj_scale": 1,
                "position": (3, 2),
            },
            {
                "obj_type": "Key",
                "obj_color": "green",
                "obj_scale": 1,
                "position": (3, 1),
            },
            {
                "obj_type": "Ball",
                "obj_color": "yellow",
                "obj_scale": 1,
                "position": (3, 3),
            },
            {
                "obj_type": "Key",
                "obj_color": "green",
                "obj_scale": 1,
                "position": (3, 1),
            },
            {
                "obj_type": "Key",
                "obj_color": "green",
                "obj_scale": 1,
                "position": (3, 1),
            },
            {
                "obj_type": "Ball",
                "obj_color": "green",
                "obj_scale": 1,
                "position": (3, 3),
            },
            {
                "obj_type": "Key",
                "obj_color": "red",
                "obj_scale": 1,
                "position": (3, 1),
            },
            {
                "obj_type": "Key",
                "obj_color": "green",
                "obj_scale": 1,
                "position": (3, 1),
            },
            {
                "obj_type": "Key",
                "obj_color": "green",
                "obj_scale": 1,
                "position": (3, 1),
            },
        ]
    else:
        raise ValueError(f"Unknown environment: {cfg.env}")
    return task_seq


def vis_tasks(env, task_seq):
    vis_together = False
    if vis_together:
        from adaptive_teaming.env.gridworld import OBJECT_TYPES

        # visualize in the same env
        # env.render_mode = "rgb_array"
        for task in task_seq:
            # env.reset_to_state(task)
            obj = OBJECT_TYPES[task["obj_type"]](task["obj_color"])
            env.objects.append({"object": obj, "position": task["position"]})

        env.reset()
        for _ in range(10):
            env.render()
        __import__("ipdb").set_trace()
    else:
        for task in task_seq:
            env.reset_to_state(task)
            for _ in range(10):
                env.render()


def save_task_imgs(env, task_seq):
    render_mode = env.render_mode
    env.render_mode = "rgb_array"
    os.makedirs("tasks")
    for i, task in enumerate(task_seq):
        env.reset_to_state(task)
        img = Image.fromarray(env.render())
        img.save(f"tasks/task_{i}.png")
    env.render_mode = render_mode


@hydra.main(
    config_path="../cfg", config_name="run_interaction_planner", version_base="1.1"
)
def main(cfg):
    logger.info(f"Output directory: {os.getcwd()}")

    env, interaction_env = init_domain(cfg)
    env.render()

    # __import__("ipdb").set_trace()
    task_seq = generate_tasks(cfg)
    if cfg.vis_tasks:
        vis_tasks(env, task_seq)
    save_task_imgs(env, task_seq)

    interaction_env.reset(task_seq)

    belief_estimator = make_belief_estimator(cfg, env, task_seq)
    planner = make_planner(interaction_env, belief_estimator, cfg)
    planner.rollout_interaction(
        task_seq, interaction_env.task_similarity, interaction_env.pref_similarity
    )

    return

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
