import logging
import os
import pdb
import random
import time
from os.path import join

import hydra
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from adaptive_teaming.planner import TaskRelevantInfoGainPlanner
from adaptive_teaming.skills.pick_place_skills import PickPlaceExpertSkill
from adaptive_teaming.utils.collect_demos import collect_demo_in_gridworld
from adaptive_teaming.utils.utils import pkl_dump, pkl_load
from hydra.utils import to_absolute_path
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_env(env_name, cfg):
    print(f"Creating environment: {env_name}")
    if env_name == "gridworld":
        from adaptive_teaming.envs import GridWorld

        env = GridWorld(render_mode="human" if cfg.render else "none")

    elif env_name == "med_gridworld":
        from adaptive_teaming.envs import MediumGridWorld

        env = MediumGridWorld(render_mode="human" if cfg.render else "none")

    elif env_name == "pick_place":

        from adaptive_teaming.envs.pick_place import PickPlaceSingle

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
        env = PickPlaceSingle(
            has_renderer=cfg.render,
            render_camera="frontview",
        )

    else:
        raise ValueError(f"Unknown environment: {env_name}")

    return env


def make_belief_estimator(cfg, env, task_seq):
    if cfg.env == "gridworld":
        from adaptive_teaming.planner import GridWorldBeliefEstimator

        return GridWorldBeliefEstimator(env, task_seq)

    elif cfg.env == "med_gridworld":
        from adaptive_teaming.planner import GridWorldBeliefEstimator

        return GridWorldBeliefEstimator(env, task_seq)

    elif cfg.env == "pick_place":
        from adaptive_teaming.planner import PickPlaceBeliefEstimator

        return PickPlaceBeliefEstimator(env, task_seq)


def make_planner(interaction_env, belief_estimator, select_planner, cfg):
    if select_planner == "fc_mip_planner":
        from adaptive_teaming.planner import FacilityLocationPlanner

        planner_cfg = cfg[select_planner]
        planner = FacilityLocationPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif select_planner == "info_gain_planner":
        from adaptive_teaming.planner import InfoGainPlanner

        planner_cfg = cfg[select_planner]
        planner = InfoGainPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif select_planner == "task_info_gain_planner":
        from adaptive_teaming.planner import TaskRelevantInfoGainPlanner

        planner_cfg = cfg[select_planner]
        planner = TaskRelevantInfoGainPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif select_planner == "fc_greedy_planner":
        from adaptive_teaming.planner import FacilityLocationGreedyPlanner

        planner_cfg = cfg[select_planner]
        planner = FacilityLocationGreedyPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif select_planner == "fc_pref_planner":
        from adaptive_teaming.planner import FacilityLocationPrefPlanner

        if cfg.fc_planner == "mip":
            from adaptive_teaming.planner import FacilityLocationPlanner

            fc_planner = FacilityLocationPlanner(
                interaction_env, belief_estimator, cfg["fc_mip_planner"], cfg.cost_cfg
            )
        elif cfg.fc_planner == "greedy":
            from adaptive_teaming.planner import FacilityLocationGreedyPlanner

            fc_planner = FacilityLocationGreedyPlanner(
                interaction_env,
                belief_estimator,
                cfg["fc_greedy_planner"],
                cfg.cost_cfg,
            )
        else:
            raise ValueError(
                f"Unknown facility location planner: {cfg.fc_planner}")

        planner_cfg = cfg[select_planner]
        planner = FacilityLocationPrefPlanner(
            fc_planner, interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif select_planner == "confidence_based_planner":
        from adaptive_teaming.planner import \
            ConfidenceBasedFacilityLocationPlanner

        planner_cfg = cfg[select_planner]
        planner = ConfidenceBasedFacilityLocationPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif select_planner == "always_human":
        from adaptive_teaming.planner import AlwaysHuman

        planner = AlwaysHuman(
            interaction_env, belief_estimator, None, cfg.cost_cfg)
    elif select_planner == "always_learn":
        from adaptive_teaming.planner import AlwaysLearn

        planner = AlwaysLearn(
            interaction_env, belief_estimator, None, cfg.cost_cfg)
    elif select_planner == "learn_then_robot":
        from adaptive_teaming.planner import LearnThenRobot

        planner = LearnThenRobot(
            interaction_env, belief_estimator, None, cfg.cost_cfg)
    elif select_planner == "fixed_planner":
        from adaptive_teaming.planner import FixedPlanner

        planner_cfg = cfg[select_planner]
        planner = FixedPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    else:
        raise ValueError(f"Unknown planner: {select_planner}")

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
    # else:
        # if cfg.env == "gridworld":
        # data_files = join(cfg.data_dir, f"{cfg.env}_demos.pkl")
        # data_path = os.path.dirname(
        #     os.path.realpath(__file__)) + "/../" + cfg.data_dir
        #
        # if os.path.isdir(data_path):
        #     try:
        #         demos = pkl_load(data_files)
        #     except FileNotFoundError:
        #         logger.error(f"There's no data in {cfg.data_dir}")
        #         exit(f"\nThere's no data in {cfg.data_dir}. Terminating.\n")
        # else:
        #     logger.error(f"The path to the data ({data_path}) is erroneous.")
        #     exit(f"See if {cfg.data_dir} exists.")
        # else:
        # logger.warning("Not loading demos from file.")
    env = make_env(cfg.env, cfg)
    env.reset()
    # __import__('ipdb').set_trace()
    if cfg.env == "gridworld":
        from adaptive_teaming.envs import GridWorldInteractionEnv

        interaction_env = GridWorldInteractionEnv(
            env, cfg.human_model, cfg.cost_cfg)

    elif cfg.env == "med_gridworld":
        from adaptive_teaming.envs import GridWorldInteractionEnv

        interaction_env = GridWorldInteractionEnv(
            env, cfg.human_model, cfg.cost_cfg)

    elif cfg.env == "pick_place":
        from adaptive_teaming.envs import PickPlaceInteractionEnv

        interaction_env = PickPlaceInteractionEnv(
            env, cfg.human_model, cfg.cost_cfg)
    else:
        raise ValueError(f"Unknown environment: {cfg.env}")

    # interaction_env.load_human_demos(demos)

    return env, interaction_env


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def randomize_gridworld_task_seq(env, cfg, n_objs=10, seed=42, gridsize=8):
    acceptable_locations = env.get_acceptable_obj_locations()
    if env.agent_start_pos in acceptable_locations:
        acceptable_locations.remove(env.agent_start_pos)

    # set the seed
    # print("seed", seed)
    set_seeds(seed)
    np.random.seed(seed)

    # randomly assign objects to locations
    objs_list = ["Box", "Ball"]
    colors_list = ["red", "green", "blue", "yellow"]

    # create a list of n_objs objects
    objs_present = []
    for i in range(n_objs):
        objs_present.append(
            (random.choice(objs_list), random.choice(colors_list)))

    # randomly assign locations to each object type and color
    unique_objs = list(set(objs_present))
    print(f"Unique objects: {unique_objs}")

    # randomly assign locations to each object type and color
    location_assignments = np.random.choice(
        len(acceptable_locations), len(unique_objs), replace=False
    )

    # create dictionary of object type and color to location
    obj_to_location = {}
    for i, obj in enumerate(unique_objs):
        obj_to_location[obj] = acceptable_locations[location_assignments[i]]

    # create a task sequence
    task_seq = []
    for i, obj in enumerate(objs_present):
        task = {
            "obj_type": obj[0],
            "obj_color": obj[1],
            "obj_scale": 1,
            "position": obj_to_location[(obj[0], obj[1])],
        }
        task_seq.append(task)

    return [task_seq]


def generate_task_seqs(cfg, n_seqs=1, seed=42):
    # set the seed
    set_seeds(seed)
    np.random.seed(seed)

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
                "obj_color": "yellow",
                "obj_scale": 1,
                "position": (3, 1),
            },
            # {
            #     "obj_type": "Key",
            #     "obj_color": "blue",
            #     "obj_scale": 1,
            #     "position": (3, 1),
            # },
            # {
            #     "obj_type": "Box",
            #     "obj_color": "red",
            #     "obj_scale": 1,
            #     "position": (3, 2),
            # },
            # {
            #     "obj_type": "Key",
            #     "obj_color": "green",
            #     "obj_scale": 1,
            #     "position": (3, 1),
            # },
            # {
            #     "obj_type": "Ball",
            #     "obj_color": "yellow",
            #     "obj_scale": 1,
            #     "position": (3, 3),
            # },
            # {
            #     "obj_type": "Key",
            #     "obj_color": "green",
            #     "obj_scale": 1,
            #     "position": (3, 1),
            # },
            # {
            #     "obj_type": "Key",
            #     "obj_color": "green",
            #     "obj_scale": 1,
            #     "position": (3, 1),
            # },
            # {
            #     "obj_type": "Ball",
            #     "obj_color": "green",
            #     "obj_scale": 1,
            #     "position": (3, 3),
            # },
            # {
            #     "obj_type": "Key",
            #     "obj_color": "red",
            #     "obj_scale": 1,
            #     "position": (3, 1),
            # },
            # {
            #     "obj_type": "Key",
            #     "obj_color": "green",
            #     "obj_scale": 1,
            #     "position": (3, 1),
            # },
            # {
            #     "obj_type": "Key",
            #     "obj_color": "green",
            #     "obj_scale": 1,
            #     "position": (3, 1),
            # },
        ]
        task_seqs = [task_seq]
    elif cfg.env == "pick_place":
        if cfg.use_sim_for_task_gen:
            from adaptive_teaming.envs.pick_place import PickPlaceTaskSeqGen

            env = PickPlaceTaskSeqGen(cfg.task_seq, has_renderer=cfg.render)
            task_seqs = env.generate_task_seq(n_seqs)
        else:
            env = make_env(cfg.env, cfg)
            task_seqs = env.generate_task_seq(n_seqs, cfg.task_seq.num_tasks)

    else:
        raise ValueError(f"Unknown environment: {cfg.env}")
    return task_seqs


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


def save_task_imgs(env, task_seq, seed, select_plannner):
    render_mode = env.render_mode
    env.render_mode = "rgb_array"
    os.makedirs(f"results/{select_plannner}/{seed}/tasks")
    for i, task in enumerate(task_seq):
        env.reset_to_state(task)
        img = Image.fromarray(env.render())
        img.save(f"results/{select_plannner}/{seed}/tasks/task_{i}.png")
    env.render_mode = render_mode


def plot_interaction(objects, actions, placements, current_seed, plannername):
    """
    Plots a bar chart of actions performed for each object.

    Parameters:
        objects (list of str): List of object names.
        actions (list of tuples): List of actions for each object. Each action is a tuple or string.
    """
    fig, ax = plt.subplots(figsize=(12, 2))

    # Define some colors for actions
    action_colors = {
        "ASK_PREF": "#c9b35b",
        "ASK_SKILL": "#9a4f69",
        "ROBOT": "#b39c85",
        "HUMAN": "#69755c",
    }

    # Define action order
    action_order = ["ASK_PREF", "ASK_SKILL", "ROBOT", "HUMAN"]

    # Create bars for each object and its corresponding actions
    for idx, (obj, action, placement) in enumerate(zip(objects, actions, placements)):
        start = idx  # Bar starts at the index
        width = 1  # Width of the bar spans one unit

        if len(action) == 2:
            # Split the bar if multiple actions are present
            half_width = width / 2
            ax.barh(
                y=action_order.index(action[0]),
                width=half_width,
                left=start,
                color=action_colors[action[0]],
                edgecolor="black",
                align="center",
            )
            ax.barh(
                y=action_order.index(action[1]),
                width=half_width,
                left=start + half_width,
                color=action_colors[action[1]],
                edgecolor="black",
                align="center",
            )
            # Add placement text for each action
            ax.text(
                start + 0.25,
                action_order.index(action[0]),
                placement[0],
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                weight="bold",
            )
            ax.text(
                start + 0.75,
                action_order.index(action[1]),
                placement[1],
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                weight="bold",
            )

        else:
            # Single action bar
            ax.barh(
                y=action_order.index(action[0]),
                width=width,
                left=start,
                color=action_colors[action[0]],
                edgecolor="black",
                align="center",
            )

            # Add placement text to the bar
            ax.text(
                start + 0.5,
                action_order.index(action[0]),
                placement[0],
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                weight="bold",
            )

        # Add a dotted vertical line to indicate the start and end of each object
        ax.axvline(x=start, color="gray", linestyle="dotted", linewidth=0.8)

    # Set the x-axis to show object names
    ax.set_xticks(range(len(objects)))
    ax.set_xticklabels(objects, rotation=90)

    # Set the y-axis to show action names
    ax.set_yticks(range(len(action_order)))
    ax.set_yticklabels(action_order)

    # Set the y-axis label and x-axis label
    ax.set_ylabel("Action Name")
    ax.set_xlabel("Object Name")
    # ax.set_title(f"Actions Performed on Objects: {plannername}")

    # Add legend
    # patches = [mpatches.Patch(color=color, label=action) for action, color in action_colors.items()]
    # ax.legend(handles=patches, title="Actions")

    # drop splines on top and right side
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"results/{plannername}/{current_seed}/interaction.png")
    plt.close()


def sample_human_pref(list_of_goals, seed):
    # set the seed
    np.random.seed(seed)

    n_human_prefs = 3
    # create a dictionary of all object, color combinations to random goals
    objs_list = ["key", "box", "ball"]
    colors_list = ["red", "green", "blue", "yellow"]
    all_combinations = [(obj, color)
                        for obj in objs_list for color in colors_list]

    human_pref_1 = {comb: np.random.choice(list_of_goals)
                    for comb in all_combinations}
    # human_pref_2 = {comb: random.choice(list_of_goals)
    #                 for comb in all_combinations}
    # human_pref_3 = {comb: random.choice(list_of_goals)
    #                 for comb in all_combinations}
    #
    # list_of_prefs = [human_pref_1, human_pref_2, human_pref_3]
    return human_pref_1


def compute_ufl_plan_time(cfg, num_objs_min=50, num_objs_max=501):
    assert cfg.env == "pick_place"
    if cfg.fc_planner == "mip":
        cfg.planner = "fc_mip_planner"
    elif cfg.fc_planner == "greedy":
        cfg.planner = "fc_greedy_planner"
    logger.info(f"Computing planning stats for {cfg.planner}")
    stats = {"planning_time": [], "num_objs": []}
    for num_tasks in range(num_objs_min, num_objs_max, 50):
        logger.info("#tasks: %d", num_tasks)
        cfg.task_seq.num_tasks = num_tasks
        env, interaction_env = init_domain(cfg)
        plan_times = []
        task_seqs = generate_task_seqs(cfg, n_seqs=5, seed=cfg.seed)
        for task_seq in task_seqs:
            belief_estimator = make_belief_estimator(cfg, env, task_seq)
            planner = make_planner(interaction_env, belief_estimator, cfg)

            plan_info = planner.compute_planning_time(
                task_seq,
                interaction_env.task_similarity,
                interaction_env.pref_similarity,
            )
            plan_times.append(plan_info["solve_time"])
        stats["planning_time"].append(plan_times)
        stats["num_objs"].append(num_tasks)
        logger.info(f"  Mean Planning Time: {np.mean(plan_times)}")
    pkl_dump(stats, "ufl_planning_time.pkl")


@hydra.main(
    config_path="../cfg", config_name="run_interaction_planner_multi_exp", version_base="1.1"
)
def main(cfg):
    logger.info(f"Output directory: {os.getcwd()}")
    if cfg.compute_ufl_plan_time:
        compute_ufl_plan_time(cfg)
        return

    N_exps = cfg.N_exps
    initial_seed = cfg.seed
    # come up with N_exps different seeds with the initial seed
    np.random.seed(initial_seed)
    seeds = np.random.randint(0, 10000, N_exps)
    planners_to_run = cfg.planner

    result_planner_to_list_rewards = {}

    for select_planner in planners_to_run:
        result_planner_to_list_rewards[select_planner] = []
        # make a folder to house results
        os.makedirs(f"results/{select_planner}", exist_ok=True)

        for i in range(N_exps):
            current_seed = int(seeds[i])
            print(f"Running experiment {i} with seed {current_seed}")
            # make a folder to house seed results
            os.makedirs(f"results/{select_planner}/{current_seed}", exist_ok=True)

            env, interaction_env = init_domain(cfg)
            obs = env.reset()

            start_time = time.time()
            if cfg.env == "gridworld":
                task_seqs = randomize_gridworld_task_seq(
                    env, cfg, n_objs=cfg.task_seq.num_tasks, seed=current_seed)
            elif cfg.env == "med_gridworld":
                task_seqs = randomize_gridworld_task_seq(
                    env, cfg, n_objs=cfg.task_seq.num_tasks, seed=current_seed)
            else:
                task_seqs = generate_task_seqs(cfg, n_seqs=1, seed=current_seed)
            logger.info(f"Time to generate task seqs: {time.time() - start_time}")
            print("task_seqs", task_seqs)

            if cfg.env == "gridworld" or cfg.env == "med_gridworld":
                true_human_pref = sample_human_pref(cfg.gridworld.list_of_goals, seed=current_seed)

            # return
            task_seq = task_seqs[0]

            if cfg.env == "gridworld" and cfg.vis_tasks:
                vis_tasks(env, task_seq)
                save_task_imgs(env, task_seq, current_seed, select_planner)
                # pdb.set_trace()

            elif cfg.env == "med_gridworld" and cfg.vis_tasks:
                vis_tasks(env, task_seq)
                save_task_imgs(env, task_seq, current_seed, select_planner)

            elif cfg.env == "pick_place":
                pass
                # for task in task_seq:
                # env.reset_to_state(task)
                # for _ in range(50):
                # env.render()
            # print("task_seq", task_seq)
            interaction_env.reset(task_seq)
            interaction_env.robot_skills = {}
            if cfg.env == "gridworld" or cfg.env == "med_gridworld":
                interaction_env.set_human_pref(true_human_pref)

            # __import__('ipdb').set_trace()

            belief_estimator = make_belief_estimator(cfg, env, task_seq)
            planner = make_planner(interaction_env, belief_estimator, select_planner, cfg)
            total_rew, resultant_objects, resultant_actions, placements = (
                planner.rollout_interaction(
                    task_seq, interaction_env.task_similarity, interaction_env.pref_similarity
                )
            )
            result_planner_to_list_rewards[select_planner].append(total_rew)
            print(f"Total reward: {total_rew}")
            print(f"Resultant objects: {resultant_objects}")

            # resultant_actions = [[a["action_type"] for a in actions] for actions in resultant_actions]
            print(f"Resultant actions: {resultant_actions}")
            print(f"Placements: {placements}")
            plot_interaction(resultant_objects, resultant_actions,
                             placements, current_seed, select_planner)
            print("total_rew", total_rew)

            # save the results
            results = {
                "total_rew": total_rew,
                "resultant_objects": resultant_objects,
                "resultant_actions": resultant_actions,
                "placements": placements,
            }
            # save to pickle
            pkl_dump(results, f"results/{select_planner}/{current_seed}/results.pkl")
            # save to text
            with open(f"results/{select_planner}/{current_seed}/results.txt", "w") as f:
                f.write(f"Total reward: {total_rew}\n")
                f.write(f"Resultant objects: {resultant_objects}\n")
                f.write(f"Resultant actions: {resultant_actions}\n")
                f.write(f"Placements: {placements}\n")
                f.write(f"Seed: {current_seed}\n")
                f.write(f"Planner: {select_planner}\n")

    print("result_planner_to_list_rewards", result_planner_to_list_rewards)

    return result_planner_to_list_rewards


def plot_reward_for_seed():
    high_cost_human = {'fc_pref_planner': [-850, -1610, -810, -1370, -850, -850, -950, -900, -900, -870, -900, -850, -780, -780, -890, -880, -850, -860, -870, -910, -890, -870, -820, -890, -850, -870, -860, -740, -860, -910], 'task_info_gain_planner': [-1160, -1740, -1010, -990, -1140, -1160, -1160, -1160, -1050, -1010, -1160, -1160, -930, -900, -1140, -1160, -1140, -1050, -1010, -1050, -1140, -1010, -920, -1160, -1030, -1010, -1030, -820, -1030, -1050], 'info_gain_planner': [-1160, -1740, -1010, -990, -1140, -1160, -1160, -1160, -1050, -1010, -1160, -1160, -930, -900, -1140, -1160, -1140, -1050, -1010, -1050, -1140, -1010, -920, -1160, -1030, -1010, -1030, -820, -1030, -1050], 'confidence_based_planner': [-1320, -1710, -840, -1450, -900, -950, -1000, -960, -960, -920, -960, -950, -830, -900, -940, -930, -870, -980, -950, -1360, -910, -920, -860, -960, -870, -950, -930, -790, -900, -960]}

    low_cost_human =  {'fc_pref_planner': [-520, -1320, -440, -480, -480, -1120, -520, -1120, -520, -520, -520, -520, -440, -520, -480, -520, -480, -520, -920, -520, -480, -520, -480, -520, -480, -520, -480, -440, -1080, -520],
                       'task_info_gain_planner': [-520, -1320, -440, -480, -480, -1120, -520, -1120, -520, -520, -520, -520, -440, -520, -480, -520, -480, -520, -920, -520, -480, -520, -480, -520, -480, -520, -480, -440, -1080, -520],
                       'info_gain_planner': [-600, -980, -520, -540, -560, -780, -600, -800, -600, -600, -600, -580, -500, -560, -560, -620, -580, -560, -580, -600, -580, -600, -560, -600, -580, -580, -540, -500, -760, -600],
                       'confidence_based_planner': [-520, -1320, -440, -480, -480, -1120, -520, -1120, -520, -520, -520, -520, -440, -520, -480, -520, -480, -520, -920, -520, -480, -520, -480, -520, -480, -520, -480, -440, -1080, -520]}

    normative_cost_human =  {'fc_pref_planner': [-680, -1880, -640, -650, -680, -920, -760, -720, -720, -670, -720, -720, -640, -690, -720, -670, -630, -1380, -720, -1110, -670, -680, -630, -720, -630, -730, -710, -600, -670, -710],
                            'task_info_gain_planner': [-760, -1960, -620, -690, -690, -960, -760, -760, -760, -760, -760, -760, -620, -760, -690, -760, -690, -1360, -960, -1160, -690, -760, -690, -760, -690, -760, -690, -620, -690, -960],
                            'info_gain_planner': [-760, -1960, -620, -790, -690, -960, -760, -760, -760, -860, -760, -760, -720, -860, -690, -760, -690, -1360, -1060, -1160, -690, -860, -690, -760, -690, -860, -690, -720, -690, -960],
                            'confidence_based_planner': [-730, -1930, -620, -670, -680, -940, -750, -740, -740, -730, -740, -740, -620, -730, -690, -730, -670, -1350, -940, -1140, -680, -730, -670, -740, -670, -740, -690, -610, -680, -940]}
    profiles = ['low_cost', 'med_cost',  'high_cost', ]
    data = {
        'low_cost': low_cost_human,
        'med_cost': normative_cost_human,
        'high_cost': high_cost_human
    }

    algorithms = list(low_cost_human.keys())
    n_algorithms = len(algorithms)
    n_profiles = len(profiles)

    # Calculate means and standard errors for each group
    means = []
    errors = []
    for profile in profiles:
        profile_means = []
        profile_errors = []
        for algo in algorithms:
            values = data[profile][algo]
            profile_means.append(np.mean(values))
            profile_errors.append(np.std(values) / np.sqrt(len(values)))  # Standard error
        means.append(profile_means)
        errors.append(profile_errors)

    # Convert to numpy arrays for easier manipulation
    means = np.array(means)  # Shape: (n_profiles, n_algorithms)
    errors = np.array(errors)

    # Plot grouped bar plot
    x = np.arange(n_profiles)  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 3))

    for i, algo in enumerate(algorithms):
        color_map = {
            'fc_pref_planner': '#E73879',
            'task_info_gain_planner': '#F26B0F',
            'info_gain_planner': '#FCC737',
            'confidence_based_planner': '#91AC8F'
        }
        ax.bar(
            x + i * width,
            means[:, i],
            width,
            label=algo,
            yerr=errors[:, i],
            capsize=5,
            color=color_map.get(algo, 'gray')  # Default to gray if algorithm not in color map
        )


    # Add labels, title, and legend
    ax.set_xlabel('Human Profiles')
    ax.set_ylabel('Mean Value')
    ax.set_title('Grouped Bar Plot of Algorithms by Human Profiles')
    ax.set_xticks(x + width * (n_algorithms - 1) / 2)
    ax.set_xticklabels(profiles)
    # ax.legend(title="Algorithms")

    # drop the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # main()
    plot_reward_for_seed()
