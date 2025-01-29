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
    elif select_planner == "naive_greedy_planner":
        from adaptive_teaming.planner import NaiveGreedyPlanner

        planner_cfg = cfg[select_planner]
        planner = NaiveGreedyPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif select_planner == "confidence_learner_planner":
        from adaptive_teaming.planner import ConfidenceLearnerPlanner

        planner_cfg = cfg[select_planner]
        planner = ConfidenceLearnerPlanner(
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

    elif select_planner == "fc_adaptive_pref_planner":
        from adaptive_teaming.planner import FacilityLocationPrefPlanner

        if cfg.fc_planner == "mip":
            from adaptive_teaming.planner import FacilityLocationPlanner
            # print("cfg", cfg["fc_mip_planner"])
            cfg["fc_mip_planner"]["teach_adaptive"] = True
            # print("cfg", cfg["fc_mip_planner"])

            fc_planner = FacilityLocationPlanner(
                interaction_env, belief_estimator, cfg["fc_mip_planner"], cfg.cost_cfg
            )
        elif cfg.fc_planner == "greedy":
            from adaptive_teaming.planner import FacilityLocationGreedyPlanner
            cfg["fc_greedy_planner"]["teach_adaptive"] = True
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
            env, cfg.human_model, cfg.cost_cfg, cfg.prob_skill_teaching_success)

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
    # objs_list = ["Box", "Ball", 'Key']
    # colors_list = ["red", "green", "blue", "yellow"]
    objs_list = ["Box", "Ball"]
    colors_list = ["red", "green", "blue"]

    # make the list of all objs by type and color, crossing the two lists
    obj_types = []
    for obj in objs_list:
        for color in colors_list:
            obj_types.append((obj, color))

    # sample the frequency of each object type
    dirichlet_prior = 2 * np.ones(len(obj_types))
    obj_freq = np.random.dirichlet(dirichlet_prior)
    logger.info(f"Sampling tasks with object frequencies: {obj_freq}")
    # TODO sample in more interesting and adversarial ways
    indices = np.random.choice(
        range(len(obj_types)), p=obj_freq, size=n_objs)
    obj_counts = {obj_index: 0 for obj_index in range(len(obj_types))}
    objs_present = []
    for index in indices:
        obj_cls = obj_types[index]
        objs_present.append(obj_cls)

    unique_objs = list(set(objs_present))
    print(f"Unique objects: {unique_objs}")
    num_unteachable_objs = int(len(unique_objs) * cfg.prob_skill_teaching_success)+(1 if cfg.prob_skill_teaching_success > 0 else 0)
    unteachable_objs = random.sample(unique_objs, num_unteachable_objs)
    obj_type_to_teach_prob = {}
    for obj in unique_objs:
        if obj in unteachable_objs:
            obj_type_to_teach_prob[obj] = 0
        else:
            obj_type_to_teach_prob[obj] = 1

    # number of skills the human wants to do
    num_skills_for_human_only = cfg.num_skills_human_wants_to_perform
    # randomly assign skills to the human
    human_only_skills = random.sample(unique_objs, num_skills_for_human_only)

    print(f"Unteachable objects: {unteachable_objs}")
    # pdb.set_trace()
    # randomly assign locations to each object type and color


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

    return [task_seq], obj_type_to_teach_prob, human_only_skills


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
    fig, ax = plt.subplots(figsize=(10, 2))

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
            third_width = width / 3
            ax.barh(
                y=action_order.index(action[0]),
                width=third_width,
                left=start,
                color=action_colors[action[0]],
                edgecolor="black",
                align="center",
                alpha=0.5,
            )
            ax.barh(
                y=action_order.index(action[1]),
                width=third_width,
                left=start + third_width,
                color=action_colors[action[1]],
                edgecolor="black",
                align="center",
                alpha=0.5,
            )
            # add another bar adding a 'Robot' action after the ask skill action
            ax.barh(
                y=action_order.index("ROBOT"),
                width=third_width,
                left=start + 2 * third_width,
                color=action_colors["ROBOT"],
                edgecolor="black",
                align="center",
                alpha=0.5,
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
                start + 0.5,
                action_order.index(action[1]),
                placement[1],
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                weight="bold",
            )
            ax.text(
                start + 0.75,
                action_order.index("ROBOT"),
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
                alpha=0.5,
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

def run_single_planner(cfg, select_planner):
    N_exps = cfg.N_exps
    initial_seed = cfg.seed

    np.random.seed(initial_seed)
    seeds = np.random.randint(0, 10000, N_exps)

    # result_planner_to_list_rewards[select_planner] = []
    # make a folder to house results
    true_human_pref = None
    os.makedirs(f"results/{select_planner}", exist_ok=True)
    reward_list = []
    for i in range(N_exps):
        current_seed = int(seeds[i])
        print(f"Running experiment {i} with seed {current_seed}")
        # make a folder to house seed results
        os.makedirs(f"results/{select_planner}/{current_seed}", exist_ok=True)

        env, interaction_env = init_domain(cfg)
        obs = env.reset()

        start_time = time.time()
        if cfg.env == "gridworld":
            task_seqs, obj_type_to_teach_prob, human_only_skills = randomize_gridworld_task_seq(
                env, cfg, n_objs=cfg.task_seq.num_tasks, seed=current_seed)
        elif cfg.env == "med_gridworld":
            task_seqs, obj_type_to_teach_prob, human_only_skills = randomize_gridworld_task_seq(
                env, cfg, n_objs=cfg.task_seq.num_tasks, seed=current_seed)
        else:
            task_seqs = generate_task_seqs(cfg, n_seqs=1, seed=current_seed)
        logger.info(f"Time to generate task seqs: {time.time() - start_time}")
        print("task_seqs", task_seqs)
        print("obj_type_to_teach_prob", obj_type_to_teach_prob)
        print("human_only_skills", human_only_skills)
        interaction_env.set_teach_probs(obj_type_to_teach_prob)
        if len(human_only_skills) > 0:
            interaction_env.set_human_only_skills(human_only_skills)

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
        total_rew, resultant_objects, resultant_actions, placements, all_rewards = (
            planner.rollout_interaction(
                task_seq, interaction_env.task_similarity, interaction_env.pref_similarity
            )
        )
        # pdb.set_trace()
        reward_list.append(total_rew)
        # result_planner_to_list_rewards[select_planner].append(total_rew)
        print(f"Total reward: {total_rew}")
        print(f"Resultant objects: {resultant_objects}")
        print("all_rewards", all_rewards)

        # resultant_actions = [[a["action_type"] for a in actions] for actions in resultant_actions]
        print(f"Resultant actions: {resultant_actions}")
        print(f"Placements: {placements}")
        print("human only skills", human_only_skills)
        # pdb.set_trace()
        plot_interaction(resultant_objects, resultant_actions,
                         placements, current_seed, select_planner)
        print("total_rew", total_rew)
        # if total_rew < -1400:
        #     pdb.set_trace()

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
    return reward_list

@hydra.main(
    config_path="../cfg", config_name="run_interaction_planner_multi_exp", version_base="1.1"
)
def main(cfg):
    logger.info(f"Output directory: {os.getcwd()}")
    if cfg.compute_ufl_plan_time:
        compute_ufl_plan_time(cfg)
        return


    # come up with N_exps different seeds with the initial seed

    planners_to_run = cfg.planner

    result_planner_to_list_rewards = {}
    all_rewards = []
    for select_planner in planners_to_run:
        result_planner_to_list_rewards[select_planner] = run_single_planner(cfg, select_planner)
        all_rewards.append(result_planner_to_list_rewards[select_planner])


    print("all_rewards", all_rewards)

    print("result_planner_to_list_rewards", result_planner_to_list_rewards)

    return result_planner_to_list_rewards



def plot_adaptive():
    # High Cost Human: 6 different objects, 25 total objects, 10, 50, 90% teachable
    ten_high_human ={'fc_pref_planner': [-1285, -1345, -2825, -2550, -1465, -2000, -2065, -1225, -2395, -1285, -1675, -1645, -1285, -1725, -1270, -2025, -1120, -1300, -1225, -1195, -1800, -1510, -2025, -1390, -1525, -1770, -1605, -1995, -2165, -2225], 'fc_adaptive_pref_planner': [-1285, -1345, -1825, -2025, -1465, -2000, -1465, -1225, -1795, -1285, -1675, -1645, -1285, -1525, -1270, -2025, -1120, -1300, -1225, -1195, -1800, -1510, -2025, -1390, -1525, -1570, -1405, -1795, -1765, -2025]}
    fifty_high_human = {'fc_pref_planner': [-2225, -3385, -3325, -2225, -2205, -2000, -2225, -1985, -2225, -2285, -2000, -3185, -2025, -2025, -2790, -2850, -1120, -4160, -3005, -1975, -1800, -2025, -2025, -2025, -2025, -2225, -2025, -2995, -2000, -2365], 'fc_adaptive_pref_planner': [-2225, -2185, -2125, -2025, -2005, -2000, -2225, -1585, -2225, -1885, -2000, -2000, -2025, -2025, -1990, -2050, -1120, -1975, -2005, -1975, -1800, -2025, -2025, -2025, -1825, -2025, -2025, -2395, -2000, -1765]}
    ninety_high_human =  {'fc_pref_planner': [-2225, -2025, -2225, -2225, -2000, -2000, -2225, -2225, -2225, -2000, -2000, -2225, -2025, -2025, -2050, -2025, -2200, -2400, -2225, -1975, -1800, -2025, -2025, -2025, -2200, -2225, -2025, -2025, -2000, -2225], 'fc_adaptive_pref_planner': [-2225, -2025, -2225, -2025, -2000, -2000, -2225, -2025, -2225, -2000, -2000, -2000, -2025, -2025, -2050, -2025, -2000, -1975, -2225, -1975, -1800, -2025, -2025, -2025, -2000, -2025, -2025, -2025, -2000, -2025]}


    # med cost human: 6 different objects, 25 total objects, 10, 50, 90% teachable
    # ten_high_human = {'fc_pref_planner': [-950, -1290, -1550, -1330, -1310, -1310, -1360, -1650, -1340, -1710, -1400, -1020, -1710, -1170, -1490, -950, -1530, -820, -1970, -950, -880, -1330, -1750, -1720, -950, -890, -1420, -1460, -1460, -1580], 'fc_adaptive_pref_planner': [-950, -1190, -1250, -1130, -1210, -1210, -1060, -1250, -1140, -1310, -1200, -1020, -1310, -1070, -1190, -950, -1230, -820, -1370, -950, -880, -1130, -1250, -1320, -950, -890, -1120, -1260, -1260, -1180]}
    # fifty_high_human ={'fc_pref_planner': [-3230, -3130, -2430, -2570, -2310, -3110, -2920, -2730, -2540, -3230, -3360, -2270, -2040, -2240, -2410, -2150, -1530, -3300, -2890, -2210, -2040, -2070, -2030, -2200, -2240, -2730, -2060, -2200, -2170, -1580], 'fc_adaptive_pref_planner': [-2030, -2030, -1730, -1670, -1810, -2110, -1720, -1730, -1740, -2070, -2160, -2040, -1940, -2140, -1610, -1550, -1230, -1900, -1790, -2010, -2040, -1970, -1430, -2200, -1940, -1970, -1940, -2100, -2170, -1180]}
    # ninety_high_human = {'fc_pref_planner': [-2140, -2140, -2140, -2270, -2170, -2100, -2240, -2240, -2200, -2270, -2300, -2270, -2040, -2240, -2170, -2170, -2110, -2210, -2140, -2210, -2040, -2070, -2170, -2200, -2240, -2270, -2040, -2200, -2170, -2240], 'fc_adaptive_pref_planner': [-2040, -2140, -2040, -2170, -2070, -2100, -2140, -2040, -2100, -2070, -2200, -2040, -1940, -2140, -2070, -2070, -1910, -2010, -2040, -2010, -2040, -1970, -2070, -2200, -1940, -1970, -1940, -2100, -2170, -1940]}

    # low cost human: 6 different objects, 25 total objects, 10, 50, 90% teachable
    # ten_high_human ={'fc_pref_planner': [-640, -730, -1360, -1215, -820, -1000, -890, -640, -855, -640, -640, -820, -1180, -820, -945, -1090, -730, -585, -1270, -820, -640, -945, -1035, -820, -640, -945, -585, -1000, -820, -1485], 'fc_adaptive_pref_planner': [-640, -700, -1120, -1005, -760, -880, -770, -640, -765, -640, -640, -760, -1000, -760, -825, -940, -700, -585, -1060, -760, -640, -825, -885, -760, -640, -825, -585, -880, -760, -1185]}
    #
    # fifty_high_human = {'fc_pref_planner': [-1630, -2260, -2080, -1515, -1900, -1900, -1640, -1810, -1515, -2080, -1990, -1540, -1990, -1720, -1875, -1990, -2440, -975, -1990, -1270, -1990, -1875, -2055, -1900, -2170, -1875, -1425, -2080, -2260, -2145], 'fc_adaptive_pref_planner': [-1360, -1780, -1660, -1245, -1540, -1540, -1310, -1480, -1245, -1660, -1600, -1300, -1600, -1420, -1485, -1600, -1900, -885, -1600, -1120, -1600, -1485, -1605, -1540, -1720, -1485, -1185, -1660, -1780, -1665]}
    # ninety_high_human ={'fc_pref_planner': [-2650, -2650, -2650, -2625, -2650, -2650, -2600, -2650, -2625, -2650, -2650, -2650, -2650, -2650, -2625, -2650, -2650, -2625, -2650, -2650, -2650, -2625, -2625, -2650, -2650, -2625, -2625, -2650, -2650, -2625], 'fc_adaptive_pref_planner': [-2080, -2080, -2080, -2025, -2080, -2080, -1970, -2080, -2025, -2080, -2080, -2080, -2080, -2080, -2025, -2080, -2080, -2025, -2080, -2080, -2080, -2025, -2025, -2080, -2080, -2025, -2025, -2080, -2080, -2025]}


    # 6 different objects, 25 total objects, 50% teachable

    # invert all the costs
    ten_high_human = {k: [-x for x in v] for k, v in ten_high_human.items()}
    fifty_high_human = {k: [-x for x in v] for k, v in fifty_high_human.items()}
    ninety_high_human = {k: [-x for x in v] for k, v in ninety_high_human.items()}

    profiles = ['10%', '50%', '90%']
    # profiles = ['low_cost', 'med_cost', ]
    data = {
        '10%': ten_high_human,
        '50%': fifty_high_human,
        '90%': ninety_high_human
    }

    algorithm_to_name = {
        'fc_adaptive_pref_planner': 'COIL Adaptive',
        'fc_pref_planner': 'COIL Nonadaptive',
        'task_info_gain_planner': 'Task Info Gain',
        'info_gain_planner': 'Info Gain',
        'confidence_based_planner': 'C-ADL',
        'naive_greedy_planner': 'Myopic Greedy',
        'confidence_learner_planner': 'C-Learn'
    }

    algorithms = list(ten_high_human.keys())
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
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots(figsize=(3, 3))

    for i, algo in enumerate(algorithms):
        color_map = {
            'fc_adaptive_pref_planner':'#6495ED',
            'fc_pref_planner': '#ADD8E6',
            'task_info_gain_planner': '#F26B0F',
            'confidence_based_planner': '#91AC8F',
            'naive_greedy_planner': '#808080',
            'confidence_learner_planner': '#EAE2C6'
        }
        ax.bar(
            x + i * width,
            means[:, i],
            width,
            label=algorithm_to_name[algo],
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

def plot_reward_for_seed():
    # 6 different objects, 25 total objects, concentration parameter = 2
    low_cost_human ={'fc_adaptive_pref_planner': [-920, -970, -920, -850, -970, -970, -730, -870, -850, -920, -920, -970, -920, -870, -850, -920, -920, -800, -870, -920, -870, -850, -800, -920, -920, -850, -800, -970, -970, -800], 'task_info_gain_planner': [-970, -970, -970, -850, -970, -970, -730, -970, -850, -970, -970, -970, -970, -970, -850, -970, -970, -850, -970, -970, -970, -850, -850, -970, -970, -850, -850, -970, -970, -850], 'confidence_based_planner': [-970, -970, -970, -850, -970, -970, -730, -970, -850, -970, -970, -970, -970, -970, -850, -970, -970, -850, -970, -970, -970, -850, -850, -970, -970, -850, -850, -970, -970, -850], 'naive_greedy_planner': [-2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000, -2000], 'confidence_learner_planner': [-970, -970, -970, -850, -970, -970, -730, -970, -850, -970, -970, -970, -970, -970, -850, -970, -970, -850, -970, -970, -970, -850, -850, -970, -970, -850, -850, -970, -970, -850]}

    med_cost_human =  {'fc_pref_planner': [-1070, -1130, -1070, -1010, -1210, -1210, -850, -1010, -1050, -1070, -1110, -1170, -1070, -1010, -1010, -1070, -990, -910, -1010, -1070, -970, -1010, -950, -1110, -1070, -1010, -910, -1170, -1170, -910], 'confidence_based_planner': [-1170, -1210, -1170, -1050, -1210, -1210, -890, -1130, -1050, -1170, -1170, -1210, -1170, -1130, -1050, -1170, -1170, -1010, -1130, -1170, -1130, -1050, -1010, -1170, -1170, -1050, -1010, -1210, -1210, -1010]}
    high_cost_human ={'fc_adaptive_pref_planner': [-1285, -1345, -1285, -1270, -1465, -1555, -1045, -1225, -1375, -1285, -1435, -1405, -1285, -1225, -1270, -1330, -1120, -1060, -1225, -1195, -1800, -1270, -1210, -1390, -1285, -1270, -1105, -1495, -1405, -1105],
                      'confidence_based_planner': [-1360, -1420, -1360, -1295, -1540, -1580, -1070, -1300, -1375, -1360, -1460, -1480, -1360, -1300, -1295, -1380, -1220, -1135, -1300, -1320, -1240, -1295, -1235, -1440, -1360, -1295, -1155, -1520, -1480, -1155],
                    'task_info_gain_planner': [-1460, -1600, -1460, -1375, -1600, -1600, -1150, -1320, -1375, -1460, -1460, -1600, -1460, -1320, -1375, -1460, -1460, -1235, -1320, -1460, -1320, -1375, -1235, -1460, -1460, -1375, -1235, -1600, -1600, -1235],
                        'naive_greedy_planner': [-1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750],
                      'confidence_learner_planner': [-1600, -1600, -1600, -1375, -1600, -1600, -1150, -1600, -1375, -1600, -1600, -1600, -1600, -1600, -1375, -1600, -1600, -1375, -1600, -1600, -1600, -1375, -1375, -1600, -1600, -1375, -1375, -1600, -1600, -1375]}

    # 6 different objects, contribution prefs
    # low_cost_human ={'fc_adaptive_pref_planner': [-1060, -760, -1360, -1005, -1060, -940, -830, -880, -1245, -1420, -1000, -1120, -1060, -1420, -1005, -1000, -1600, -645, -1060, -1420, -1000, -1065, -1125, -940, -820, -1305, -1005, -1060, -1060, -885],
    #                  'confidence_based_planner': [-1060, -760, -1360, -1005, -1060, -940, -830, -880, -1245, -1420, -1000, -1120, -1060, -1420, -1005, -1000, -1600, -645, -1060, -1420, -1000, -1065, -1125, -940, -820, -1305, -1005, -1060, -1060, -885],
    #               'task_info_gain_planner': [-1060, -760, -1360, -1005, -1060, -940, -830, -880, -1245, -1420, -1000, -1120, -1060, -1420, -1005, -1000, -1600, -645, -1060, -1420, -1000, -1065, -1125, -940, -820, -1305, -1005, -1060, -1060, -885],
    #                 'naive_greedy_planner': [-1060, -760, -1360, -1005, -1060, -940, -830, -880, -1245, -1420, -1000, -1120, -1060, -1420, -1005, -1000, -1600, -645, -1060, -1420, -1000, -1065, -1125, -940, -820, -1305, -1005, -1060, -1060, -885],
    #                  'confidence_learner_planner': [-2380, -1380, -3380, -2325, -2380, -1980, -1870, -1780, -3125, -3580, -2180, -2580, -2380, -3580, -2325, -2180, -4180, -1125, -2380, -3580, -2180, -2525, -2725, -1980, -1580, -3325, -2325, -2380, -2380, -1925]}
    #
    # med_cost_human ={'fc_adaptive_pref_planner': [-1270, -1010, -1230, -1530, -1190, -1310, -1080, -1350, -1300, -1390, -1340, -1600, -950, -1170, -1410, -1210, -1130, -1040, -1170, -1030, -1200, -1270, -1470, -1240, -1840, -1230, -1200, -1160, -1160, -1020],
    #                  'confidence_based_planner': [-1310, -1070, -1270, -1540, -1190, -1310, -1110, -1410, -1300, -1450, -1370, -1610, -1010, -1230, -1420, -1270, -1210, -1080, -1230, -1070, -1270, -1300, -1500, -1270, -1690, -1240, -1240, -1190, -1190, -1080],
    #                   'task_info_gain_planner': [-1310, -1070, -1310, -1540, -1190, -1310, -1110, -1490, -1300, -1490, -1370, -1610, -1010, -1310, -1420, -1310, -1250, -1120, -1310, -1070, -1310, -1300, -1540, -1310, -1730, -1240, -1240, -1190, -1190, -1120],
    #                     'naive_greedy_planner': [-1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750],
    #                  'confidence_learner_planner': [-2630, -1830, -2630, -3700, -2230, -2630, -2570, -3230, -2900, -3230, -2830, -3630, -1630, -2630, -3300, -2630, -2430, -2300, -2630, -1830, -2630, -2900, -3700, -2630, -4030, -2700, -2700, -2230, -2230, -2300]}
    #
    # high_cost_human = {'fc_adaptive_pref_planner': [-1605, -1685, -1385, -1470, -1800, -1595, -1265, -1825, -1575, -1445, -1800, -1545, -1285, -1325, -1370, -1490, -1800, -1100, -1225, -1775, -1800, -1430, -1825, -1430, -1800, -1410, -1385, -1535, -1565, -1105],
    #                    'confidence_based_planner': [-1680, -1760, -1460, -1495, -1800, -1620, -1290, -1640, -1575, -1520, -1780, -1620, -1360, -1400, -1395, -1540, -1900, -1175, -1300, -1900, -1860, -1455, -1455, -1480, -1800, -1435, -1435, -1560, -1640, -1155],
    #                   'task_info_gain_planner': [-1780, -1860, -1480, -1575, -1860, -1620, -1290, -1640, -1575, -1540, -1780, -1740, -1360, -1420, -1395, -1540, -2140, -1275, -1300, -2040, -1940, -1455, -1455, -1500, -1900, -1515, -1435, -1560, -1740, -1155],
    #                 'naive_greedy_planner': [-1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750],
    #                    'confidence_learner_planner': [-4000, -3800, -3000, -3375, -3800, -3000, -2950, -4000, -3375, -3200, -4000, -3400, -2600, -2800, -2775, -3200, -5200, -2375, -2400, -4400, -5000, -2975, -2975, -2600, -4400, -3175, -3375, -2800, -3400, -1975]}

    # high_cost_human = {
    #     'confidence_based_planner': [-1360, -1760, -1800, -1395, -1680, -1720, -1530, -1640, -1755, -1520, -1540, -1760, -1740, -1400, -1555, -1540, -1220, -1175, -1580, -1320, -1460, -1335, -1455, -1580, -1360, -1395, -1495, -1660, -1740, -1415]
    # }

    # swap the ordering of confidence_based_planner and task_info_gain_planner
    # drop confidence_learner_planner
    # low_cost_human.pop('confidence_learner_planner')
    # med_cost_human.pop('confidence_learner_planner')
    # high_cost_human.pop('confidence_learner_planner')

    # invert all the costs
    low_cost_human = {k: [-x for x in v] for k, v in low_cost_human.items()}
    med_cost_human = {k: [-x for x in v] for k, v in med_cost_human.items()}
    high_cost_human = {k: [-x for x in v] for k, v in high_cost_human.items()}

    profiles = ['low_cost', 'med_cost',  'high_cost', ]
    profiles = ['low_cost',  ]
    # drop naive greedy planner
    # med_cost_human = {k: v for k, v in med_cost_human.items() if k != 'naive_greedy_planner'}

    data = {
        'low_cost': low_cost_human,
        # 'med_cost': med_cost_human,
        # 'high_cost': high_cost_human
    }

    algorithm_to_name = {
        'fc_adaptive_pref_planner': 'Adaptive COIL',
        'fc_pref_planner': 'COIL',
        'task_info_gain_planner': 'Task Info Gain',
        'info_gain_planner': 'Info Gain',
        'confidence_based_planner': 'CADL',
        'naive_greedy_planner': 'Greedy',
        'confidence_learner_planner': 'CBA'
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
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(3, 3))

    for i, algo in enumerate(algorithms):
        color_map = {
            'fc_adaptive_pref_planner':'#6495ED',
            'fc_pref_planner': '#6495ED',
            'task_info_gain_planner': '#F26B0F',
            'confidence_based_planner': '#91AC8F',
            'naive_greedy_planner': '#808080',
            'confidence_learner_planner': '#EAE2C6'
        }
        ax.bar(
            x + i * width,
            means[:, i],
            width,
            label=algorithm_to_name[algo],
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

def plot_reward_for_seed_with_break():
    # 6 different objects, 25 total objects, concentration parameter = 2
    med_cost_human = {'fc_adaptive_pref_planner': [-950, -1010, -950, -890, -1030, -1030, -760, -890, -900, -950, -960, -1020, -950, -890, -890, -950, -930, -820, -890, -950, -880, -890, -830, -960, -950, -890, -820, -1020, -1020, -820], 'task_info_gain_planner': [-1030, -1030, -1030, -900, -1030, -1030, -770, -1030, -900, -1030, -1030, -1030, -1030, -1030, -900, -1030, -1030, -900, -1030, -1030, -1030, -900, -900, -1030, -1030, -900, -900, -1030, -1030, -900], 'confidence_based_planner': [-990, -1030, -990, -900, -1030, -1030, -770, -950, -900, -990, -990, -1030, -990, -950, -900, -990, -990, -860, -950, -990, -950, -900, -860, -990, -990, -900, -860, -1030, -1030, -860], 'naive_greedy_planner': [-1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750, -1750], 'confidence_learner_planner': [-1030, -1030, -1030, -900, -1030, -1030, -770, -1030, -900, -1030, -1030, -1030, -1030, -1030, -900, -1030, -1030, -900, -1030, -1030, -1030, -900, -900, -1030, -1030, -900, -900, -1030, -1030, -900]}


    # invert all the costs
    # low_cost_human = {k: [-x for x in v] for k, v in low_cost_human.items()}
    med_cost_human = {k: [-x for x in v] for k, v in med_cost_human.items()}
    # high_cost_human = {k: [-x for x in v] for k, v in high_cost_human.items()}

    profiles = ['low_cost', 'med_cost',  'high_cost', ]
    profiles = ['med_cost',  ]
    data = {
        # 'low_cost': low_cost_human,
        'med_cost': med_cost_human,
        # 'high_cost': high_cost_human
    }

    algorithm_to_name = {
        'fc_adaptive_pref_planner': 'Adaptive COIL',
        'fc_pref_planner': 'COIL',
        'task_info_gain_planner': 'Task Info Gain',
        'info_gain_planner': 'Info Gain',
        'confidence_based_planner': 'CADL',
        'naive_greedy_planner': 'Greedy',
        'confidence_learner_planner': 'CBA'
    }

    algorithms = list(med_cost_human.keys())
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
    width = 0.15  # the width of the bars

    f, (ax, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': (0.3,2)}, figsize=(3, 3))




    for i, algo in enumerate(algorithms):
        color_map = {
            'fc_adaptive_pref_planner':'#E73879',
            'fc_pref_planner': '#6495ED',
            'task_info_gain_planner': '#F26B0F',
            'confidence_based_planner': '#91AC8F',
            'naive_greedy_planner': '#808080',
            'confidence_learner_planner': '#EAE2C6'
        }
        ax.bar(
            x + i * width,
            means[:, i],
            width,
            label=algorithm_to_name[algo],
            yerr=errors[:, i],
            capsize=5,
            color=color_map.get(algo, 'gray') ,# Default to gray if algorithm not in color map
        )

    for i, algo in enumerate(algorithms):
        color_map = {
            'fc_adaptive_pref_planner':'#E73879',
            'fc_pref_planner': '#6495ED',
            'task_info_gain_planner': '#F26B0F',
            'confidence_based_planner': '#91AC8F',
            'naive_greedy_planner': '#808080',
            'confidence_learner_planner': '#EAE2C6'
        }
        ax2.bar(
            x + i * width,
            means[:, i],
            width,
            label=algorithm_to_name[algo],
            yerr=errors[:, i],
            capsize=5,
            color=color_map.get(algo, 'gray') , # Default to gray if algorithm not in color map
        )

    # zoom-in / limit the view to different portions of the data
    ax.set_ylim(1400, 1600)  # outliers only
    ax2.set_ylim(0, 1000)  # most of the data

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # set yaxis to be by 200
    # ax.set_yticks(np.arange(900, 1200, 200))
    # ax2.set_yticks(np.arange(0, 800, 200))

    d = .01  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    # arguments to pass to plot, just so we don't keep repeating them
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # Add labels, title, and legend
    # ax.set_xlabel('Human Profiles')
    ax.set_ylabel('Mean Value')
    ax.set_title('Grouped Bar Plot of Algorithms by Human Profiles')
    ax.set_xticks(x + width * (n_algorithms - 1) / 2)
    # ax.set_xticklabels(profiles)
    # ax.legend(title="Algorithms")

    # drop the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_info_gain_baselines():
    # 6 different objects, 25 total objects, concentration parameter = 2


    info_gain_ablation = {
        '0.001': [-580, -580, -580, -525, -580, -580, -470, -580, -525, -580, -1030, -1030, -1030, -900, -1030, -1030, -770, -1030, -900, -1030, -1600, -1600, -1600, -1375, -1600, -1600, -1150, -1600, -1375, -1600],
        '0.01': [-580, -580, -580, -525, -580, -580, -470, -580, -525, -580, -1030, -1030, -1030, -900, -1030, -1030, -770, -1030, -900, -1030, -1460, -1600, -1460, -1375, -1600, -1600, -1150, -1320, -1375, -1460],
        '0.05':[-580, -580, -580, -525, -580, -580, -470, -580, -525, -580, -990, -1030, -990, -900, -1030, -1030, -770, -950, -900, -990, -1400, -1560, -1560, -1595, -1680, -1900, -1170, -1560, -1655, -1560],
        '0.1': [-580, -580, -580, -525, -580, -580, -470, -580, -525, -580, -1090, -1150, -1090, -920, -1270, -1110, -790, -1030, -900, -1090, -1900, -1900, -1900, -1875, -1900, -1900, -1850, -1900, -1875, -1900],
        '0.5': [-580, -580, -580, -525, -580, -580, -470, -580, -525, -580, -1930, -1930, -1930, -1900, -1930, -1930, -1870, -1930, -1900, -1930, -1900, -1900, -1900, -1875, -1900, -1900, -1850, -1900, -1875, -1900],
        '1': [-580, -580, -580, -525, -580, -580, -470, -580, -525, -580, -1930, -1930, -1930, -1900, -1930, -1930, -1870, -1930, -1900, -1930, -1900, -1900, -1900, -1875, -1900, -1900, -1850, -1900, -1875, -1900]
    }

    # invert all the costs
    info_gain_ablation = {k: [-x for x in v] for k, v in info_gain_ablation.items()}

    profiles = ['info_gain',  ]
    data = {
        'info_gain': info_gain_ablation
    }

    algorithm_to_name = {
        '0.001': '0.001',
        '0.1': '0.1',
        '0.5': '0.5',
        '1': '1',
        '0.01': '0.01',
        '0.05': '0.05',

    }

    algorithms = list(info_gain_ablation.keys())
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
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(5, 5))

    for i, algo in enumerate(algorithms):
        # color map should be five different shades of orange
        color_map = {
            '0.001': '#FFBF00',
            '0.01': '#E49B0F',
            '0.05': '#CC5500',
            '0.1': '#F26B0F',
            '0.5': '#E3963E',
            '1': '#E97451'


        }
        ax.bar(
            x + i * width,
            means[:, i],
            width,
            label=algorithm_to_name[algo],
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
    ax.legend(title="Beta")

    # drop the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # main()
    # plot_reward_for_seed_with_break()
    plot_reward_for_seed()
    # plot_adaptive()
    # plot_info_gain_baselines()
