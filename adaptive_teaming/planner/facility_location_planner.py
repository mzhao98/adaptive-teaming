import logging
from itertools import product
from pprint import pprint

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

# planner_type = "informed_greedy"
# planner_type = "greedy"
planner_type = "facility_location"


class FacilityLocationPlanner:

    def __init__(self, cost_cfg):
        self.cost_cfg = cost_cfg

    def update_pref_belief(self, task, pref_obs):
        """Update each preference belief based on the observed preference."""
        pass

    def plan(self, task_seq, task_similarity_fn, pref_similarity_fn):
        """
        task_seq: sequence of tasks each described a vector
        task_similarity_fn: function to compute similarity between tasks
        pref_similarity_fn: function to compute similarity between task preferences
        """
        N = len(task_seq)
        # some large constant
        M = sum(val for val in self.cost_cfg.values())

        # # three types of facilities
        # # delegate to human
        # C_hum = np.zeros((N, 1)) + self.cost_cfg['hum']
        # # demo query
        # C_demo = np.zeros((N, N)) + self.cost_cfg['demo']
        # # pref query
        # # n train_task_id, pref_id, test_task_id
        # C_pref = np.zeros((N, N, N)) + self.cost_cfg['pref']

        # # perfect transfer on self
        # Pi_transfer = np.eye(N)
        Pi_transfer = np.zeros((N, N))
        for i, a in enumerate(task_seq):
            for j, b in enumerate(task_seq):
                Pi_transfer[i, j] = task_similarity_fn(a, b)

        print("Pi_transfer")
        pprint(Pi_transfer.round(2))

        # # assume perfect transfer on self
        Pref_transfer = np.eye(N)

        # # both demo will have P(unsafe) * c_fail penalty + P(wrong pref) * c_pref_penalty
        # for i in range(N):
        # # can't help on previous tasks
        # C_demo[i, :i] += M
        # for j in range(i, N):
        # C_demo[i, j] += ((1 - Pi_transfer[i, j]) * self.cost_cfg['fail_cost'] +
        # (1 - Pref_transfer[i, j]) * self.cost_cfg['pref_cost'])
        # if i != j:
        # C_demo[i, j] += self.cost_cfg['rob']

        demands = task_seq

        facilities = []
        setup_costs = {}
        service_costs = {}
        # create facilities
        # gurobi wants all tuples to be of the same length
        # setup_costs:(type-of-query, train_task_id, demo_task_id)
        # service_costs:(type-of-query, train_task_id, demo_task_id, pref_task_id)
        for task_id, task in enumerate(demands):
            # hum
            facility = ("hum", f"task-{task_id}")
            facilities.append(facility)
            setup_costs[facility] = self.cost_cfg["hum"]

            # always succeeds
            # Note: I could either set a high cost for all the other tasks or I
            # "think" I can just ignore the other combinations by not creating
            # the corresponding decision variables.
            service_costs[(facility, task_id)] = 0

            # demo
            facility = ("demo", f"task-{task_id}")
            facilities.append(facility)
            setup_costs[facility] = self.cost_cfg["demo"]

            # demo affects only current and future tasks.
            # assume demo also complete the task perfectly
            service_costs[(facility, task_id)] = 0
            for test_task_id in range(task_id + 1, N):
                service_costs[(facility, test_task_id)] = (
                    self.cost_cfg["rob"]
                    + (1 - Pi_transfer[task_id, test_task_id])
                    * self.cost_cfg["fail_cost"]
                    + (1 - Pref_transfer[task_id, test_task_id])
                    * self.cost_cfg["pref_cost"]
                )

            # pref
            # each task i creates i-1 pref facilities
            # pref on task_id with all previous demos
            for demo_task_id in range(task_id):
                facility = ("pref", f"task-{task_id}, demo-{demo_task_id}")
                # facility = ('pref', str(task_id) + '-' + str(demo_task_id))
                facilities.append(facility)
                setup_costs[facility] = self.cost_cfg["pref"]

                # now compute service costs
                # XXX shouldn't the preferene also generalize to other tasks??
                # XXX this will require another for loop
                service_costs[(facility, task_id)] = (
                    self.cost_cfg["rob"]
                    + (1 - Pi_transfer[demo_task_id, task_id])
                    * self.cost_cfg["fail_cost"]
                    + (1 - Pref_transfer[task_id, task_id]
                       ) * self.cost_cfg["pref_cost"]
                )

        # N tasks = demands
        self._solve_facility_location(
            demands, facilities, setup_costs, service_costs)

        # __import__('ipdb').set_trace()

        # # pref options
        # # train task id
        # for i in range(N):
        # # skill id
        # for j in range(N):
        # # test task id
        # for k in range(N):
        # # C_pref[i, j] += (self.cost_cfg['fail'] * +
        # # self.cost_cfg['pref_penalty'] * task_seq[i]['P(wrong pref)'])

        # compute similarity matrix between task preferences

        # compute cost matrix

    # def _solve_facility_location(self, demands, facilities, setup_costs, service_costs):
    def _solve_facility_location(self, demands, facilities, setup_costs, service_costs):
        """
        This function uses the Gurobi dictionary API which associates a tuple with each Gurobi variable. The
        variables can then be accessed using the tuple as a key. This is useful
        for managing a large number of variables.

        Main classes: tuplelist and tuple dict
        """
        # cartesian_prod = list(product(range(num_demands),
        # range(num_facilities)))
        # service_costs = {(d, f): service_costs[d, f] for d, f in cartesian_prod}

        model = gp.Model("facility_location")

        select = model.addVars(facilities, vtype=GRB.BINARY, name="Select")
        assign = model.addVars(service_costs, ub=1,
                               vtype=GRB.CONTINUOUS, name="Assign")

        for key in service_costs:
            f, c = key[0], key[1]
            model.addConstr(assign[key] <= select[f], name="Setup2ship")

        for i, _ in enumerate(demands):
            for f in facilities:
                model.addConstr(
                    gp.quicksum(assign[(f, i)]
                                for f in facilities if (f, i) in assign)
                    == 1,
                    name=f"Demand-{i}",
                )

        model.update()

        model.setObjective(
            select.prod(setup_costs) + assign.prod(service_costs), GRB.MINIMIZE
        )

        model.optimize()

        __import__("ipdb").set_trace()


def vis_world(world_state, fig=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    tasks = world_state["tasks"]
    prefs = world_state["prefs"]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True)
    for i, task in enumerate(tasks):
        shape = task["shape"]
        color = task["color"]
        size = task["size"]
        pos = task["pos"]
        ax.scatter(pos[0], pos[1], c=color, s=size, marker=shape)
        ax.text(pos[0], pos[1], i, fontsize=12, color="black")

    for pref_id, pref in prefs.items():
        shape = pref["shape"]
        color = pref["color"]
        size = pref["size"]
        pos = pref["pos"]
        ax.scatter(pos[0], pos[1], c=color, s=size, marker=shape)
        ax.text(pos[0], pos[1], pref_id, fontsize=12, color="black")
    # plt.show()
    return fig, ax


def vis_pref_beliefs(pref_beliefs, fig=None):
    # ax = fig.add_subplot(1, 2, 2)
    fig, axs = plt.subplots(1, len(pref_beliefs),
                            figsize=(3 * len(pref_beliefs), 3))
    # ax.set_xlim(-2, 2)
    for ax in axs:
        ax.set_ylim(0, 1)
        ax.grid(True)
    for i, beliefs in enumerate(pref_beliefs):
        # axs[i].bar(np.arange(len(beliefs)), beliefs, color='blue')
        axs[i].bar(["G1", "G2"], beliefs, color="blue")
        axs[i].set_title(f"Task-{i}")
    return fig, axs


def precond_prediction(train_task, test_task):
    if train_task["shape"] != test_task["shape"]:
        return 0

    if np.linalg.norm(np.array(train_task["pos"]) - np.array(test_task["pos"])) < 0.6:
        return 1
    else:
        return 0


def skill_library_precond(skills, task):
    return max([precond_prediction(skill["train_task"], task) for skill in skills])


if __name__ == "__main__":
    # test code
    cost_cfg = {
        "rob": 10,
        "hum": 90,
        "demo": 150,
        "pref": 20,
        "fail_cost": 100,
        "pref_cost": 50,
        # "pref_cost": 0
    }

    # tasks
    task_seq = [
        {"shape": "s", "color": "red", "size": 300, "pos": [0, 0]},
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
        def task_similarity_fn(x, y): return np.exp(-np.linalg.norm(x - y))
        pref_similarity_fn = task_similarity_fn
        planner.plan(task_seq, task_similarity_fn, pref_similarity_fn)
