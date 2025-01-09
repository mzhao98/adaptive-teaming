import logging
from copy import copy, deepcopy
from itertools import product
from pprint import pprint, pformat

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

from .base_planner import InteractionPlanner
import pdb

logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)

# planner_type = "informed_greedy"
# planner_type = "greedy"
planner_type = "facility_location"


class FacilityLocationPlanner(InteractionPlanner):

    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)

    def plan(
        self,
        task_seq,
        pref_beliefs,
        task_similarity_fn,
        pref_similarity_fn,
        current_task_id,
    ):
        """
        task_seq: sequence of tasks each described a vector
        task_similarity_fn: function to compute similarity between tasks
        pref_similarity_fn: function to compute similarity between task preferences
        """
        N = len(task_seq[current_task_id:])
        # some large constant
        M = sum(val for val in self.cost_cfg.values())
        pref_space = self.interaction_env.pref_space

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
        # task_similarity = np.zeros((N, N))
        # for i, a in enumerate(task_seq):
        # for j, b in enumerate(task_seq):
        # task_similarity[i, j] = task_similarity_fn(a, b)

        # print("Pi_transfer")
        # pprint(Pi_transfer.round(2))

        # # assume perfect transfer on self
        # Pref_transfer = np.eye(N)

        def Pi_transfer(task1, task2, pref1, pref2):
            """
            Estimates the probability of transferring the skill from task1 to
            task2 given the preference param with which the skill on task1 was
            trained on.
            """
            # task_similarity_fn(task, test_task)
            # pref_similarity_fn(mle_train_pref, 'G1')
            # TODO for already learned skills, simulate the prob
            task_sim = task_similarity_fn(task1, task2)
            pref_sim = pref_similarity_fn(pref1, pref2)
            return task_sim * pref_sim

        # # both demo will have P(unsafe) * c_fail penalty + P(wrong pref) * c_pref_penalty
        # for i in range(N):
        # # can't help on previous tasks
        # C_demo[i, :i] += M
        # for j in range(i, N):
        # C_demo[i, j] += ((1 - Pi_transfer[i, j]) * self.cost_cfg['fail_cost'] +
        # (1 - Pref_transfer[i, j]) * self.cost_cfg['pref_cost'])
        # if i != j:
        # C_demo[i, j] += self.cost_cfg['rob']

        demands = task_seq[current_task_id:]

        facilities = []
        setup_costs = {}
        service_costs = {}
        best_prefs = {}  # for ROBOT and ASK_SKILL actions
        best_skills = {}  # for ROBOT actions
        # create facilities
        # gurobi wants all tuples to be of the same length
        # setup_costs:(type-of-query, train_task_id, demo_task_id)
        # service_costs:(type-of-query, train_task_id, demo_task_id, pref_task_id)
        for task_id, task in enumerate(task_seq[current_task_id:], start=current_task_id):
            train_pref_belief = pref_beliefs[task_id]
            mle_train_pref = pref_space[np.argmax(train_pref_belief)]
            # hum
            facility = ("HUMAN", f"task-{task_id}")
            facilities.append(facility)
            setup_costs[facility] = self.cost_cfg["HUMAN"]

            # always succeeds
            # Note: I could either set a high cost for all the other tasks or I
            # "think" I can just ignore the other combinations by not creating
            # the corresponding decision variables.
            service_costs[(facility, task_id)] = 0

            # demo
            facility = ("ASK_SKILL", f"task-{task_id}")
            facilities.append(facility)
            setup_costs[facility] = self.cost_cfg["ASK_SKILL"]

            # demo affects only current and future tasks.
            # assume demo also complete the task perfectly
            service_costs[(facility, task_id)] = 0
            # for test_task_id in range(task_id + 1, N):

            for test_task_id in range(task_id, len(task_seq)):
                test_task = task_seq[test_task_id]
                # XXX robot specifically asks for a skill with a pref params
                # XXX this will be the mle pref of the user ofc
                service_costs[(facility, test_task_id)
                              ] = self.cost_cfg["ROBOT"]
                pref_belief = pref_beliefs[test_task_id]
                # mle_test_pref = pref_space[np.argmax(pref_belief)]
                # argmin over pref params
                pi_transfers = [
                    Pi_transfer(task, test_task, mle_train_pref, pref)
                    for pref in pref_space
                ]
                execution_costs = (1 - np.array(pi_transfers)
                                   ) * self.cost_cfg["FAIL"]
                # pref_costs = (
                    # np.array(
                        # [
                            # 1 - pref_similarity_fn(mle_train_pref, pref)
                            # for pref in pref_space
                        # ]
                    # )
                    # * self.cost_cfg["PREF_COST"]
                # )
                pref_costs = (1 - np.array(pref_belief)) * self.cost_cfg["PREF_COST"]
                logger.debug(f" Belief pref costs: {pref_costs}")

                # save the best pref for use during execution
                # pdb.set_trace()
                best_prefs[(facility, test_task_id)] = pref_space[
                    np.argmin(execution_costs + pref_costs)
                ]
                service_costs[(facility, test_task_id)] += min(
                    execution_costs + pref_costs
                )
                # __import__('ipdb').set_trace()

            # print("current_task_id", current_task_id)
            # if task_id == 9:
            #     pdb.set_trace()

            # TODO execute the best previously learned skill in sim
            facility = ("ROBOT", "skill library")
            test_task = task
            # TODO simulate
            best_execution_cost = np.inf
            best_skill, best_pref = None, None
            logger.debug("  Computing best skill for ROBOT facility")
            for skill in self.robot_skills:
                for test_pref_id, test_pref in enumerate(pref_space):
                    train_task, train_pref = skill["task"], skill["pref"]
                    execution_cost = (
                        1 - Pi_transfer(train_task, test_task,
                                        train_pref, test_pref)
                    ) * self.cost_cfg["FAIL"]
                    logger.debug(f"  train_pref, test_pref: {train_pref}, {test_pref}")
                    # pref_cost = (
                        # 1 - pref_similarity_fn(train_pref, test_pref)
                    # ) * self.cost_cfg["PREF_COST"]
                    pref_belief = pref_beliefs[task_id]
                    pref_cost = (1 - pref_belief[test_pref_id]) * self.cost_cfg["PREF_COST"]
                    if execution_cost + pref_cost < best_execution_cost:
                        best_execution_cost = execution_cost + pref_cost
                        best_skill, best_pref = skill, test_pref
                # save the best pref for use during execution
                best_prefs[(facility, task_id)] = best_pref
                best_skills[(facility, task_id)] = best_skill
            if np.isinf(best_execution_cost):
                logger.debug(
                    "  Skipping ROBOT facility as it has infinite cost")
            else:
                if facility not in facilities:
                    facilities.append(facility)
                    setup_costs[facility] = 0  # already learned
                logger.debug(f"  Adding ROBOT facility  for task {task_id}")
                service_costs[(facility, task_id)] = (
                    self.cost_cfg["ROBOT"] + best_execution_cost
                )

            """
            Alg 2: Pref facilities
            ----------------------

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
            """

        # N tasks = demands
        solver_info = self._solve_facility_location(
            demands, facilities, setup_costs, service_costs, current_task_id
        )
        assignments = solver_info["assignments"]
        print("assignments", assignments)
        # construct a plan based on facility location assingnments
        plan = [None] * len(task_seq)
        for key in assignments:
            facility, demand = key[0], int(key[1])
            facility_type, train_task = facility
            plan[demand] = {"action_type": facility_type,
                            "service_cost": service_costs[key]}
            if facility_type == "ASK_SKILL":
                train_task = int(train_task.split("-")[-1])
                if demand != train_task:
                    plan[demand]["skill_id"] = train_task
                    plan[demand]["pref"] = best_prefs[key]
                else:
                    # reqeust skill with specific pref param
                    # pdb.set_trace()
                    # print("key", key)
                    # print("demand", demand)
                    # print("len(keys)", best_prefs.keys())
                    # if key not in best_prefs:
                    #     __import__("ipdb").set_trace()
                    plan[demand]["pref"] = best_prefs[key]
            elif facility_type == "ROBOT":
                try:
                    plan[demand]["skill_id"] = best_skills[key]["skill_id"]
                    plan[demand]["pref"] = best_prefs[key]
                except KeyError:
                    __import__("ipdb").set_trace()
        plan = plan[current_task_id:]
        logger.debug(f"  Plan: {pformat(plan)}")
        # __import__('ipdb').set_trace()
        logger.debug(f"  Plan cost: {solver_info['cost']}")
        # __import__("ipdb").set_trace()
        plan_info = solver_info

        return plan, plan_info

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
    def _solve_facility_location(
        self, demands, facilities, setup_costs, service_costs, demand_start_index
    ):
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
        model.setParam("OutputFlag", 0)

        # __import__("ipdb").set_trace()
        select = model.addVars(facilities, vtype=GRB.BINARY, name="Select")
        assign = model.addVars(service_costs, ub=1,
                               vtype=GRB.CONTINUOUS, name="Assign")

        for key in service_costs:
            f, c = key[0], key[1]
            model.addConstr(assign[key] <= select[f], name="Setup2ship")

        for i, _ in enumerate(demands, start=demand_start_index):
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

        solver_info = {}

        if model.status == GRB.OPTIMAL:
            selected_facilities = {
                f: select[f].X for f in facilities if select[f].X > 0.5
            }
            assignments = {
                key: assign[key].X for key in service_costs if assign[key].X > 0.5
            }

            total_cost = model.objVal
            logger.debug("OPTIMAl facility location plan found.")

            solver_info["cost"] = total_cost
            solver_info["assignments"] = assignments
            return solver_info
        else:
            logger.warning("No optimal solution found.")
            __import__("ipdb").set_trace()
            return None


class FacilityLocationPrefPlanner(FacilityLocationPlanner):
    """
    Uses the facility location planner to decide among learn, human and robot.
    Decides whether or not to ask for preference by computing expected
    improvement in objective due to belief update.
    """

    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)

    def plan(
        self,
        task_seq,
        pref_beliefs,
        task_similarity_fn,
        pref_similarity_fn,
        current_task_id,
    ):
        plan1, plan_info1 = super().plan(
            task_seq,
            pref_beliefs,
            task_similarity_fn,
            pref_similarity_fn,
            current_task_id,
        )
        # return plan1, plan_info1
        # compute plan with updated belief
        pref_belief = pref_beliefs[current_task_id]
        logger.debug(f"  Pref belief: {pref_beliefs[current_task_id:]}")

        possible_costs = []
        for possible_pref_response in self.interaction_env.pref_space:
            # update belief
            belief_estimator = copy(self.belief_estimator)
            belief_estimator.beliefs = deepcopy(pref_beliefs)
            belief_estimator.update_beliefs(
                current_task_id, {"pref": possible_pref_response}
            )
            new_pref_beliefs = belief_estimator.beliefs
            logger.debug(f"  New pref beliefs: {new_pref_beliefs[current_task_id:]}")
            # __import__('ipdb').set_trace()
            plan2, plan_info2 = super().plan(
                task_seq,
                new_pref_beliefs,
                task_similarity_fn,
                pref_similarity_fn,
                current_task_id,
            )
            possible_costs.append(plan_info2["cost"])
            logger.debug(f"  Considering possible preference response: {possible_pref_response}")
            logger.debug(f"  Plan 2 cost: {plan_info2['cost']}")
            # XXX this is a hack
            # expected_improvement = plan_info1["cost"] - plan_info2["cost"]
            # if expected_improvement > 0:
            # return plan2, plan_info2
        # assume robot asks for preference
        expected_possible_cost = np.dot(possible_costs, pref_belief)
        logger.debug(f"    Expected cost after pref: {expected_possible_cost} vs {plan_info1['cost']} now ")
        if expected_possible_cost + self.cost_cfg["ASK_PREF"] < plan_info1["cost"]:
            logger.debug("  Asking for preference")
            # __import__('ipdb').set_trace()
            plan = plan1
            plan[0] = {"action_type": "ASK_PREF"}
            plan_info = plan_info1
            plan_info["cost"] = expected_possible_cost + self.cost_cfg["ASK_PREF"]
            return plan, plan_info
        else:
            logger.debug("  Not asking for preference")
            return plan1, plan_info1


class ConfidenceBasedFacilityLocationPlanner(FacilityLocationPlanner):
    """
    Uses the facility location planner to decide among learn, human and robot.
    Decides whether or not to ask for preference based on the confidence in the
    preference belief.
    """

    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)

    def plan(
        self,
        task_seq,
        pref_beliefs,
        task_similarity_fn,
        pref_similarity_fn,
        current_task_id,
    ):
        plan, plan_info = super().plan(
            task_seq,
            pref_beliefs,
            task_similarity_fn,
            pref_similarity_fn,
            current_task_id,
        )
        # return plan1, plan_info1
        # compute plan with updated belief
        pref_belief = pref_beliefs[current_task_id]
        logger.debug(f"  Pref belief: {pref_beliefs[current_task_id:]}")
        if np.max(pref_belief) < self.planner_cfg["confidence_threshold"]:
            logger.debug("  Asking for preference")
            plan[0] = {"action_type": "ASK_PREF"}
        return plan, plan_info


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

        def task_similarity_fn(x, y):
            return np.exp(-np.linalg.norm(x - y))

        pref_similarity_fn = task_similarity_fn
        planner.plan(task_seq, task_similarity_fn, pref_similarity_fn)
