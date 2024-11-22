from itertools import product
import numpy as np
from pprint import pprint

import gurobipy as gp
from gurobipy import GRB


class FacilityLocationPlanner():

    def __init__(self, cost_cfg):
        self.cost_cfg = cost_cfg

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
            facility = ('hum', f"task-{task_id}")
            facilities.append(facility)
            setup_costs[facility] = self.cost_cfg['hum']

            # always succeeds
            # Note: I could either set a high cost for all the other tasks or I
            # "think" I can just ignore the other combinations by not creating
            # the corresponding decision variables.
            service_costs[(facility, task_id)] = 0

            # demo
            facility = ('demo', f"task-{task_id}")
            facilities.append(facility)
            setup_costs[facility] = self.cost_cfg['demo']

            # demo affects only current and future tasks.
            # assume demo also complete the task perfectly
            service_costs[(facility, task_id)] = 0
            for test_task_id in range(task_id + 1, N):
                service_costs[(facility, test_task_id)] = (self.cost_cfg['rob'] +
                                         (1 - Pi_transfer[task_id, test_task_id]) * self.cost_cfg['fail_cost'] +
                                         (1 - Pref_transfer[task_id, test_task_id]) * self.cost_cfg['pref_cost'])

            # pref
            # each task i creates i-1 pref facilities
            # pref on task_id with all previous demos
            for demo_task_id in range(task_id):
                facility = ('pref', f"task-{task_id}, demo-{demo_task_id}")
                # facility = ('pref', str(task_id) + '-' + str(demo_task_id))
                facilities.append(facility)
                setup_costs[facility] = self.cost_cfg['pref']

                # now compute service costs
                # XXX shouldn't the preferene also generalize to other tasks??
                # XXX this will require another for loop
                service_costs[(facility, task_id)] = (self.cost_cfg['rob'] +
                                         (1 - Pi_transfer[demo_task_id, task_id]) * self.cost_cfg['fail_cost'] +
                                         (1 - Pref_transfer[task_id, task_id]) * self.cost_cfg['pref_cost'])

        # N tasks = demands
        self._solve_facility_location(demands,
                                      facilities,
                                      setup_costs,
                                      service_costs)

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
    def _solve_facility_location(self, demands, facilities, setup_costs,
                                 service_costs):
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
        assign = model.addVars(service_costs, ub=1, vtype=GRB.CONTINUOUS, name='Assign')

        for key in service_costs:
            f, c  = key[0], key[1]
            model.addConstr(assign[key] <= select[f], name="Setup2ship")

        for i, _ in enumerate(demands):
            for f in facilities: 
                model.addConstr(gp.quicksum(assign[(f, i)] for f in facilities
                                            if (f, i) in assign) == 1, name=f"Demand-{i}")

        model.update()

        model.setObjective(select.prod(setup_costs) + assign.prod(service_costs), GRB.MINIMIZE)

        model.optimize()

        __import__('ipdb').set_trace()


if __name__ == "__main__":
    # test code
    cost_cfg = {
        "rob": 10,
        "hum": 90,
        "demo": 100,
        "pref": 20,
        "fail_cost": 100,
        "pref_cost": 50
    }

    planner = FacilityLocationPlanner(cost_cfg)
    # task_seq = np.random.rand(10, 2)
    task_seq = np.random.normal(0, 1, (10, 2))
    task_similarity_fn = lambda x, y: np.exp(-np.linalg.norm(x - y))
    pref_similarity_fn = task_similarity_fn
    planner.plan(task_seq,
                 task_similarity_fn,
                 pref_similarity_fn)
