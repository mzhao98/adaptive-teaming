from copy import deepcopy
import re
from time import sleep

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.utils.sim_utils import check_contact


class PickPlaceExpertSkill:
    # def __init__(self):

    def step(self, env, pref_params, obs, render=True):
        # TODO: choose a plan based on the most likely goal
        total_rew = 0
        gamma, discount = 1.0, 1.0
        obj_name = env.obj_to_use
        obj_pos = obs[f"{obj_name}_pos"]
        obj = env.objects[env.object_id]
        # obj_height = obj.get_bounding_box_size()[2]
        # obj_height = obj.top_offset[2]
        bin_height = 0.05 + 0.01 #(buffer)
        bin_z = env.bin1_pos[2]

        # self._check_collision(env, obj)

        def get_gripper_pos():
            return deepcopy(env.sim.data.site_xpos[env.robots[0].eef_site_id])

        def get_gripperr_ori():
            ori = T.quat2axisangle(
                T.mat2quat(
                    np.array(
                        env.sim.data.site_xmat[
                            env.sim.model.site_name2id(
                                env.robots[0].gripper.naming_prefix +
                                "grip_site"
                            )
                        ]
                    ).reshape(3, 3)
                )
            )
            return deepcopy(ori)

        ee_ori = get_gripperr_ori()

        # move on top of the object
        target_pos = obj_pos + np.array([0, 0, 0.1])
        for i in range(50):
            gripper_action = [-1]
            action = np.concatenate([target_pos, ee_ori, gripper_action])
            obs, rew, done, info = env.step(action)
            if render:
                env.render()

        obj_pos = obs[f"{obj_name}_pos"]

        # go down
        z_offset = max(0, obj.top_offset[2] - 0.05)
        target_pos = obj_pos + np.array([0, 0, z_offset])
        for i in range(50):
            action = np.concatenate([target_pos, ee_ori, [-1]])
            obs, rew, done, info = env.step(action)
            if render:
                env.render()

        # pickup the object
        for _ in range(20):
            action = np.concatenate([target_pos, ee_ori, [1]])
            obs, rew, done, info = env.step(action)
            if render:
                env.render()

        # go up
        obj_pos = obs[f"{obj_name}_pos"]
        target_pos = get_gripper_pos()
        z_target= obj_pos[2] + abs(obj.bottom_offset[2]) + bin_height + 0.02
        target_pos[2] = z_target
        for _ in range(50):
            action = np.concatenate([target_pos, ee_ori, [1]])
            # print("target", np.round(target_pos, 2))
            obs, rew, done, info = env.step(action)
            sleep(0.01)
            if render:
                env.render()

        # travel to the goal
        target_bin = int(re.findall(r"\d+", pref_params)[0])
        target_bin_pos = env.target_bin_placements[target_bin]
        # target_pos = target_bin_pos + np.array([0, 0, 0.25 + obj_height])
        target_pos = target_bin_pos
        target_pos[2] = z_target
        for _ in range(50):
            action = np.concatenate([target_pos, ee_ori, [1]])
            # print("target", np.round(target_pos, 2))
            obs, rew, done, info = env.step(action)
            if render:
                env.render()

        # go down and release the object
        # target_pos = target_bin_pos + np.array([0, 0, 0.12])
        target_pos = get_gripper_pos() + np.array(
            [0, 0, -obj.get_bounding_box_half_size()[2]]
        )
        for _ in range(50):
            action = np.concatenate([target_pos, ee_ori, [1]])
            obs, rew, done, info = env.step(action)
            if render:
                env.render()

        for _ in range(20):
            action = np.concatenate([target_pos, ee_ori, [-1]])
            obs, rew, done, info = env.step(action)
            if render:
                env.render()

        # go up
        target_pos = get_gripper_pos() + np.array([0, 0, 0.1])
        for _ in range(20):
            action = np.concatenate([target_pos, ee_ori, [-1]])
            obs, rew, done, info = env.step(action)
            if render:
                env.render()

        info["safety_violated"] = False
        total_rew = 0

        # XXX we are not optimizing for the reward here.
        # The true reward is the human's preference which is not available to the robot.
        return obs, total_rew, done, info

    def _check_collision(self, env, obj):
        obj_geoms = obj.contact_geoms
        bin1_geoms = env.model.mujoco_arena.bin1_body
        bin2_geoms = env.model.mujoco_arena.bin2_body
        # __import__('ipdb').set_trace()
        # # Search for collisions between each gripper geom group and the object geoms group
        # for g_group in g_geoms:
            # if not check_contact(env.sim, obj_geoms, o_geoms):
                # return False
        return False
