import logging
import re

import matplotlib.pyplot as plt
import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.models.objects import (BreadObject, CanObject, CerealObject,
                                      MilkObject)
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import (SequentialCompositeSampler,
                                                UniformRandomSampler)

logger = logging.getLogger(__name__)


class PickPlaceEnv(PickPlace):
    """
    Slightly easier task where we limit z-rotation to 0 to 90 degrees for all object initializations (instead of full 360).

    Each task contains only one object which can be changed with every reset.

    Supported objects: ["bread", "milk", "cereal", "can"]
    TODO Add more objects (mugs)
    """

    def __init__(self, **kwargs):
        # initial state placeholder
        self._state = {
            "obj_qpos": [0.1, -0.5, 0.8, 1, 0, 0, 0],
            "obj_type": "bread",
        }

        super().__init__(
            single_object_mode=2,
            object_type="bread",
            z_rotation=(0.0, np.pi / 2.0),
            robots="Panda",
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=False,
            control_freq=20,
            controller_configs=load_controller_config(
                default_controller="OSC_POSE"),
            **kwargs,
        )

    def pref_space(self):
        # TODO also consider goal orientation
        return ["Bin1", "Bin2", "Bin3", "Bin4"]

    def reset_to_state(self, state):
        self._set_object_type(state["obj_type"])
        # set obj position
        self._state = state
        return self.reset()

    def _reset_internal(self):
        """
        Sets the object position
        """
        self.deterministic_reset = True
        super()._reset_internal()
        obj = self.objects[self.object_id]
        self.sim.data.set_joint_qpos(obj.joints[0], self._state["obj_qpos"])
        self.sim.forward()
        # XXX could use them to show ground truth preferences
        # Set the visual object body locations
        # if "visual" in obj.name.lower():
        # self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
        # self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
        # else:
        # Set the collision object joints

    def _set_object_type(self, object_type):
        assert (
            object_type in self.object_to_id.keys()
        ), "invalid @object_type argument - choose one of {}".format(
            list(self.object_to_id.keys())
        )
        # use for convenient indexing
        self.object_id = self.object_to_id[object_type]


class PickPlaceTaskSeqGen(PickPlace):
    """
    Generates a sequence of tasks for the PickPlace environment.
    """

    def __init__(self, task_seq_cfg, **kwargs):

        self.task_seq_cfg = task_seq_cfg
        super().__init__(
            single_object_mode=0,
            z_rotation=(0.0, np.pi / 2.0),
            robots="Panda",
            has_offscreen_renderer=True,
            ignore_done=True,
            control_freq=20,
            controller_configs=load_controller_config(
                default_controller="OSC_POSE"),
            hard_reset=True,
            use_object_obs=False,  # to avoid dealing with observables
            use_camera_obs=True,  # for saving images
            camera_names=["frontview", "birdview", "agentview"],
            camera_heights=1024,
            camera_widths=1024,
            **kwargs,
        )

    def generate_task_seq(self, n_seqs):
        """
        Generate a sequence of tasks.
        """
        task_seqs = []
        for seq_id in range(n_seqs):
            task_seq = []
            # resample the placement initializer
            obs = self.reset()
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            for i, camera_name in enumerate(self.camera_names):
                axs[i].imshow(obs[camera_name + "_image"][::-1])
                axs[i].axis("off")
            fig.savefig(f"task_seq_{seq_id}.png")
            # save each object's position and orientation sequentially
            for obj in self.objects:
                obj_type = re.findall(r"\D+", obj.name)[0].lower()
                obj_qpos = self.sim.data.get_joint_qpos(obj.joints[0])
                task = dict(obj_type=obj_type, obj_qpos=obj_qpos)
                task_seq.append(task)
            task_seqs.append(task_seq)
        return task_seqs

    def _construct_objects(self):
        self.objects = []
        num_tasks = self.task_seq_cfg["num_tasks"]
        self.objects = []
        obj_types = ["Milk", "Bread", "Cereal", "Can"]
        obj_classes = [MilkObject, BreadObject, CerealObject, CanObject]
        # sample the frequency of each object type
        dirichlet_prior = np.ones(len(obj_classes))
        obj_freq = np.random.dirichlet(dirichlet_prior)
        logger.info(f"Sampling tasks with object frequencies: {obj_freq}")
        # TODO sample in more interesting and adversarial ways
        indices = np.random.choice(
            range(len(obj_classes)), p=obj_freq, size=num_tasks)
        obj_counts = {obj_index: 0 for obj_index in range(len(obj_classes))}
        for index in indices:
            obj_cls = obj_classes[index]
            obj_name = obj_types[index] + str(obj_counts[index])
            obj_counts[index] += 1
            obj = obj_cls(name=obj_name)
            self.objects.append(obj)
