import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.pick_place import PickPlace


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
            "obj_pos": [0.1, -0.5, 0.8],
            "obj_quat": [1, 0, 0, 0],
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
        self.sim.data.set_joint_qpos(
            obj.joints[0],
            np.concatenate(
                [np.array(self._state["obj_pos"]),
                 np.array(self._state["obj_quat"])]
            ),
        )
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
