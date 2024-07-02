"""A class for simple manipulation objects."""

from typing import Dict, List, Tuple, Union

import numpy as np


class Object:
    """An object to be manipulated."""

    def __init__(
        self,
        pose: Union[Dict, None] = None,
        dimensions: List = [4, 4],
    ):
        """Initialize the object.

        ::Inputs:
            ::category: The object category [CUP, MUG, FORK, SPOON]
            ::pose: Dict[x, y, orientation], with the TOP-left (x,y)
                    coordinate as origin, and orientation. In practice,
                    orientation is one of 4 axis-aligned angles
            ::dimensions: [x, y] dimensions

        """
        self.dimensions = dimensions
        if pose is None:
            pose = {
                "x": 0,
                "y": 0,
                "orientation": 0.0,
            }
            self.pose = pose
        else:
            self.pose = pose
        self.set_coordinates()

    @property
    def dimensions(self) -> List:
        """Get Object dimensions."""
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dims: List) -> None:
        """Set Object dimensions."""
        self._dimensions = dims

    @property
    def category(self) -> str:
        """Get Object category."""
        return self._category

    @category.setter
    def category(self, obj_type: str) -> None:
        """Set Object category."""
        self._category = obj_type.upper()

    @property
    def coordinates(self) -> List:
        """Get Object coordinates."""
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coords: List) -> None:
        """Set Object coordinates."""
        self._coordinates = coords

    @property
    def pose(self) -> Dict:
        """Get Object pose."""
        return self._pose

    @pose.setter
    def pose(self, new_pose: Dict) -> None:
        """Set Object pose."""
        self._pose = new_pose

    @property
    def png(self) -> List:
        """Get Object png."""
        return self._png

    @png.setter
    def png(self, image) -> None:
        """Set Object png."""
        self._png = image

    def set_coordinates(self) -> None:
        """Set object coordinates according to pose."""
        x = np.arange(self.pose["x"], self.pose["x"] + self.dimensions[0])
        y = np.arange(self.pose["y"], self.pose["y"] + self.dimensions[1])
        xs, ys = np.meshgrid(x, y)
        self.rotate(coords=[xs, ys], degrees=self.pose["orientation"])

    def translate(self, tx: int, ty: int) -> None:
        """Translate the object."""
        x_coords = self.coordinates[0]
        y_coords = self.coordinates[1]

        x_coords += tx
        y_coords += ty

        self.coordinates = [x_coords, y_coords]

    def rotate(
        self, coords: Union[List, None] = None, degrees: int = 90
    ) -> None:
        """Rotate the object COUNTER-CLOCKWISE about origin.

        ::Inputs:
            ::coords: If not None, a list of x- and y-coordinates
                      as the output of np.meshgrid()
        Code comes (mostly) from:
            https://stackoverflow.com/questions/72906207/
            how-to-rotate-points-from-a-meshgrid-and-preserve-orthogonality
        """
        if coords is not None:
            x_coords = coords[0]
            y_coords = coords[1]
        else:
            x_coords = self.coordinates[0]
            y_coords = self.coordinates[1]

        # Get the origin of the object at its top-left
        # coordinate:
        x_origin = x_coords[0, 0]
        y_origin = y_coords[0, 0]

        # Translate s.t. top-left is (0,0):
        translated_x_coords = x_coords - x_origin
        translated_y_coords = y_coords - y_origin

        # NOW rotate:
        pts = np.column_stack(
            (translated_x_coords.flatten(), translated_y_coords.flatten())
        )
        radians = np.radians(degrees)
        rot_mat = np.array(
            [
                [np.cos(radians), -np.sin(radians)],
                [np.sin(radians), np.cos(radians)],
            ]
        )
        pts_rotated = pts @ rot_mat

        # Translate back to object origin:
        pts_rotated[:, 0] = pts_rotated[:, 0] + x_origin
        pts_rotated[:, 1] = pts_rotated[:, 1] + y_origin
        pts_rotated = pts_rotated.astype(int)

        # Make a meshgrid and set the coordinates:
        new_x_coords, new_y_coords = np.meshgrid(
            pts_rotated[:, 0], pts_rotated[:, 1]
        )
        self.coordinates = [new_x_coords, new_y_coords]


class Cup(Object):
    """A Cup object."""

    def __init__(self, pose: Dict, dimensions: List) -> None:
        """Initialize a Cup object."""
        if len(dimensions) == 0:
            dims = [4, 3]
        else:
            dims = dimensions
        super().__init__(pose=pose, dimensions=dims)

        # self.png = TODO


class Mug(Object):
    """A Mug object."""

    def __init__(
        self, pose: Union[Dict, None] = None, dimensions: List = []
    ) -> None:
        """Initialize a Mug object."""
        if len(dimensions) == 0:
            dims = [4, 3]
        else:
            dims = dimensions
        super().__init__(pose=pose, dimensions=dims)

        # self.png = TODO


class Fork(Object):
    """A Fork object."""

    def __init__(
        self, pose: Union[Dict, None] = None, dimensions: List = []
    ) -> None:
        """Initialize a Fork object."""
        if len(dimensions) == 0:
            dims = [5, 1]
        else:
            dims = dimensions
        super().__init__(pose=pose, dimensions=dims)

        # self.png = TODO


class Spoon(Object):
    """A Spoon object."""

    def __init__(
        self, pose: Union[Dict, None] = None, dimensions: List = []
    ) -> None:
        """Initialize a Spoon object."""
        if len(dimensions) == 0:
            dims = [5, 1]
        else:
            dims = dimensions
        super().__init__(pose=pose, dimensions=dims)

        # self.png = TODO
