from typing_extensions import Annotated, Literal
from typing import (Union, Any, Optional)
import enum 
import base64 
from tensorpc.core.datamodel.asdict import DataClassWithUndefined

import numpy as np 
from pydantic import field_validator

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import Undefined, undefined
from tensorpc.core.datamodel import typemetas
from tensorpc.dock.core.component import (UIType, FrontendEventType)
from collections.abc import Sequence
from tensorpc.dock.core import colors
from tensorpc.dock.core.appcore import Event, EventDataType
from tensorpc.dock.core.common import handle_standard_event
from tensorpc.core.datamodel.typemetas import RangedFloat, RangedInt
from tensorpc.dock.components.mui import (Image as MUIImage)

from .base import PyDanticConfigForNumpy, NumberType, Object3dContainerBaseProps, ThreeComponentBase, Object3dBaseProps, Vector3Type, ThreeLayoutType, ThreeComponentType, O3dContainerWithEventBase



class BufferMeshControlType(enum.Enum):
    UpdateBuffers = 0
    CalculateVertexNormals = 1


@dataclasses.dataclass(config=PyDanticConfigForNumpy)
class BufferMeshProps(Object3dContainerBaseProps):
    initialBuffers: Union[dict[str, np.ndarray], Undefined] = undefined
    initialIndex: Union[np.ndarray, Undefined] = undefined
    limit: Union[int, Undefined] = undefined
    initialCalcVertexNormals: Union[bool, Undefined] = undefined


@dataclasses.dataclass(config=PyDanticConfigForNumpy)
class BufferMeshUpdate(DataClassWithUndefined):
    data: np.ndarray
    offset: Union[int, Undefined] = undefined
    # newCount: Union[int, Undefined] = undefined


class BufferMesh(O3dContainerWithEventBase[BufferMeshProps,
                                           ThreeComponentType]):

    def __init__(
            self,
            initial_buffers: dict[str, np.ndarray],
            limit: int,
            children: ThreeLayoutType,
            initial_index: Union[np.ndarray, Undefined] = undefined) -> None:
        """initialIndex and initialBuffers must be specified in init,
        they can't be setted in update_event.
        WARNING: this element should only be used for advanced usage.
        if you use this with wrong inputs, the frontend may crash. 
        Args:
            initial_index: if undefined, user can't setted in update_buffers.
            initial_buffers: dict of threejs buffer attributes.
                if unsupported data format (for float, only f32 supported),
                will be casted to f32 implicitly.
        """
        first_dim = -1
        for k, v in initial_buffers.items():
            assert v.shape[0] <= limit, "initial buffer size exceeds limit"
            if first_dim == -1:
                first_dim = v.shape[0]
            else:
                assert first_dim == v.shape[0], "buffer size mismatch"
            if v.dtype == np.float16 or v.dtype == np.float64:
                initial_buffers[k] = v.astype(np.float32)
        # TODO children must be material or Edges
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreeBufferMesh, BufferMeshProps, children)
        self.props.initialBuffers = initial_buffers
        self.props.limit = limit
        self.props.initialIndex = initial_index
        self.initial_buffers = initial_buffers
        self.initial_index = initial_index

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def calc_vertex_normals_in_frontend(self):
        res = {
            "type": BufferMeshControlType.CalculateVertexNormals.value,
        }
        return await self.send_and_wait(self.create_comp_event(res))

    async def update_buffers(self,
                             updates: dict[str, BufferMeshUpdate],
                             update_bound: bool = False,
                             new_index: Optional[np.ndarray] = None,
                             new_count: Optional[int] = None):
        """
        Args: 
            updates: contains the updates for each buffer, the key must be in initialBuffers.
            update_bound: if true, the bound will be updated. user should update this when they 
                change the position.
            new_index: if not None, the index will be updated.
        """
        if isinstance(self.initial_index, Undefined):
            assert new_index is None, "new_index must be None"
        assert not isinstance(self.props.limit, Undefined)
        updates_dict = {}
        for k, v in updates.items():
            assert k in self.initial_buffers, "key not found"
            if v.data.dtype == np.float16 or v.data == np.float64:
                v.data = v.data.astype(np.float32)
            offset = v.offset
            if isinstance(offset, Undefined):
                offset = 0
            assert offset + v.data.shape[
                0] <= self.props.limit, "update size exceeds limit"
            updates_dict[k] = v.get_dict()
        res = {
            "type": BufferMeshControlType.UpdateBuffers.value,
            "updates": updates_dict,
            "updateBound": update_bound,
        }
        if new_index is not None:
            res["newIndex"] = new_index
        if new_count is not None:
            assert self.props.limit >= new_count
            res["newCount"] = new_count
        return await self.send_and_wait(self.create_comp_event(res))


@dataclasses.dataclass(config=PyDanticConfigForNumpy)
class VoxelMeshProps(Object3dContainerBaseProps):
    size: Union[NumberType, Undefined] = undefined
    centers: Union[np.ndarray, Undefined] = undefined
    colors: Union[np.ndarray, Undefined] = undefined
    limit: Union[int, Undefined] = undefined


class VoxelMesh(O3dContainerWithEventBase[VoxelMeshProps, ThreeComponentType]):

    def __init__(self,
                 centers: np.ndarray,
                 size: float,
                 limit: int,
                 children: ThreeLayoutType,
                 colors: Union[np.ndarray, Undefined] = undefined) -> None:
        if not isinstance(colors, Undefined):
            assert centers.shape[0] == colors.shape[
                0], "centers and colors must have same length"
        assert centers.shape[0] <= limit
        if centers.dtype != np.float32:
            centers = centers.astype(np.float32)
        # TODO children must be material or Edges
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreeVoxelMesh, VoxelMeshProps, children)
        self.props.limit = limit
        self.props.colors = colors
        self.props.size = size
        self.props.centers = centers

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass(config=PyDanticConfigForNumpy)
class InstancedMeshProps(Object3dContainerBaseProps):
    transforms: Union[np.ndarray, Undefined] = undefined
    scales: Union[np.ndarray, Undefined] = undefined
    colors: Union[np.ndarray, Undefined] = undefined
    limit: Union[int, Undefined] = undefined
    # when your data have special form such as 2d aabb, you can use faster raycaster.
    # 2d_aabb: your transforms must be (n, 3) array and located in a z = constant plane.
    raycaster: Union[Literal["2d_aabb"], Undefined] = undefined


class InstancedMesh(O3dContainerWithEventBase[InstancedMeshProps,
                                              ThreeComponentType]):

    def __init__(self,
                 transforms: np.ndarray,
                 limit: int,
                 children: ThreeLayoutType,
                 colors: Union[np.ndarray, Undefined] = undefined,
                 scales: Union[np.ndarray, Undefined] = undefined,) -> None:
        """
        Args:
            transforms: (n, 4, 4) or (n, 7) or (n, 3) array, 
                for (n, 4, 4) array, each 4x4 matrix is a transform.
                for (n, 7) array, each row is [x, y, z, qx, qy, qz, qw].
                for (n, 3) array, each row is [x, y, z].
            colors: (n, 3) array, each row is a color.
        """
        if not isinstance(colors, Undefined):
            assert transforms.shape[0] == colors.shape[
                0], "centers and colors must have same length"
        assert transforms.shape[0] <= limit
        assert transforms.ndim == 2 or transforms.ndim == 3
        if transforms.ndim == 2:
            assert transforms.shape[1] == 3 or transforms.shape[1] == 2 or transforms.shape[1] == 7
        if transforms.ndim == 3:
            assert transforms.shape[1] == 4 and transforms.shape[2] == 4
        if transforms.dtype != np.float32:
            transforms = transforms.astype(np.float32)
        # TODO children must be material or Edges
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreeInstancedMesh, InstancedMeshProps,
                         children)
        self.props.limit = limit
        self.props.colors = colors
        self.props.transforms = transforms
        self.props.scales = scales

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)
