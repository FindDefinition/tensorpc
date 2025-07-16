from collections.abc import Sequence
import enum
from functools import reduce
from tensorpc.core.datamodel.draft import DraftBase
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout
from tensorpc.core import dataclass_dispatch as dataclasses, pfl
import numpy as np
from typing import Any, Optional 
from tensorpc.dock.components.plus.hud.minimap import MinimapModel
from tensorpc.dock.components.three.event import KeyboardHoldEvent, PointerEvent
from tensorpc.core.pfl.backends.js import ColorUtil, Math, MathUtil
import tensorpc.core.datamodel as D

MAX_MATRIX_SIZE = 512 * 256

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class MatrixBase:
    name: str
    width: float 
    height: float
    # tensor is always converted to a matrix, this store the shape of the tensor.
    shape: list[int]
    # vis layers
    # [N，2] aabb
    persist_fill_pos: Optional[np.ndarray] = None
    persist_fill_color: Optional[np.ndarray] = None

    # if height is too small, we scale height to get better visibility.
    height_scale: float = 1.0

    def get_vis_wh(self, padding: int = 2):
        return (self.width + padding * 2, self.height * self.height_scale + padding * 2)

def _get_matrix_shape_from_tensor_shape(tensor_shape: Sequence[int]):
    ndim = len(tensor_shape)
    if ndim < 2:
        return [1, tensor_shape[0]]
    elif ndim == 2:
        return list(tensor_shape)
    else:
        first_dim = reduce(lambda x, y: x * y, tensor_shape[:-1], 1)
        return [first_dim, tensor_shape[-1]]

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class Matrix(MatrixBase):
    # vis data
    offsetX: float 
    offsetY: float
    data: Optional[np.ndarray] = None 
    # vis layers
    temp_fill_pos: Optional[np.ndarray] = None
    temp_fill_color: Optional[np.ndarray] = None

    # [N, 2] segments
    persist_aabb_line_pos: Optional[np.ndarray] = None
    persist_aabb_line_size: Optional[np.ndarray] = None

    temp_aabb_line_pos: Optional[np.ndarray] = None
    temp_aabb_line_size: Optional[np.ndarray] = None
    
    temp_mask_pos: Optional[np.ndarray] = None
    # currently impossible to use datamodel to control uniforms of shader.
    # temp_mask_color1: str = "sliver"
    # temp_mask_color2: str = "gray"
    # temp_mask_opacity1: float = 0.6
    # temp_mask_opacity2: float = 0.1


    linePosX: Optional[float] = None
    linePosY: Optional[float] = None
    fontSize: float = 3

    @classmethod 
    def from_array(cls, name: str, arr: np.ndarray, padding: int = 2):
        return cls.from_shape(name, list(arr.shape), padding)

    @classmethod 
    def from_shape(cls, name: str, shape: list[int], padding: int = 2):
        shape = _get_matrix_shape_from_tensor_shape(shape)
        raw_shape = shape
        width = shape[1] 
        height = shape[0]
        assert width * height <= MAX_MATRIX_SIZE, f"Matrix size {width}x{height} exceeds maximum size {MAX_MATRIX_SIZE}"
        res = cls(
            name=name,
            width=width,
            height=height,
            shape=raw_shape,
            offsetX=0.0,
            offsetY=0.0,
        )

        desc_length = res.get_desc_length()
        layout_w, layout_h = res.get_vis_wh(padding=2)
        res.fontSize = max(1 + padding, layout_w / desc_length)
        return res 

    def get_desc_length(self):
        shape_str = ",".join(str(x) for x in self.shape)
        desc = f"{self.name}|[{shape_str}]"
        return len(desc)

    @pfl.mark_pfl_compilable
    def _on_hover_pfl(self, data: PointerEvent):
        point_unified_x = data.pointLocal[0] + 0.5
        point_unified_y = -data.pointLocal[1] + 0.5
        idx_x = Math.floor(point_unified_x * self.width)
        idx_y = Math.floor(point_unified_y * self.height)
        self.linePosX = (idx_x + 0.5) - self.width / 2
        self.linePosY = ((-(idx_y + 0.5)) + self.height / 2) * self.height_scale

    @staticmethod 
    def get_value_pos_and_color_gray(tensor: np.ndarray, height_scale: float = 1.0):
        shape = _get_matrix_shape_from_tensor_shape(tensor.shape)
        mat = tensor.reshape(shape)
        if mat.dtype.kind != "f":
            mat = mat.astype(np.float32)
        mat_max = np.max(mat)
        mat_min = np.min(mat)
        width = shape[1]
        height = shape[0]
        mat_flat_inds = np.arange(mat.size, dtype=np.int32)
        fill_pos_x = (mat_flat_inds % width) + 0.5 - width / 2
        fill_pos_y = -(np.floor(mat_flat_inds / width) + 0.5 - height / 2)
        fill_pos = np.stack([fill_pos_x, fill_pos_y * height_scale], axis=-1)

        # get gray color based on value
        mat_flat = mat.flatten()
        color_res = np.zeros((mat_flat.size, 3), dtype=np.float32)
        if mat_max != mat_min:
            
            mat_flat = (mat_flat - mat_min) / (mat_max - mat_min)
            color_res[:, 0] = mat_flat  # R
            color_res[:, 1] = mat_flat  # G
            color_res[:, 2] = mat_flat  # B
        else:
            color_res[:, 0] = 0.5  # R
            color_res[:, 1] = 0.5  # G
            color_res[:, 2] = 0.5  # B

        return fill_pos, color_res

    def get_global_fill(self, global_key: str, inds: np.ndarray, is_persist: bool = True):
        inds_flat = inds.reshape(-1).astype(np.float32)
        fill_pos_x = (inds_flat % self.width) + 0.5 - self.width / 2
        fill_pos_y = -(np.floor(inds_flat / self.width) + 0.5 - self.height / 2)
        fill_pos = np.stack([fill_pos_x, fill_pos_y * self.height_scale], axis=-1)
        fill_color = np.empty([inds_flat.shape[0], 3], np.float32)
        if is_persist:
            color = ColorUtil.getPerfettoColorRGB(global_key)
        else:
            color = ColorUtil.getPerfettoVariantColorRGB(global_key)

        fill_color[:, 0] = color[0] / 255
        fill_color[:, 1] = color[1] / 255
        fill_color[:, 2] = color[2] / 255
        return fill_pos,  fill_color

@dataclasses.dataclass
class GlobalMemoryModel:
    matrices: dict[str, Matrix]
    minimap: plus.hud.MinimapModel

    @staticmethod 
    def empty():
        return GlobalMemoryModel(
            matrices={},
            minimap=plus.hud.MinimapModel(1, 1)
        )


class GlobalMemLayers(enum.IntEnum):
    BKGD = -8
    PERSIST_FILL = -7
    TEMP_FILL = -6
    PERSIST_LINE = -5
    TEMP_LINE = -4
    TEMP_MASK = -3

    TEXT = -2
    INDICATOR = -1

class GlobalMatrixPanel(three.Group):
    def __init__(self, draft: Matrix, enable_hover_line: bool = False):
        assert isinstance(draft, DraftBase)
        trs_empty = np.zeros((0, 2), dtype=np.float32)
        lines_empty = np.zeros((0, 2), dtype=np.float32)

        self._event_plane = three.Mesh([
            three.PlaneGeometry(1, 1),
            three.MeshBasicMaterial().prop(transparent=True, opacity=0.0),
        ]).prop(position=(0, 0, int(GlobalMemLayers.BKGD)))

        self._event_plane.bind_fields_unchecked_dict({
            "scale-x": draft.width,
            "scale-y": draft.height,
        })
        self._hover_line = three.LineShape(three.Shape.from_aabb(0, 0, 1, 1))
        self._hover_line.prop(color="blue", lineWidth=1, position=(0, 0, int(GlobalMemLayers.INDICATOR)))
        self._hover_line_cond = mui.MatchCase.binary_selection(True, self._hover_line)
        if enable_hover_line:
            self._hover_line.bind_fields_unchecked_dict({
                "position-x": draft.linePosX,
                "position-y": draft.linePosY,
            })
            dm = mui.DataModel.get_datamodel_from_draft(draft)
            self._hover_line_cond.bind_fields(condition=f"{draft.linePosX} != `null`")
            self._event_plane.event_leave.add_frontend_draft_set_none(draft, "linePosX")
            self._event_plane.event_move.add_frontend_handler_v2(dm, Matrix._on_hover_pfl, targetPath=str(draft))

        self._border = three.LineShape(three.Shape.from_aabb(0, 0, 1, 1))
        self._border.prop(color="black", lineWidth=2)
        self._border.bind_fields_unchecked_dict({
            "scale-x": draft.width,
            "scale-y": draft.height,
        }).prop(position=(0, 0, int(GlobalMemLayers.BKGD)))
        fill_material = three.MeshShaderMaterial([
            three.ShaderUniform("color2", three.ShaderUniformType.Color, "white"),
            three.ShaderUniform("mask_color1", three.ShaderUniformType.Color, "silver"),
            three.ShaderUniform("mask_color2", three.ShaderUniformType.Color, "white"),
            three.ShaderUniform("mask_distance", three.ShaderUniformType.Number, 5.0),

            three.ShaderUniform("opacity1", three.ShaderUniformType.Number, 1.0),
            three.ShaderUniform("opacity2", three.ShaderUniformType.Number, 1.0),

        ], f"""
        varying vec3 localPosition;
        varying vec3 vInstanceColor;
        varying vec3 worldPosition;

        void main() {{
            localPosition = position;
            vInstanceColor = instanceColor;
            worldPosition = (instanceMatrix * vec4(position, 1.0)).xyz;
            gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
        }}
        """, f"""
        varying vec3 worldPosition;
        varying vec3 vInstanceColor;
        varying vec3 localPosition;

        uniform vec3 color2;
        uniform vec3 mask_color1;
        uniform vec3 mask_color2;

        uniform float opacity1;
        uniform float opacity2;
        uniform float mask_distance;

        void main() {{
            // normal part
            vec2 uv = localPosition.xy * 0.5 + 0.5;
            vec2 uv1 = vec2(uv.x, 1.0 - uv.y);
            vec2 uv2 = vec2(1.0 - uv.x, uv.y);
            vec4 color1Alpha = vec4(vInstanceColor, opacity1);
            vec4 color2Alpha = vec4(color2, opacity2);
            vec4 normalColor = mix(color1Alpha, color2Alpha, uv1.x * uv2.y);
            normalColor.a = max(normalColor.a, 0.01);  // ensure not fully transparent
            // mask part
            float c = worldPosition.y - worldPosition.x;
            // unify c to [0, distance] range (like c % distance)
            c = mod(c + mask_distance, mask_distance);
            vec4 maskColor1Alpha = vec4(mask_color1, opacity1);
            vec4 maskColor2Alpha = vec4(mask_color2, opacity2);
            vec4 maskColor = c > (mask_distance / 2.0) ? maskColor1Alpha : maskColor2Alpha;
            // when rgb all zero, use mask color, otherwise use normal color.
            if (normalColor.rgb == vec3(0.0)) {{
                gl_FragColor = maskColor;
            }} else {{
                gl_FragColor = normalColor;
            }}
            #include <tonemapping_fragment>
            #include <colorspace_fragment>
        }}
        """)
        self._persist_fill = three.InstancedMesh(trs_empty, MAX_MATRIX_SIZE, [
            three.PlaneGeometry(),
            # three.MeshBasicMaterial(),
            fill_material,

        ]).prop(position=(0, 0, int(GlobalMemLayers.PERSIST_FILL)))
        self._temp_fill = three.InstancedMesh(trs_empty, MAX_MATRIX_SIZE, [
            three.PlaneGeometry(),
            fill_material,
        ]).prop(position=(0, 0, int(GlobalMemLayers.TEMP_FILL)))
        # 45 degree rotated masks.
        self._temp_mask = three.InstancedMesh(trs_empty, MAX_MATRIX_SIZE, [
            three.PlaneGeometry(),
            three.MeshShaderMaterial([
                three.ShaderUniform("distance", three.ShaderUniformType.Number, 5.0),

                three.ShaderUniform("color1", three.ShaderUniformType.Color, "silver"),
                three.ShaderUniform("color2", three.ShaderUniformType.Color, "white"),
                three.ShaderUniform("opacity1", three.ShaderUniformType.Number, 0.8),
                three.ShaderUniform("opacity2", three.ShaderUniformType.Number, 0.1),
            ], f"""
            varying vec3 worldPosition;

            void main() {{
                worldPosition = (instanceMatrix * vec4(position, 1.0)).xyz;
                gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
            }}
            """, f"""
            varying vec3 worldPosition;
            uniform vec3 color1;
            uniform vec3 color2;
            uniform float opacity1;
            uniform float opacity2;
            uniform float distance;

            void main() {{
                float c = worldPosition.y - worldPosition.x;
                // unify c to [0, distance] range (like c % distance)
                c = mod(c + distance, distance);
                vec4 color1Alpha = vec4(color1, opacity1);
                vec4 color2Alpha = vec4(color2, opacity2);
                vec4 color = c > distance / 2.0 ? color1Alpha : color2Alpha;
                gl_FragColor = color;
                #include <tonemapping_fragment>
                #include <colorspace_fragment>
            }}
            """).prop(transparent=True),
        ]).prop(position=(0, 0, int(GlobalMemLayers.TEMP_MASK)))

        self._persist_lines = three.Line(lines_empty).prop(position=(0, 0, int(GlobalMemLayers.PERSIST_LINE)), 
            color="green", lineWidth=1, opacity=0.7, segments=True, variant="aabb")
        self._temp_lines = three.Line(lines_empty).prop(position=(0, 0, int(GlobalMemLayers.TEMP_LINE)), 
            color="aqua", lineWidth=1, opacity=0.7, segments=True, variant="aabb")

        self._label = three.Text("").prop(position=(0, 0, int(GlobalMemLayers.TEXT)), color="blue", fillOpacity=0.5)
        self._label.bind_fields(fontSize=draft.fontSize)
        self._persist_fill.bind_fields(transforms=draft.persist_fill_pos, colors=draft.persist_fill_color)
        self._temp_fill.bind_fields(transforms=draft.temp_fill_pos, colors=draft.temp_fill_color)
        self._persist_lines.bind_fields(points=draft.persist_aabb_line_pos, aabbSizes=draft.persist_aabb_line_size)
        self._temp_lines.bind_fields(points=draft.temp_aabb_line_pos, aabbSizes=draft.temp_aabb_line_size)
        self._label.bind_fields(value=f"cformat('%s|%s', {draft.name}, to_string({draft.shape}))")
        self._temp_mask.bind_fields(transforms=draft.temp_mask_pos)

        super().__init__([
            self._persist_fill,
            self._temp_fill,
            self._persist_lines,
            self._temp_lines,
            self._border,
            self._event_plane,
            self._label,
            self._hover_line_cond,
            self._temp_mask,
        ])
        self.bind_fields_unchecked_dict({
            "position-x": draft.offsetX,
            "position-y": f"-{draft.offsetY}",
        })


class GlobalMemContainer(mui.FlexBox):
    def __init__(self, init_matrices: Optional[dict[str, np.ndarray]] = None, 
            external_dm: Optional[mui.DataModel] = None, 
            external_draft: Optional[GlobalMemoryModel] = None,
            use_view: bool = False):
        if init_matrices is not None:
            assert external_draft is None

            matrices, max_width, max_height = self._get_global_matrix(init_matrices)
            empty_model = GlobalMemoryModel(
                matrices, MinimapModel(max_width, max_height)
            )
            dm = mui.DataModel(empty_model, [])
            draft = dm.get_draft()
            minimap = plus.hud.MiniMap(draft.minimap, {
                k: GlobalMatrixPanel(draft.matrices[k]) for k, v in matrices.items()
            })
        else:
            if external_draft is not None:
                assert isinstance(external_draft, DraftBase)
                assert external_dm is not None 
                dm = external_dm
                draft = external_draft
            else:
                empty_model = self._create_empty_vis_model()
                dm = mui.DataModel(empty_model, [])
                draft = dm.get_draft()
            minimap = plus.hud.MiniMap(draft.minimap, [])
        self.minimap = minimap
        self._draft = draft
        cam = three.OrthographicCamera(near=0.1, far=1000, children=[
            minimap,
        ]).prop(position=(0, 0, 10))
        if use_view:
            canvas = three.View([
                cam.prop(makeDefault=True),
            ]).prop(allowKeyboardEvent=True)
        else:
            canvas = three.Canvas([
                cam.prop(makeDefault=True),
            ]).prop(allowKeyboardEvent=True)
        minimap.install_canvas_events(draft.minimap, canvas)
        self._dm = dm
        if external_dm is None:
            dm.init_add_layout([
                canvas.prop(flex=1),
            ])
            # self.dm = dm
            layout = [dm]
        else:
            layout = [canvas.prop(flex=1)]
        super().__init__(layout)
        self.prop(minHeight=0,
                minWidth=0,
                flexFlow="row nowrap",
                width="100%",
                height="100%",
                overflow="hidden")

    def _get_global_matrix(self, matrices: dict[str, np.ndarray]):
        padding = 1
        cur_offset_y = 0
        max_width = 1
        matrixe_objs: dict[str, Matrix] = {}
        for k, v in matrices.items():
            gmat = Matrix.from_array(k, v)
            layout_w, layout_h = gmat.get_vis_wh(padding)
            gmat.offsetX = layout_w / 2
            gmat.offsetY = cur_offset_y + layout_h / 2
            gmat.temp_fill_pos = np.zeros((1, 3), dtype=np.float32)
            gmat.temp_fill_color = np.array([0.4, 0.7, 0.1], dtype=np.float32)
            cur_offset_y += layout_h
            matrixe_objs[k] = gmat
            max_width = max(max_width, layout_w)
        for k, v in matrixe_objs.items():
            v.offsetX -= max_width / 2
            v.offsetY -= cur_offset_y / 2

        return matrixe_objs, max_width, cur_offset_y


    async def set_matrix_dict(self, matrices: dict[str, np.ndarray]):
        matrixe_panels: dict[str, GlobalMatrixPanel] = {}
        gmatrices, max_width, max_height = self._get_global_matrix(matrices)
        async with self._dm.draft_update():
            for k, v in gmatrices.items():
                self._draft.matrices[k] = v
                matrixe_panels[k] = GlobalMatrixPanel(self._draft.matrices[k])
            self._draft.minimap.width = max_width
            self._draft.minimap.height = max_height

        await self.minimap.set_new_childs(matrixe_panels)


    def _create_empty_vis_model(self) -> GlobalMemoryModel:
        return GlobalMemoryModel(
            {}, 
            plus.hud.MinimapModel(1, 1))
