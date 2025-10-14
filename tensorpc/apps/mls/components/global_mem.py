from collections.abc import Sequence
import enum
from functools import reduce
from tensorpc.core.datamodel.draft import DraftBase
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout
from tensorpc.core import dataclass_dispatch as dataclasses, pfl
import numpy as np
from typing import Any, Optional, Union 
from tensorpc.dock.components.plus.hud.minimap import MinimapModel
from tensorpc.dock.components.three.event import KeyboardHoldEvent, PointerEvent
from tensorpc.core.pfl.backends.js import ColorUtil, Math, MathUtil
import tensorpc.core.datamodel as D

MAX_MATRIX_SIZE = 2048 * 256

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class Label:
    text: str
    fontSize: float = 3
    offsetX: float = 0
    offsetY: float = 0

FONTSIZE_WIDTH_APPROX = 0.7

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class MatrixBase:
    name: str
    width: float 
    height: float
    widthVis: float 
    heightVis: float
    # for autolayout 
    widthLayout: float 
    heightLayout: float

    # tensor is always converted to a matrix, this store the shape of the tensor.
    shape: list[int]
    # vis layers
    # [N，2] aabb
    persist_fill_pos: Optional[np.ndarray] = None
    persist_fill_color: Optional[np.ndarray] = None

    # if height is too small, we scale height to get better visibility.
    height_scale: float = 1.0
    transposed: bool = False

    def get_vis_wh(self, padding: int = 2):
        res = (self.width + padding * 2, self.height * self.height_scale + padding * 2)
        if self.transposed:
            return res[::-1]
        return res

    @staticmethod
    def get_vis_wh_static(width: int, height: int, height_scale: int = 1, padding: int = 2, transposed: bool = False):
        res = (width + padding * 2, height * height_scale + padding * 2)
        if transposed:
            return res[::-1]
        return res

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
    selected: bool = False
    hovered: bool = False

    @classmethod 
    def empty(cls):
        return cls(
            name="",
            width=1,
            height=1,
            widthVis=1,
            heightVis=1,
            widthLayout=1,
            heightLayout=1,
            shape=[1, 1],
            offsetX=0.0, 
            offsetY=0.0,
        )


    @classmethod 
    def from_array(cls, name: str, arr: np.ndarray, padding: int = 2, transposed: bool = False, label_with_shape: bool = True):
        return cls.from_shape(name, list(arr.shape), padding, transposed, label_with_shape)

    @classmethod 
    def from_shape(cls, name: str, shape: list[int], padding: int = 2, transposed: bool = False, label_with_shape: bool = True):
        shape = _get_matrix_shape_from_tensor_shape(shape)
        raw_shape = shape
        width = shape[1] 
        height = shape[0]
        assert width * height <= MAX_MATRIX_SIZE, f"Matrix size {width}x{height} exceeds maximum size {MAX_MATRIX_SIZE}"
        res = cls(
            name=name,
            width=width,
            height=height,
            widthVis=height if transposed else width,
            heightVis=width if transposed else height,
            widthLayout=0,
            heightLayout=0,
            shape=raw_shape,
            offsetX=0.0,
            offsetY=0.0,
            transposed=transposed,
        )

        desc_length = res.get_desc_length(label_with_shape) * FONTSIZE_WIDTH_APPROX
        layout_w, layout_h = res.get_vis_wh(padding=2)
        res.widthLayout = layout_w
        res.heightLayout = layout_h
        res.fontSize = min(max(1 + padding, layout_w / desc_length), layout_h)
        return res 

    def get_desc_length(self, label_with_shape: bool):
        if not label_with_shape:
            return len(self.name)
        shape_str = ",".join(str(x) for x in self.shape)
        desc = f"{self.name}|[{shape_str}]"
        return len(desc)

    @pfl.js.mark_js_compilable
    def _on_hover_pfl(self, data: PointerEvent):
        point_unified_x = data.pointLocal[0] + 0.5
        point_unified_y = -data.pointLocal[1] + 0.5
        idx_x = Math.floor(point_unified_x * self.width)
        idx_y = Math.floor(point_unified_y * self.height)
        self.linePosX = (idx_x + 0.5) - self.width / 2
        self.linePosY = ((-(idx_y + 0.5)) + self.height / 2) * self.height_scale

    @staticmethod 
    def get_value_pos_and_color_gray(tensor: np.ndarray, height_scale: float = 1.0, transposed: bool = False):
        shape = _get_matrix_shape_from_tensor_shape(tensor.shape)
        mat = tensor.reshape(shape)
        if mat.dtype.kind != "f":
            mat = mat.astype(np.float32)
        mat_inf_nan_mask = np.isinf(mat) | np.isnan(mat)
        mat_no_inf_nan_mask = ~(mat_inf_nan_mask.reshape(-1))
        mat_no_inf_nan = mat.reshape(-1)[mat_no_inf_nan_mask]
        width = shape[1]
        height = shape[0]
        mat_flat_inds = np.arange(mat.size, dtype=np.int32)
        fill_pos_x = (mat_flat_inds % width) + 0.5 - width / 2
        fill_pos_y = (np.floor(mat_flat_inds / width) + 0.5 - height / 2)
        if transposed:
            fill_pos_x, fill_pos_y = fill_pos_y, fill_pos_x
        fill_pos = np.stack([fill_pos_x, -fill_pos_y * height_scale], axis=-1)
        if mat_no_inf_nan.size == 0:
            return None, None, fill_pos
        mat_max = np.max(mat_no_inf_nan)
        mat_min = np.min(mat_no_inf_nan)

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
        if mat_no_inf_nan.size != mat_flat.size:
            # has mask 
            mask_pos = fill_pos[~mat_no_inf_nan_mask]
        else:
            mask_pos = None
        return fill_pos[mat_no_inf_nan_mask], color_res[mat_no_inf_nan_mask], mask_pos

    def get_global_fill(self, global_key: str, inds: np.ndarray, is_persist: bool = True, color_advance: Optional[np.ndarray] = None):
        inds_flat = inds.reshape(-1).astype(np.float32)
        fill_pos_x = (inds_flat % self.width) + 0.5 - self.width / 2
        fill_pos_y = (np.floor(inds_flat / self.width) + 0.5 - self.height / 2)
        if self.transposed:
            fill_pos_x, fill_pos_y = fill_pos_y, fill_pos_x
        fill_pos = np.stack([fill_pos_x, -fill_pos_y * self.height_scale], axis=-1)
        fill_color = np.empty([inds_flat.shape[0], 3], np.float32)
        if is_persist:
            color = ColorUtil.getPerfettoColorRGB(global_key)
        else:
            color = ColorUtil.getPerfettoVariantColorRGB(global_key)
        if color_advance is None:
            color_advance_val = 0
        else:
            color_advance_val = color_advance
        fill_color[:, 0] = (color[0] / 255 + color_advance_val) % 1.0
        fill_color[:, 1] = (color[1] / 255 + color_advance_val) % 1.0
        fill_color[:, 2] = (color[2] / 255 + color_advance_val) % 1.0
        return fill_pos,  fill_color

@dataclasses.dataclass
class GlobalMemoryModel:
    matrices: dict[str, Matrix]
    minimap: plus.hud.MinimapModel
    autolayout_width: float = -1
    @staticmethod 
    def empty():
        return GlobalMemoryModel(
            matrices={},
            minimap=plus.hud.MinimapModel(1, 1, fit_mode=int(plus.hud.MinimapFitMode.WIDTH))
        )

    @pfl.js.mark_js_compilable
    def _do_autolayout(self, width: float):
        if self.autolayout_width > 0:
            whs: list[tuple[float, float]] = []
            for k, m in self.matrices.items():
                whs.append((m.widthLayout, m.heightLayout))
            layout_res = pfl.js.MathUtil.binpack(whs, pfl.js.Math.min(self.autolayout_width, width))
            cnt = 0
            for k, m in self.matrices.items():
                new_x, new_y = layout_res.result[cnt]
                m.offsetX = new_x + m.widthLayout / 2 - layout_res.width / 2
                m.offsetY = new_y + m.heightLayout / 2 - layout_res.height / 2
                cnt += 1
            self.minimap.width = layout_res.width
            self.minimap.height = layout_res.height
            self.minimap._do_layout()

    @pfl.js.mark_js_compilable
    def _handle_layout_event(self, ev: three.HudLayoutChangeEvent):
        self._do_autolayout(ev.innerSizeX)

    @pfl.js.mark_js_compilable
    def _do_layout_event_on_model_change(self):
        self._do_autolayout(self.minimap.layout.innerSizeX)

class GlobalMemLayers(enum.IntEnum):
    BKGD = -8
    PERSIST_FILL = -7
    TEMP_FILL = -6
    PERSIST_LINE = -5
    TEMP_LINE = -4
    TEMP_MASK = -3

    TEXT = -2
    INDICATOR = -1

class MatrixPanel(three.Group):
    def __init__(self, draft: Matrix, enable_hover_line: bool = False, label_with_shape: bool = True,
            selected_color: str = "red", hovered_color: str = "blue"):
        assert isinstance(draft, DraftBase)
        trs_empty = np.zeros((0, 2), dtype=np.float32)
        lines_empty = np.zeros((0, 2), dtype=np.float32)

        self.event_plane = three.Mesh([
            three.PlaneGeometry(1, 1),
            three.MeshBasicMaterial().prop(transparent=True, opacity=0.0),
        ]).prop(position=(0, 0, int(GlobalMemLayers.BKGD)))

        self.event_plane.bind_fields_unchecked_dict({
            "scale-x": draft.widthVis,
            "scale-y": draft.heightVis,
        })
        self._hover_line = three.LineShape(three.Shape.from_aabb(0, 0, 1, 1))
        self._hover_line.prop(color="blue", lineWidth=1, position=(0, 0, int(GlobalMemLayers.INDICATOR)))
        self._hover_line_cond = mui.MatchCase.binary_selection(True, self._hover_line)
        self._hover_line.bind_fields_unchecked_dict({
            "position-x": draft.linePosX,
            "position-y": draft.linePosY,
        })
        dm = mui.DataModel.get_datamodel_from_draft(draft)
        self._hover_line_cond.bind_fields(condition=f"{draft.linePosX} is not None")

        if enable_hover_line:
            self.event_plane.event_leave.add_frontend_draft_set_none(draft, "linePosX")
            self.event_plane.event_move.add_frontend_handler(dm, Matrix._on_hover_pfl, targetPath=str(draft))

        self._border = three.LineShape(three.Shape.from_aabb(0, 0, 1, 1))
        self._border.prop(lineWidth=1)
        self._border.bind_fields_unchecked_dict({
            "scale-x": draft.widthVis,
            "scale-y": draft.heightVis,
        }).prop(position=(0, 0, int(GlobalMemLayers.BKGD)))
        self._border.bind_fields(color=f"where({draft.selected}, '{selected_color}', where({draft.hovered}, '{hovered_color}', 'black'))")
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
                three.ShaderUniform("opacity1", three.ShaderUniformType.Number, 0.9),
                three.ShaderUniform("opacity2", three.ShaderUniformType.Number, 0.6),
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
        if label_with_shape:
            self._label.bind_fields(value=f"cformat('%s|%s', {draft.name}, to_string({draft.shape}))")
        else:
            self._label.bind_fields(value=draft.name)
        self._temp_mask.bind_fields(transforms=draft.temp_mask_pos)

        super().__init__([
            self._persist_fill,
            self._temp_fill,
            self._persist_lines,
            self._temp_lines,
            self._border,
            self.event_plane,
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
                k: MatrixPanel(draft.matrices[k]) for k, v in matrices.items()
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
        dm.install_model_update_callback("_gmem_do_auto_layout", GlobalMemoryModel._do_layout_event_on_model_change,   
            submodel_draft=draft)
        minimap.viewport_group.event_hud_layout_change.add_frontend_handler(
            dm, 
            GlobalMemoryModel._handle_layout_event,
            use_immer=True,
            targetPath=str(draft))
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

    def _get_global_matrix(self, matrices: dict[str, np.ndarray], layout: Optional[list[list[Optional[str]]]] = None):
        matrices = matrices.copy()
        padding = 1
        cur_offset_y = 0
        max_width = 1
        matrixe_objs: dict[str, Matrix] = {}
        if layout is not None:
            # do user layout first. if remain, do regular layout.
            max_row_cnt = 0
            for row in layout:
                max_row_cnt = max(max_row_cnt, len(row))

            # first pass: determine width and height of each cell.
            layout_wh = np.zeros((len(layout), max_row_cnt, 2), dtype=np.int32)
            for i, row in enumerate(layout):
                assert row, "row must not empty"
                for j, key in enumerate(row):
                    if key is None:
                        continue
                    transposed = False 
                    if key.endswith(".T"):
                        key = key[:-2]
                        transposed = True
                    arr = matrices[key]
                    shape = _get_matrix_shape_from_tensor_shape(arr.shape)
                    vis_wh = MatrixBase.get_vis_wh_static(shape[1], shape[0], padding=padding, transposed=transposed)
                    layout_wh[i, j] = (vis_wh[0], vis_wh[1])
            # determine width of all columns and height of all rows.
            widths = np.max(layout_wh[..., 0], axis=0)
            heights = np.max(layout_wh[..., 1], axis=1)
            widths_cumsum = np.cumsum(np.concatenate([[0], widths]))[:-1]
            heights_cumsum = np.cumsum(np.concatenate([[0], heights]))[:-1]
            for i, row in enumerate(layout):
                for j, key in enumerate(row):
                    if key is None:
                        continue
                    transposed = False 
                    if key.endswith(".T"):
                        key = key[:-2]
                        transposed = True
                    arr = matrices[key]
                    gmat = Matrix.from_array(key, arr, transposed=transposed)
                    if transposed:
                        gmat.name += ".T"
                    layout_w, layout_h = gmat.get_vis_wh(padding)
                    # gmat.offsetX = layout_w / 2 + widths_cumsum[j]
                    # gmat.offsetY = layout_h / 2 + heights_cumsum[i]
                    gmat.offsetX = widths[j] / 2 + widths_cumsum[j]
                    gmat.offsetY = heights[i] / 2 + heights_cumsum[i]
                    matrices.pop(key)
                    matrixe_objs[key] = gmat
            cur_offset_y = int(heights_cumsum[-1] + heights[-1])
            max_width = int(widths_cumsum[-1] + widths[-1])
        for k, v in matrices.items():
            gmat = Matrix.from_array(k, v)
            layout_w, layout_h = gmat.get_vis_wh(padding)
            gmat.offsetX = layout_w / 2
            gmat.offsetY = cur_offset_y + layout_h / 2
            cur_offset_y += layout_h
            matrixe_objs[k] = gmat
            max_width = max(max_width, layout_w)
        for k, v in matrixe_objs.items():
            v.offsetX -= max_width / 2
            v.offsetY -= cur_offset_y / 2

        return matrixe_objs, max_width, cur_offset_y


    async def set_matrix_dict(self, matrices: dict[str, np.ndarray], layout: Optional[list[list[Optional[str]]]] = None):
        matrixe_panels: dict[str, MatrixPanel] = {}
        gmatrices, max_width, max_height = self._get_global_matrix(matrices, layout)
        await self.minimap.set_new_childs([])

        async with self._dm.draft_update():
            self._draft.matrices = {}
            for k, v in gmatrices.items():
                self._draft.matrices[k] = v
                matrixe_panels[k] = MatrixPanel(self._draft.matrices[k])
            self._draft.minimap.width = max_width
            self._draft.minimap.height = max_height
            if layout is None:
                self._draft.autolayout_width = 512
            else:
                self._draft.autolayout_width = -1
        await self.minimap.set_new_childs(matrixe_panels)


    def _create_empty_vis_model(self) -> GlobalMemoryModel:
        return GlobalMemoryModel(
            {}, 
            plus.hud.MinimapModel(1, 1))


def layout_table_inplace(table: list[list[Optional[Union[Label, Matrix]]]], elem_padding: int = 2):
    layout_wh = np.zeros((len(table), len(table[0]), 2), dtype=np.int32)
    for i, row in enumerate(table):
        for j, cell in enumerate(row):
            if cell is None:
                continue 
            if isinstance(cell, Label):
                # string cell, use font height to determine size.
                layout_wh[i, j] = (len(cell.text) * cell.fontSize * FONTSIZE_WIDTH_APPROX + elem_padding * 2, cell.fontSize + elem_padding * 2)
            elif isinstance(cell, Matrix):
                # matrix cell, use matrix size.
                vis_wh = cell.get_vis_wh(padding=elem_padding)
                layout_wh[i, j] = (vis_wh[0], vis_wh[1])
            else:
                raise TypeError(f"Unsupported cell type: {type(cell)}")
    # determine width of all columns and height of all rows.
    widths = np.max(layout_wh[..., 0], axis=0)
    heights = np.max(layout_wh[..., 1], axis=1)
    widths_cumsum = np.cumsum(np.concatenate([[0], widths]))[:-1]
    heights_cumsum = np.cumsum(np.concatenate([[0], heights]))[:-1]
    for i, row in enumerate(table):
        for j, cell in enumerate(row):
            if cell is None:
                continue 
            cell.offsetX = widths[j] / 2 + widths_cumsum[j]
            cell.offsetY = heights[i] / 2 + heights_cumsum[i]

    cur_offset_y = int(heights_cumsum[-1] + heights[-1])
    max_width = int(widths_cumsum[-1] + widths[-1])
    return max_width, cur_offset_y