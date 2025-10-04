import abc
import enum
from typing import Annotated, Any, Callable, Optional, Union 
from tensorpc.apps.dbg import pmql
from tensorpc.core.datamodel.draft import DraftFieldMeta
from tensorpc.dock import mui, three, plus, chart
import contextlib 
import torch 
from tensorpc.core import dataclass_dispatch as dataclasses
import numpy as np 
from tensorpc.core import pfl
from tensorpc.core.pfl.backends.js import Math, MathUtil
import cmap
import dataclasses as dataclasses_plain

from tensorpc.dock.components.plus.shaders.mask2d import get_mask2d_shader_material


@dataclasses_plain.dataclass
class VideoAttnAnalysisResult:
    num_steps: int 
    num_layers: int
    num_heads: int
    sm_scale: float
    inputs_per_step: list[tuple[Any, dict[str, Any]]]
    # qk must be BSHD
    get_qk_fn: Callable[..., list[tuple[torch.Tensor, torch.Tensor, Any]]]
    cur_qk_list: Optional[list[tuple[torch.Tensor, torch.Tensor, Any]]] = None
    infer_score_fn: Optional[Callable[[torch.Tensor, torch.Tensor, Any, int], torch.Tensor]] = None
    qk_share_key: Optional[str] = None
    score_threshold: float = 0.001
    show_score_text: bool = True

def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b

class Layers(enum.IntEnum):
    IMAGE = -8
    VISUAL_MASK = -7

    ATTN_BOX = -6
    ATTN_SCORE_TEXT = -5

    MASK = -4
    TEXT = -2

_ATTN_IMG_PADDING = 10

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class FrameAttnModel:
    # video generation use one token for multiple frame, so we still
    # need frame idx and video.
    id: str
    offsetX: float 
    offsetY: float
    token_frame_idx: int
    video_jpegs: list[bytes]
    video_shape: list[int] # nhw
    sub_frame_idx: int
    frame_desc: str
    frame_text_color: str

    # selected attn score map
    selected_score_pos: Optional[np.ndarray] = None
    selected_score_color: Optional[np.ndarray] = None
    selected_score_texts: Optional[list[str]] = None
    mask_pos: Optional[np.ndarray] = None
    visual_mask_pos: Optional[np.ndarray] = None

    fontSize: float = 1.0
    textOffsetX: float = 0.0
    textOffsetY: float = 0.0

MAX_MATRIX_SIZE = 10000 # enough for 720P 16x16 downsampled video.

class AttnFramePanel(three.Group):
    def __init__(self, draft: FrameAttnModel, downsample_stride_hw: tuple[int, int]):
        image = three.Image().prop(cached=False)
        image.bind_fields(image=f"getItem({draft.video_jpegs}, {draft.sub_frame_idx})", scale=f"{draft.video_shape[1]}")
        text = three.Text("").prop(position=(0, 0, int(Layers.TEXT)), color="green", 
            fillOpacity=0.9, anchorX="left", anchorY="top")
        text.bind_fields(value=f"cformat('%d|%s', {draft.token_frame_idx}, {draft.frame_desc})",
                fontSize=f"{draft.fontSize}", color=draft.frame_text_color)
        text.bind_fields_unchecked_dict({
            "position-x": f"{draft.textOffsetX}",
            "position-y": f"{draft.textOffsetY}",
        })
        # text.bind_fields(value=f"printForward($)")
        trs_empty = np.zeros((0, 2), dtype=np.float32)
        lines_empty = np.zeros((0, 2), dtype=np.float32)

        temp_mask = three.InstancedMesh(trs_empty, MAX_MATRIX_SIZE, [
            three.PlaneGeometry(downsample_stride_hw[0], downsample_stride_hw[1]),
            get_mask2d_shader_material(20.0).prop(transparent=True),
        ]).prop(position=(0, 0, int(Layers.MASK)))
        temp_mask.bind_fields(transforms=draft.mask_pos)

        # visual_mask = three.InstancedMesh(trs_empty, MAX_MATRIX_SIZE, [
        #     three.PlaneGeometry(downsample_stride_hw[0], downsample_stride_hw[1]),
        #     get_mask2d_shader_material(20.0, coeff=-1.0).prop(transparent=True),
        # ]).prop(position=(0, 0, int(Layers.VISUAL_MASK)))
        # visual_mask.bind_fields(transforms=draft.visual_mask_pos)

        lines = three.Line(lines_empty).prop( 
            color="green", lineWidth=1, opacity=0.7, segments=True, variant="aabb",
            aabbSizes=(downsample_stride_hw[0], downsample_stride_hw[1], 1),)
        lines.bind_fields(points=draft.selected_score_pos, 
            vertexColors=draft.selected_score_color
        )
        text_scores = three.Text("").prop(position=(0, 0, int(Layers.ATTN_SCORE_TEXT)), color="blue", 
            fillOpacity=0.7, fontSize=downsample_stride_hw[0] * 0.50)
        text_scores.bind_fields(value=draft.selected_score_texts, 
            positions=draft.selected_score_pos,
            colors=draft.selected_score_color,)
        super().__init__([
            text.prop(position=(0, 0, int(Layers.TEXT))),
            image.prop(position=(0, 0, int(Layers.IMAGE))),
            lines.prop(position=(0, 0, int(Layers.ATTN_BOX))),
            text_scores,
            temp_mask,
            # visual_mask,
        ])
        self.bind_fields_unchecked_dict({
            "position-x": draft.offsetX,
            "position-y": f"-{draft.offsetY}",
        })

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class VideoAttnAnalysisState:
    num_steps: int 
    num_layers: int
    num_heads: int
    cur_step: int
    cur_head: int 
    cur_layer: int 
    score_img: np.ndarray
    video_jpegs: list[bytes]
    video_shape: list[int] # nhw
    selected_token_pixel_idx: int
    token_shape: list[int] # nhw
    cur_frame_idx: int
    cur_token_frame_idx: int

    downsample_stride: list[int] 
    preview_minimap: plus.hud.MinimapModel
    attnview_minimap: plus.hud.MinimapModel
    attn_frames: list[FrameAttnModel]
    hoverlinePosX: Optional[float] = None
    hoverlinePosY: Optional[float] = None
    sellinePosX: Optional[float] = None
    sellinePosY: Optional[float] = None
    autolayout_width: int = -1
    frameScoreSeries: list[chart.BarSeries] = dataclasses.field(default_factory=list)
    frameScoreXAxis: list[chart.XAxis] = dataclasses.field(default_factory=list)
    
    cur_analysis_idx: int = 0
    num_analysis: int = 0

    @staticmethod 
    def empty():
        return VideoAttnAnalysisState(
            num_steps=0, num_layers=0, num_heads=0,
            # inputs_per_step=[],
            cur_step=0, cur_head=0, cur_layer=0,
            score_img=np.zeros((0, 1, 3), dtype=np.uint8),
            video_jpegs=[],
            video_shape=[0, 1, 1],
            selected_token_pixel_idx=-1,
            token_shape=[1, 1, 1],
            cur_frame_idx=0,
            cur_token_frame_idx=0,
            downsample_stride=[1, 1, 1],
            preview_minimap=plus.hud.MinimapModel(),
            attnview_minimap=plus.hud.MinimapModel(fit_mode=int(plus.hud.MinimapFitMode.WIDTH)),
            attn_frames=[],
        )


    @pfl.mark_pfl_compilable
    def _img_move_pfl(self, data: three.PointerEvent):
        stride_h = self.downsample_stride[1]
        stride_w = self.downsample_stride[2]
        video_h_downsampled = self.video_shape[2] // stride_h
        video_w_downsampled = self.video_shape[1] // stride_w
        pixel_x = Math.floor((data.pointLocal[0] + 0.5) * video_h_downsampled)
        pixel_y = Math.floor((-data.pointLocal[1] + 0.5) * video_w_downsampled)
        x = (pixel_x + 0.5) - video_h_downsampled / 2
        y = (-(pixel_y + 0.5)) + video_w_downsampled / 2
        self.hoverlinePosX = x * stride_w
        self.hoverlinePosY = y * stride_h
        # print("hoverlinePosX", self.hoverlinePosX, "hoverlinePosY", self.hoverlinePosY)

    @pfl.mark_pfl_compilable
    def _img_move_clear_pfl(self, data: three.PointerEvent):
        self.hoverlinePosX = None
        self.hoverlinePosY = None

    @pfl.js.mark_js_compilable
    def _on_token_click_pfl(self, data: three.PointerEvent):
        stride_h = self.downsample_stride[1]
        stride_w = self.downsample_stride[2]
        video_h_downsampled = self.video_shape[2] // stride_h
        video_w_downsampled = self.video_shape[1] // stride_w
        pixel_x = Math.floor((data.pointLocal[0] + 0.5) * video_h_downsampled)
        pixel_y = Math.floor((-data.pointLocal[1] + 0.5) * video_w_downsampled)
        x = (pixel_x + 0.5) - video_h_downsampled / 2
        y = (-(pixel_y + 0.5)) + video_w_downsampled / 2
        self.sellinePosX = x * stride_w
        self.sellinePosY = y * stride_h

    @pfl.js.mark_js_compilable
    def _do_autolayout(self, width: float):
        padding = _ATTN_IMG_PADDING
        if self.autolayout_width > 0:
            whs: list[tuple[float, float]] = []
            for f in self.attn_frames:
                whs.append((f.video_shape[2] + padding, f.video_shape[1] + padding))
            # print("WHS", whs)
            # layout_res = pfl.js.MathUtil.binpack(whs, pfl.js.Math.min(self.autolayout_width, width))
            layout_res = pfl.js.MathUtil.binpack(whs, self.autolayout_width)

            cnt = 0
            for f in self.attn_frames:
                new_x, new_y = layout_res.result[cnt]
                f.offsetX = new_x + (f.video_shape[2] + padding) / 2 - layout_res.width / 2
                f.offsetY = new_y + (f.video_shape[1] + padding) / 2 - layout_res.height / 2
                cnt += 1
            self.attnview_minimap.width = layout_res.width
            self.attnview_minimap.height = layout_res.height
            # print("LAYOUT RES", layout_res)
            self.attnview_minimap._do_layout()

    @pfl.js.mark_js_compilable
    def _handle_layout_event(self, ev: three.HudLayoutChangeEvent):
        self._do_autolayout(ev.innerSizeX)

    @pfl.js.mark_js_compilable
    def _do_layout_event_on_model_change(self):
        self._do_autolayout(self.attnview_minimap.layout.innerSizeX)

# model_fwd_query: str, self_attn_q_query: str, self_attn_k_query: str

class VideoAttentionViewer(mui.FlexBox):
    def __init__(self, downsample_stride_hw: tuple[int, int]):
        super().__init__()
        self.dm = mui.DataModel(VideoAttnAnalysisState.empty(), [])

        draft = self.dm.get_draft()
        image = three.Image().prop(cached=False)
        hover_line = three.Group([
            three.Line([(0.0, 0.0, 0.0), ]).prop(color="blue", lineWidth=1, variant="aabb", aabbSizes=(1, 1, 1))
        ]).prop(position=(0, 0, 0.2))
        hover_line_cond = mui.MatchCase.binary_selection(True, hover_line)
        select_line = three.Group([
            three.Line([(0.0, 0.0, 0.0), ]).prop(color="red", lineWidth=2, variant="aabb", aabbSizes=(1, 1, 1))
        ]).prop(position=(0, 0, 0.1))
        select_line_cond = mui.MatchCase.binary_selection(True, select_line)
        hover_line_cond.bind_fields(condition="hoverlinePosX is not None")
        select_line_cond.bind_fields(condition="sellinePosX is not None")
        hover_line.bind_fields(scale=draft.downsample_stride[2])
        select_line.bind_fields(scale=draft.downsample_stride[2])
        image.event_move.add_frontend_handler(self.dm, VideoAttnAnalysisState._img_move_pfl)
        image.event_leave.add_frontend_handler(self.dm, VideoAttnAnalysisState._img_move_clear_pfl)
        image.event_click.add_frontend_handler(self.dm, VideoAttnAnalysisState._on_token_click_pfl)
        image.event_click.configure(stop_propagation=True)
        image.event_click.on_standard(self._on_token_click)
        self._analysis_list: list[VideoAttnAnalysisResult] = []
        hover_line.bind_fields_unchecked_dict({
            "position-x": "hoverlinePosX",
            "position-y": "hoverlinePosY",
        })
        select_line.bind_fields_unchecked_dict({
            "position-x": "sellinePosX",
            "position-y": "sellinePosY",
        })

        img_group = three.Group([
            image,
            hover_line_cond,
            select_line_cond,
        ]).prop(position=(0, 0, 0))
        image.bind_fields(image=f"getItem({draft.video_jpegs}, {draft.cur_frame_idx})", scale=draft.preview_minimap.height)
        # image.bind_fields(image=f"console.log(draft.video)", scale=draft.preview_minimap.height)

        preview_minimap = plus.hud.MiniMap(draft.preview_minimap, [
            img_group
        ])
        self._cm = cmap.Colormap("cool")
        cam = three.OrthographicCamera(near=0.1, far=1000, children=[
            preview_minimap,
        ]).prop(position=(0, 0, 10))
        preview_view = three.View([
            cam.prop(makeDefault=True),
        ]).prop(allowKeyboardEvent=True)
        preview_minimap.install_canvas_events(draft.preview_minimap, preview_view)

        frame_slider = mui.Slider(0, 0, 1, label="frame").bind_draft_change(draft.cur_frame_idx)
        frame_slider.bind_fields(max=f"{draft.video_shape}[0] - 1")
        frame_slider.event_change.on(self._on_frame_slider_change)
        frame_slider.prop(valueInput=True)

        step_slider = mui.Slider(0, 0, 1, label="step").bind_draft_change(draft.cur_step)
        step_slider.bind_fields(max=f"{draft.num_steps} - 1")
        step_slider.prop(valueInput=True)
        step_slider.event_change.on(self._on_step_slider_change)

        layer_slider = mui.Slider(0, 0, 1, label="layer").bind_draft_change(draft.cur_layer)
        layer_slider.bind_fields(max=f"{draft.num_layers} - 1")
        layer_slider.prop(valueInput=True)
        layer_slider.event_change.on(self._on_layer_slider_change)

        head_slider = mui.Slider(0, 0, 1, label="head").bind_draft_change(draft.cur_head)
        head_slider.bind_fields(max=f"{draft.num_heads} - 1")
        head_slider.prop(valueInput=True)
        head_slider.event_change.on(self._on_head_slider_change)

        analysis_slider = mui.Slider(0, 0, 1, label="Attn Item").bind_draft_change(draft.cur_analysis_idx)
        analysis_slider.bind_fields(max=f"{draft.num_analysis} - 1")
        analysis_slider.prop(valueInput=True)
        analysis_slider.event_change.on(self._on_attn_item_change)


        draft_nested = self.dm.create_external_draft_with_self(FrameAttnModel)
        attnframe = AttnFramePanel(draft_nested, downsample_stride_hw)
        attnframe_minimap = plus.hud.MiniMap(draft.attnview_minimap, [
            three.DataListGroup(attnframe).bind_fields(dataList=draft.attn_frames),
        ], minimap_event_key="attn_frames")
        cam_attn_view = three.OrthographicCamera(near=0.1, far=1000, children=[
            attnframe_minimap,
        ]).prop(position=(0, 0, 10))
        attn_view = three.View([
            cam_attn_view.prop(makeDefault=True),
        ]).prop(allowKeyboardEvent=True)
        attnframe_minimap.install_canvas_events(draft.attnview_minimap, attn_view)
        self.dm.install_model_update_callback("_gmem_do_auto_layout", VideoAttnAnalysisState._do_layout_event_on_model_change,   
            submodel_draft=draft)
        attnframe_minimap.viewport_group.event_hud_layout_change.add_frontend_handler(
            self.dm, 
            VideoAttnAnalysisState._handle_layout_event,
            use_immer=True,
            targetPath=str(draft))
        frame_chart = chart.BarChart().bind_fields(series=draft.frameScoreSeries, xAxis=draft.frameScoreXAxis)
        frame_chart.event_axis_click.on(self._on_frame_chart_click)
        self.dm.init_add_layout([
            mui.VBox([
                mui.HBox([
                    mui.Typography().prop(variant="body1", flex=1).bind_fields(value=f"cformat('Token Frame Idx: %d', {draft.cur_token_frame_idx})"),
                    analysis_slider.prop(flex=1),
                ]),
                preview_view.prop(flex=1),
                frame_slider,
                step_slider,
                layer_slider,
                head_slider,
                mui.HBox([
                    frame_chart
                ]).prop(height="200px")
            ]).prop(flex=1),
            mui.VDivider(),
            attn_view.prop(flex=1),
        ])
        super().__init__([
            three.ViewCanvas([
                self.dm
            ]).prop(display="flex", flexFlow="row nowrap", flex=1, overflow="hidden")
        ])
        self.prop(width="100%", height="100%", overflow="hidden", minWidth=0, minHeight=0)

    async def _on_frame_slider_change(self, value):
        # sync token frame idx
        async with self.dm.draft_update() as draft:
            downsample_stride = self.dm.model.downsample_stride
            if value == 0:
                token_frame_idx = 0
            else:
                token_frame_idx = (value - 1) // downsample_stride[0] + 1
            draft.cur_token_frame_idx = token_frame_idx
        if self.dm.model.selected_token_pixel_idx >= 0:
            pidx = self.dm.model.selected_token_pixel_idx
            fidx = self.dm.model.cur_token_frame_idx
            selected_token_idx = pidx + self.dm.model.token_shape[1] * self.dm.model.token_shape[2] * fidx
            await self._set_selected_token_scores(selected_token_idx, self.dm.model.cur_token_frame_idx)

    def _get_attn_frame_scores_from_score(self, score: np.ndarray, width: int, height: int, 
            downsample_stride: list[int], threshold: float = 0.01,
            norm_score: bool = True):
        selected_mask = score > threshold
        ys, xs = np.where(selected_mask)
        ys_mask, xs_mask = np.where(~selected_mask)
        # print(ys, xs)
        selected_score_pos = np.stack([
            (xs + 0.5) * downsample_stride[2] - width / 2,
            -(ys + 0.5) * downsample_stride[1] + height / 2,
        ], axis=1).astype(np.float32)
        mask_score_pos = np.stack([
            (xs_mask + 0.5) * downsample_stride[2] - width / 2,
            -(ys_mask + 0.5) * downsample_stride[1] + height / 2,
        ], axis=1).astype(np.float32)
        if norm_score:
            selected_score = score[selected_mask]
            selected_score_max = selected_score.max() if selected_score.size > 0 else 0.0
            normed_score = selected_score / selected_score_max if selected_score_max > 0 else selected_score
        else:
            normed_score = score[selected_mask]
        gray_colors = cmap.Colormap("cool")(normed_score.reshape(-1)).astype(np.float32)[:, :3]
        
        gray_colors = np.ascontiguousarray(gray_colors)
        text_scores = [str(min(999, int(s * 1000))) for s in score[selected_mask].reshape(-1)]
        return selected_score_pos, mask_score_pos, gray_colors, text_scores
    
    async def _on_token_click(self, ev: mui.Event):
        data = ev.data
        assert isinstance(data, three.PointerEvent)
        video_h_downsampled = self.dm.model.token_shape[2]
        video_w_downsampled = self.dm.model.token_shape[1] 
        pixel_x = int(Math.floor((data.pointLocal[0] + 0.5) * video_h_downsampled))
        pixel_y = int(Math.floor((-data.pointLocal[1] + 0.5) * video_w_downsampled))
        pixel_idx = pixel_y * video_h_downsampled + pixel_x
        cur_token_frame_idx = self.dm.model.cur_token_frame_idx
        token_idx = cur_token_frame_idx * video_h_downsampled * video_w_downsampled + pixel_idx
        # print(token_idx, pixel_x, pixel_y, cur_token_frame_idx)
        async with self.dm.draft_update() as draft:
            draft.selected_token_pixel_idx = pixel_idx
        if self._analysis_list is not None:
            await self._set_selected_token_scores(token_idx, cur_token_frame_idx)

    def _infer_scores(self, token_idx: int, analysis: VideoAttnAnalysisResult, *, cur_step: Optional[int] = None,
            cur_layer: Optional[int] = None, cur_head: Optional[int] = None) -> np.ndarray:
        if cur_step is None:
            cur_step = self.dm.model.cur_step
        if cur_layer is None:
            cur_layer = self.dm.model.cur_layer
        if cur_head is None:
            cur_head = self.dm.model.cur_head
        if analysis.cur_qk_list is None:
            if analysis.qk_share_key is not None:
                # try to find from other analysis
                for a in self._analysis_list:
                    if a is not analysis and a.qk_share_key == analysis.qk_share_key and a.cur_qk_list is not None:
                        analysis.cur_qk_list = a.cur_qk_list
                        break
            if analysis.cur_qk_list is None:
                cur_step = self.dm.model.cur_step
                args, kwargs = analysis.inputs_per_step[cur_step]
                qk_list = analysis.get_qk_fn(*args, **kwargs)
                # print(len(qk_list), "qk_list")
                analysis.cur_qk_list = qk_list

        assert analysis.cur_qk_list is not None
        q, k, additional = analysis.cur_qk_list[cur_layer] # qk is BSHD
        # assume B == 1, we can also use B * H as slider index
        q_head = q[0, :, cur_head] # SD
        k_head = k[0, :, cur_head] # SD
        if analysis.infer_score_fn is not None:
            score = analysis.infer_score_fn(q_head, k_head, additional, token_idx)
            return score.view(self.dm.model.token_shape).cpu().numpy()

        q_token = q_head[token_idx:token_idx + 1, :] # 1D
        # print(q.shape, q_head.shape, k_head.shape, token_idx)
        if q_token.device.type == "cpu":
            q_token = q_token.cuda()
            k_head = k_head.cuda()
        score = (q_token.float() @ k_head.float().T) * analysis.sm_scale
        score = torch.softmax(score, dim=-1).reshape(self.dm.model.token_shape)
        if torch.isnan(score).any():
            print("NAN in score, raise.")
            raise ValueError("NAN in score")
        return score.cpu().numpy()

    async def _set_selected_token_scores(self, token_idx: int, token_frame_idx: int,
            *, cur_step: Optional[int] = None,
            cur_layer: Optional[int] = None, cur_head: Optional[int] = None):
        # analysis = self.get_cur_analysis()
        # assert analysis is not None 
        assert self._analysis_list is not None
        cur_analysis = self.get_cur_analysis()
        assert cur_analysis is not None 
        scores: list[np.ndarray] = []
        for analysis in self._analysis_list:
            score = self._infer_scores(token_idx, analysis, cur_step=cur_step,
                cur_layer=cur_layer, cur_head=cur_head)
            scores.append(score)
        await self.set_attn_frame(scores, token_frame_idx, threshold=cur_analysis.score_threshold)

    async def _on_step_slider_change(self, value):
        # old = self.dm.model.cur_step
        # if old == value:
        #     return 
        analysis = self.get_cur_analysis()

        assert analysis is not None  
        args, kwargs = analysis.inputs_per_step[value]
        qk_list = analysis.get_qk_fn(*args, **kwargs)
        analysis.cur_qk_list = qk_list
        if self.dm.model.selected_token_pixel_idx >= 0:
            pidx = self.dm.model.selected_token_pixel_idx
            fidx = self.dm.model.cur_token_frame_idx
            selected_token_idx = pidx + self.dm.model.token_shape[1] * self.dm.model.token_shape[2] * fidx
            await self._set_selected_token_scores(selected_token_idx, self.dm.model.cur_token_frame_idx,
                cur_step=value)

    async def _on_layer_slider_change(self, value):
        # old = self.dm.model.cur_layer
        # if old == value:
        #     return
        if self.dm.model.selected_token_pixel_idx >= 0:
            pidx = self.dm.model.selected_token_pixel_idx
            fidx = self.dm.model.cur_token_frame_idx
            selected_token_idx = pidx + self.dm.model.token_shape[1] * self.dm.model.token_shape[2] * fidx
            await self._set_selected_token_scores(selected_token_idx, self.dm.model.cur_token_frame_idx,
                cur_layer=value)

    async def _on_head_slider_change(self, value):
        # old = self.dm.model.cur_head
        # if old == value:
        #     return
        if self.dm.model.selected_token_pixel_idx >= 0:
            pidx = self.dm.model.selected_token_pixel_idx
            fidx = self.dm.model.cur_token_frame_idx
            selected_token_idx = pidx + self.dm.model.token_shape[1] * self.dm.model.token_shape[2] * fidx
            await self._set_selected_token_scores(selected_token_idx, self.dm.model.cur_token_frame_idx,
                cur_head=value)

    async def set_video(self, video: np.ndarray, analysis: Optional[Union[VideoAttnAnalysisResult, list[VideoAttnAnalysisResult]]] = None):
        downsample_stride = [4, 16, 16]
        assert video.ndim == 4 and video.shape[3] == 3
        # video_flip_y = video[:, ::-1, :, :]
        if isinstance(analysis, VideoAttnAnalysisResult):
            analysis = [analysis]
        video_jpegs = []
        for img in video:
            jpeg_bytes = mui.Image.encode_image_bytes(img)
            video_jpegs.append(jpeg_bytes)
        prev_model = self.dm.model
        async with self.dm.draft_update() as draft:
            draft.preview_minimap.width = video.shape[2]
            draft.preview_minimap.height = video.shape[1] 
            draft.video_jpegs = video_jpegs
            draft.video_shape = list(video.shape)
            draft.downsample_stride = downsample_stride
            draft.token_shape = [
                1 + _div_up(video.shape[0] - 1, downsample_stride[0]),
                video.shape[1] // downsample_stride[1],
                video.shape[2] // downsample_stride[2],
            ]
            draft.sellinePosX = None
            draft.sellinePosY = None
            if prev_model.video_shape[0] != video.shape[0]:
                draft.cur_token_frame_idx = 0
                draft.cur_frame_idx = 0
            num_frame = video.shape[0]
            height = video.shape[1]
            width = video.shape[2]
            num_token_frame_except_first = _div_up(num_frame - 1, downsample_stride[0])
            first_attn_frame = FrameAttnModel(
                "0",
                0, 0, 0, [video_jpegs[0]],
                [1, video.shape[1], video.shape[2]],
                0, "0",
                frame_text_color="green",
            )
            first_attn_frame.fontSize = min(width, height) * 0.1
            first_attn_frame.textOffsetX = -width * 0.5 + 5
            first_attn_frame.textOffsetY = height * 0.5 - 5

            first_attn_frame.selected_score_pos = None
            first_attn_frame.mask_pos = None
            first_attn_frame.selected_score_color = None #  * 255
            first_attn_frame.selected_score_texts = None
            attn_frames = [first_attn_frame]
            fs = downsample_stride[0]
            for j in range(num_token_frame_except_first):
                sub_frames = video_jpegs[(j + 1) * fs:(j + 1) * fs + fs]
                attn_frame = FrameAttnModel(
                    str(j + 1),
                    0, 0, j + 1, sub_frames,
                    [len(sub_frames), video.shape[1], video.shape[2]],
                    0, "0",
                    frame_text_color="green",
                ) 
                attn_frame.fontSize = min(width, height) * 0.1
                attn_frame.textOffsetX = -width * 0.5 + 5
                attn_frame.textOffsetY = height * 0.5 - 5
                attn_frame.selected_score_pos = None
                attn_frame.mask_pos = None
                attn_frame.selected_score_color = None #  * 255
                attn_frame.selected_score_texts = None

                attn_frames.append(attn_frame)

            draft.attn_frames = attn_frames
            if height > width:
                draft.autolayout_width = max(video.shape[2] * 5, 1024)
            else:
                draft.autolayout_width = max(video.shape[2] * 3, 1024)
            if analysis is not None:
                draft.num_analysis = len(analysis)
                draft.selected_token_pixel_idx = -1
                self._analysis_list = analysis

        if analysis is not None:
            await self.switch_analysis(0)

    def get_cur_analysis(self) -> Optional[VideoAttnAnalysisResult]:
        if self._analysis_list is None:
            return None
        return self._analysis_list[self.dm.model.cur_analysis_idx]

    async def switch_analysis(self, idx: int):
        assert self._analysis_list is not None 
        analysis = self._analysis_list[idx]
        prev_model = self.dm.model
        async with self.dm.draft_update() as draft:
            if prev_model.num_steps != analysis.num_steps:
                draft.cur_step = analysis.num_steps - 1
            if prev_model.num_heads != analysis.num_heads:
                draft.cur_head = 0
            if prev_model.num_layers != analysis.num_layers:
                draft.cur_layer = 0
            draft.num_steps = analysis.num_steps
            draft.num_layers = analysis.num_layers
            draft.num_heads = analysis.num_heads
            draft.cur_analysis_idx = idx
        if prev_model.selected_token_pixel_idx >= 0:
            pidx = prev_model.selected_token_pixel_idx
            fidx = prev_model.cur_token_frame_idx
            selected_token_idx = pidx + prev_model.token_shape[1] * prev_model.token_shape[2] * fidx
            await self._set_selected_token_scores(selected_token_idx, prev_model.cur_token_frame_idx)

    async def set_attn_frame(self, token_attn_scores: list[np.ndarray], token_frame_idx: int, threshold: float = 0.01):
        # token_attn_score: [N_all]
        video_shape = self.dm.model.video_shape
        downsample_stride = self.dm.model.downsample_stride
        if token_frame_idx == 0:
            video_frame_start = 0
            video_frame_end = 1
        else:
            fs = downsample_stride[0]
            video_frame_start = 1 + (token_frame_idx - 1) * fs
            video_frame_end = min(video_frame_start + fs, self.dm.model.video_shape[0])
        token_shape = self.dm.model.token_shape
        assert self._analysis_list is not None
        width = video_shape[2]
        height = video_shape[1]
        cur_analysis_idx = self.dm.model.cur_analysis_idx
        cur_analysis = self._analysis_list[cur_analysis_idx]
        token_attn_score = token_attn_scores[cur_analysis_idx].reshape(token_shape)
        async with self.dm.draft_update() as draft:
            # draft.cur_token_frame_idx = token_frame_idx
            # draft.cur_frame_idx = video_frame_start
            num_token_frame = token_shape[0]
            frame_scores = []
            for j in range((num_token_frame)):
                frame_token_score = token_attn_score[j]
                frame_sum_threshold = float(frame_token_score[frame_token_score > threshold].sum())
                frame_sum = float(frame_token_score.sum())

                frame_scores.append(frame_sum)
                selected_score_pos, mask_score_pos, gray_colors, text_scores = self._get_attn_frame_scores_from_score(
                    token_attn_score[j], width, height, downsample_stride, threshold=threshold)
    
                draft.attn_frames[j].selected_score_pos = selected_score_pos
                draft.attn_frames[j].mask_pos = mask_score_pos
                draft.attn_frames[j].selected_score_color = gray_colors #  * 255
                if cur_analysis.show_score_text:
                    draft.attn_frames[j].selected_score_texts = text_scores
                else:
                    draft.attn_frames[j].selected_score_texts = None
                draft.attn_frames[j].frame_desc = f"{frame_sum:.3f}\\{frame_sum_threshold:.3f}"

                frame_sum_score_color = self._cm(frame_sum)
                frame_sum_score_color_css = frame_sum_score_color.hex
                draft.attn_frames[j].frame_text_color = frame_sum_score_color_css
            series: list[chart.BarSeries] = []
            for i in range(len(self._analysis_list)):
                frame_sums = token_attn_scores[i].reshape(token_attn_score.shape[0], -1).sum(-1)
                series.append(chart.BarSeries(data=frame_sums.tolist()))
            draft.frameScoreSeries = series
            draft.frameScoreXAxis = [
                chart.XAxis(data=[str(i) for i in range(len(frame_scores))], label="Frame idx")
            ]

    async def _on_frame_chart_click(self, ev: Optional[chart.ChartsAxisData]):
        if ev is None:
            return 
        token_frame_idx = int(ev.dataIndex)
        async with self.dm.draft_update() as draft:
            draft.cur_token_frame_idx = token_frame_idx
            if token_frame_idx == 0:
                draft.cur_frame_idx = 0
            else:
                fs = self.dm.model.downsample_stride[0]
                draft.cur_frame_idx = 1 + (token_frame_idx - 1) * fs
        if self.dm.model.selected_token_pixel_idx >= 0:
            pidx = self.dm.model.selected_token_pixel_idx
            fidx = self.dm.model.cur_token_frame_idx
            selected_token_idx = pidx + self.dm.model.token_shape[1] * self.dm.model.token_shape[2] * fidx
            await self._set_selected_token_scores(selected_token_idx, self.dm.model.cur_token_frame_idx)

    async def _on_attn_item_change(self, value):
        await self.switch_analysis(value)