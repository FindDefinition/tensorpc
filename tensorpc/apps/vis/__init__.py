from typing import List
from . import figure, objects
from tensorpc import simple_chunk_call
from .vis_util import vis_in_relay, vis_pc_in_relay, Vis, get_img_fig, get_pc_fig


def vis_figures(addr: str, figs: List[figure.Figure]):
    assert figs, "at least one fig is needed"
    datas = []
    for d in figs:
        assert isinstance(d, figure.Figure)
        datas.append(d.data())
    simple_chunk_call(addr,
                      "tensorpc.services.vis::VisService.send_vis_message",
                      datas)
