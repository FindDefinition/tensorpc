# Copyright 2022 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np 
from . import figure
from tensorpc import simple_chunk_call
from typing import List

DEFAULT_VIS_IP = "127.0.0.1:51051"
def color_step_np(perc, min_value, max_value, color_list):
    perc = (perc - min_value) / (max_value - min_value)
    selections = np.floor(perc * len(color_list)).astype(np.int32)
    selections = np.maximum(selections, 0)
    selections = np.minimum(selections, len(color_list) - 1)
    rgb = color_list[selections]
    return rgb

int16 = lambda x: int(x, 16)

patte = [
    '#00429d', '#0d469f', '#1649a1', '#1d4da2', '#2351a4', '#2955a6',
    '#2e59a8', '#325da9', '#3660ab', '#3b64ad', '#3f68af', '#426cb0',
    '#4670b2', '#4a74b4', '#4e78b5', '#517cb7', '#5580b9', '#5884ba',
    '#5c88bc', '#5f8cbe', '#6390bf', '#6694c1', '#6a98c2', '#6e9cc4',
    '#71a0c6', '#75a5c7', '#78a9c9', '#7cadca', '#80b1cc', '#84b5cd',
    '#88b9cf', '#8cbdd0', '#90c1d1', '#94c5d3', '#98c9d4', '#9dced6',
    '#a1d2d7', '#a6d6d8', '#abdad9', '#b0dedb', '#b5e2dc', '#bae6dd',
    '#c0eade', '#c6eddf', '#cdf1e0', '#d4f5e0', '#dcf8e1', '#e5fbe1',
    '#f0fee1', '#ffffe0'
]
patte = [(int16('0x' + d[1:3]), int16('0x' + d[3:5]), int16('0x' + d[5:7]))
         for d in patte]
patte = np.array(patte).astype(np.uint8)

def get_pointcloud_color(pc, min_offset=0, max_offset=0):
    ranges = np.linspace(0, 1, len(patte), endpoint=False)
    # colors = color_step(pc_origin[:, 3], 0, 1, patte, ranges)
    colors = color_step_np(pc[:, 2], pc[:, 2].min() + min_offset, pc[:, 2].max() + max_offset, patte)
    colors = colors.astype(np.float32) / 255
    return colors

def intensity_to_color(inten, rate=255):
    return np.tile((inten / rate).reshape(-1, 1), [1, 3])

def get_pc_fig(pc, uid=0, colors=None, factor=100, **props) -> figure.PointCloudFigure:
    if pc is not None:
        if colors is None:
            if pc.shape[1] >= 4:
                ind = intensity_to_color(pc[:, 3])
                colors = ind
            else:
                colors = np.zeros((pc.shape[0], 3), np.uint8)
                
        pc = pc[:, :3]
    fig = figure.PointCloudFigure(uid, pc, color=colors, factor=factor)
    fig.data()["type"] = "pointcloud"
    fig.data()["props"] = props
    return fig

def get_img_fig(img, uid=0, encode_suffix="jpg", **props):
    fig = figure.ImageFigure(uid, img)
    fig.data()["props"] = props
    return fig

def vis_in_relay(figs):
    if isinstance(figs, figure.Figure):
        figs = [figs]
    assert isinstance(figs, list)
    datas = []
    for d in figs:
        assert isinstance(d, figure.Figure)
        datas.append(d.data())
    simple_chunk_call(DEFAULT_VIS_IP, "tensorpc.services.vis::VisService.send_vis_message", datas)


def vis_pc_in_relay(pc, uid=0, factor=100, **props):
    simple_chunk_call(DEFAULT_VIS_IP, "tensorpc.services.vis::VisService.send_vis_message",
                      [get_pc_fig(pc, uid, factor=factor, **props).data()])


class Vis:
    def __init__(self) -> None:
        self.figs: List[figure.Figure] = []

    def pc(self, pc, uid=0, factor=-1, **props):
        fig = get_pc_fig(pc, uid, factor=factor, **props)
        self.figs.append(fig)
        return fig 

    def img(self, img, uid=0, **props):
        fig = get_img_fig(img, uid=uid, **props)
        self.figs.append(fig)
        return fig 

    def send(self):
        vis_in_relay(self.figs)
        self.figs.clear()
        return self



