import inspect 
from tensorpc.core import inspecttools
from tensorpc.apps.vis import vis_figures, figure

def intensity_to_color(inten, rate=255):
    return np.tile((inten / rate).reshape(-1, 1), [1, 3])

def get_pc_fig(pc, uid=0, colors=None, factor=-1, **props) -> figure.PointCloudFigure:
    if pc is not None:
        if colors is None:
            if pc.shape[1] >= 4:
                ind = intensity_to_color(pc[:, 3])
                colors = ind
        pc = pc[:, :3]
    fig = figure.PointCloudFigure(uid, pc, color=colors, factor=factor)
    fig.data()["type"] = "pointcloud"
    fig.data()["props"] = props
    return fig


if __name__ == "__main__":
    import numpy as np 
    # pc = np.load("/home/yy/test.npy")
    # pc_fig = get_pc_fig(pc)
    # vis_figures("localhost:51051", [pc_fig])
    import pickle 
    with open("/home/yy/test.pkl", "rb") as f:
        figs = pickle.load(f)
        vis_figures("localhost:51051", figs)
