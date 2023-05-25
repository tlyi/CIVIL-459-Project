import os

import numpy as np
try:
    import matplotlib.cm as mplcm
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass

import openpifpaf


#Images in openlane dataset seem to have standard format
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1280

LANE_KEYPOINTS_24 = [
    '1',       # 1 the nearest
]

LANE_SKELETON_24 = [
  (1,1)
]


LANE_SIGMAS_24 = [0.05] * len(LANE_KEYPOINTS_24)  # why dont these scales add up to 1

split, error = divmod(len(LANE_KEYPOINTS_24), 4)
LANE_SCORE_WEIGHTS_24 = [10.0] * split + [3.0] * split + \
    [1.0] * split + [0.1] * split + [0.1] * error  

assert len(LANE_SCORE_WEIGHTS_24) == len(LANE_KEYPOINTS_24)



LANE_CATEGORIES_24 = ['unkown',             # 0
                        'white-dash',         # 1
                        'white-solid',        # 2
                        'double-white-dash',  # 3
                        'double-white-solid', # 4
                        'white-ldash-rsolid', # 5
                        'white-lsolid-rdash', # 6
                        'yellow-dash',        # 7
                        'yellow-solid',       # 8
                        'double-yellow-dash', # 9
                        'double-yellow-solid',# 10
                        'yellow-ldash-rsolid',# 11
                        'yellow-lsolid-rdash',# 12
                        'left-curbside',      # 20
                        'right-curbside'      # 21
                      ]

LANE_POSE_STRAIGHT_24 = np.array([
    [3.0, 4.0, 1.0], # 1

])
'''
LANE_POSE_RIGHT_24 = np.array([
    [0.0, 0.0, -6.0], # 1
    [0.011, 0.0, -5.5], # 2
    [0.044, 0.0, -5.0], # 3
    [0.098, 0.0,-4.5], # 4
    [0.175,0.0, -4.0], # 5
    [0.275,0.0, -3.5], # 6
    [ 0.398, 0.0,-3.0], # 7
    [0.546, 0.0,-2.5], # 8
    [0.718,0.0, -2.0], # 9
    [0.916,0.0, -1.5], # 10
    [1.144,0.0, -1.0], # 11
    [1.4,0.0, -0.5], # 12
    [1.689, 0.0,0.0],  # 13
    [2.013,0.0, 0.5],  # 14
    [2.376,0.0, 1.0],  # 15
    [2.782,0.0, 1.5],  # 16
    [3.239,0.0, 2.0],  # 17
    [3.754,0.0, 2.5],  # 18
    [4.341,0.0, 3.0],  # 19
    [5.019,0.0, 3.5],  # 20
    [5.821,0.0, 4.0],  # 21
    [6.810, 0.0, 4.5],  # 22
    [8.146,0.0, 5.0],  # 23
    [11.5,0.0, 5.5],  # 24
    
])


LANE_POSE_LEFT_24 = np.array([
    [0.0, 0.0, -6.0], # 1
    [-0.011, 0.0, -5.5], # 2
    [-0.044, 0.0, -5.0], # 3
    [-0.098, 0.0,-4.5], # 4
    [-0.175, 0.0, -4.0], # 5
    [-0.275, 0.0, -3.5], # 6
    [-0.398, 0.0,-3.0], # 7
    [-0.546, 0.0,-2.5], # 8
    [-0.718, 0.0, -2.0], # 9
    [-0.916, 0.0, -1.5], # 10
    [-1.144, 0.0, -1.0], # 11
    [-1.4, 0.0, -0.5], # 12
    [-1.689, 0.0,0.0],  # 13
    [-2.013, 0.0, 0.5],  # 14
    [-2.376, 0.0, 1.0],  # 15
    [-2.782, 0.0, 1.5],  # 16
    [-3.239, 0.0, 2.0],  # 17
    [-3.754, 0.0, 2.5],  # 18
    [-4.341, 0.0, 3.0],  # 19
    [-5.019, 0.0, 3.5],  # 20
    [-5.821, 0.0, 4.0],  # 21
    [-6.810, 0.0, 4.5],  # 22
    [-8.146, 0.0, 5.0],  # 23
    [-11.5, 0.0, 5.5],  # 24
    
])
'''
def get_constants():
  return [LANE_KEYPOINTS_24, LANE_SKELETON_24, LANE_SIGMAS_24,
                LANE_POSE_STRAIGHT_24, LANE_CATEGORIES_24, LANE_SCORE_WEIGHTS_24]

def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose, sigmas, skel, kps, scr_weights):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    keypoint_painter = show.KeypointPainter()
    ann = Annotation(keypoints=kps, skeleton=skel, score_weights=scr_weights)
    ann.set(pose, np.array(sigmas) * scale)
    os.makedirs('docs', exist_ok=True)
    draw_ann(ann, filename='docs/skeleton_lane.png', keypoint_painter=keypoint_painter)


def plot3d_red(ax_2D, p3d, skeleton):
    skeleton = [(bone[0] - 1, bone[1] - 1) for bone in skeleton]

    rot_p90_x = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    p3d = p3d @ rot_p90_x

    fig = ax_2D.get_figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_axis_off()
    ax_2D.set_axis_off()

    ax.view_init(azim=-90, elev=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = np.array([p3d[:, 0].max() - p3d[:, 0].min(),
                          p3d[:, 1].max() - p3d[:, 1].min(),
                          p3d[:, 2].max() - p3d[:, 2].min()]).max() / 2.0
    mid_x = (p3d[:, 0].max() + p3d[:, 0].min()) * 0.5
    mid_y = (p3d[:, 1].max() + p3d[:, 1].min()) * 0.5
    mid_z = (p3d[:, 2].max() + p3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)  # pylint: disable=no-member

    for ci, bone in enumerate(skeleton):
        c = mplcm.get_cmap('tab20')((ci % 20 + 0.05) / 20)  # Same coloring as Pifpaf preds
        ax.plot(p3d[bone, 0], p3d[bone, 1], p3d[bone, 2], color=c)

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig

    return FuncAnimation(fig, animate, frames=360, interval=100)


def print_associations():
    print("\nAssociations of the lane skeleton with 24 keypoints")
    for j1, j2 in LANE_SKELETON_24:
        print(LANE_KEYPOINTS_24[j1 - 1], '-', LANE_KEYPOINTS_24[j2 - 1])


def main():
    print_associations()
# ===========================================================================================
#     draw_skeletons(LANE_POSE_STRAIGHT_24, sigmas = LANE_SIGMAS_24, skel = LANE_SKELETON_24,
#                    kps = CAR_KEYPOINTS_24, scr_weights = LANE_SCORE_WEIGHTS_24)
# 
# ===========================================================================================
    
    with openpifpaf.show.Canvas.blank(nomargin=True) as ax_2D:
        anim_24 = plot3d_red(ax_2D, LANE_POSE_STRAIGHT_24, LANE_SKELETON_24)
        anim_24.save('openpifpaf/plugins/openlane/docs/LANE_24_STRAIGHT_Pose.gif', fps=30)


if __name__ == '__main__':
    main()
