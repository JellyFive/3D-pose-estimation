import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from .Math import get_corners
mpl.use('QT5Agg')


def draw_projection(corners, P2, ax, color):
    projection = np.dot(P2, np.vstack([corners, np.ones(8, dtype=np.int32)]))
    projection = (projection / projection[2])[:2]
    orders = [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [2, 6], [3, 7], [1, 5], [0, 4]]
    for order in orders:
        ax.plot(projection[0, order],
                projection[1, order],
                color=color,
                linewidth=2)
    return


def draw_2dbbox(bbox, ax, color):
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]
    ax.add_patch(
        plt.Rectangle((xmin, ymin),
                      xmax - xmin,
                      ymax - ymin,
                      color=color,
                      fill=False,
                      linewidth=2))
    ax.text(xmin,
            ymin,
            'building',
            size='x-large',
            color='white',
            bbox={
                'facecolor': 'green',
                'alpha': 1.0
            })
    return


def draw(image,
         bbox,
         proj_matrix,
         dimensions,
         gt_trans,
         est_trans,
         rotation_x,
         rotation_y,
         rotation_z=0):
    fig = plt.figure(figsize=(8, 8))

    # 绘制3DBBOX
    ax = fig.gca()
    ax.grid(False)
    ax.set_axis_off()
    ax.imshow(image)

    # 获取8个顶点的世界坐标
    truth_corners = get_corners(dimensions, gt_trans, rotation_x, rotation_y,
                                rotation_z)

    est_corners = get_corners(dimensions, est_trans, rotation_x, rotation_y,
                              rotation_z)

    draw_projection(truth_corners, proj_matrix, ax, 'orange')  # 真实3D框
    draw_projection(est_corners, proj_matrix, ax, 'red')  # 预测3D框

    draw_2dbbox(bbox, ax, 'green')