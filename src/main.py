import numpy as np

from plotter import RRTPlotter


# 示例1：经典树结构（已通过测试）
class RRTNode:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.children = []


if __name__ == "__main__":
    # test_angle_annotation()
    root = RRTNode((5, 3))
    node1 = RRTNode((1, 1), parent=root)

    root.children.append(node1)

    plotter = RRTPlotter()
    plotter.draw_rrt_tree(
        root,
        start_angle=np.radians(-30),
        start_length=1.5,
        node_label=False,
        edge_label=True,
    )
    plotter.show()
