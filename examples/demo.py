# examples/demo.py
import sys
import os
import numpy as np

# 解决路径问题：将src目录添加到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), "src")
sys.path.insert(0, src_dir)

from plotter import RRTPlotter, RRTStyleConfig

class RRTNode:
    """简单的RRT节点结构"""
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.children = []
        self.element = None  # 关联的绘图元素

def main():
    # 创建自定义样式配置
    style = RRTStyleConfig()
    style.update_style('node', color='#3498db', radius=0.3)
    style.update_style('obstacle', color='#e74c3c', alpha=0.7)

    # 初始化绘图器
    plotter = RRTPlotter(figsize=(10.5, 12), style_config=style)

    # 构建简单树结构
    root = RRTNode((2, 5))
    node1 = RRTNode((4, 8), parent=root)
    node2 = RRTNode((3, 10), parent=root)
    node3 = RRTNode((6 ,10), parent=node1)
    root.children = [node1, node2]
    node1.children = [node3]

    # 绘制主树结构
    plotter.draw_rrt_tree(
            root,
            tree_type="classic",
            node_label=True,
            edge_label=False,
            start_angle=np.radians(-45),
            start_length=1.2
        )
    
    # 标记root到node1和node2的角度差
    plotter.add_angle_annotation(
        node1.edge_element,
        node2.edge_element,
        arc_radius = 1.5,
        fontdict={"fontsize": 12, "fontweight": "bold", "fontfamily": "Times New Roman", "color": "red"},
        arrowprops={"color": "red"},
    )

    # 添加障碍物
    obstacle1 = np.array([[3, 4], [3.5, 5], [2.5, 5.5]])
    obstacle2 = np.array([[2, 1], [8, 1], [8, 3],[2,3]])
    plotter.add_labeled_obstacle(obstacle1, label_visible=True)
    plotter.add_labeled_obstacle(obstacle2, label_visible=True, color = 'black')

    # 添加角度标注示例
    node_a = plotter.add_circle_node((6, 5), label_visible=True)[0]
    node_b = plotter.add_circle_node((8, 6))[0]
    node_c = plotter.add_circle_node((7, 8))[0]
    
    edge_ab = plotter.connect_nodes(node_a, node_b)[0]
    edge_ac = plotter.connect_nodes(node_a, node_c)[0]
    
    plotter.add_angle_annotation(edge_ab, edge_ac, label="θ")


    # 添加路径扩展示例
    plotter.radial_extension(
        node_c, 
        direction_angle=30,
        length=2.5,
        color='#2ecc71',
        linestyle='--'
    )

    # 保存并显示结果
    output_path = os.path.join(os.path.dirname(current_dir), "docs", "example_figures", "plotter.png")
    plotter.save(output_path, dpi=150)
    plotter.show()

if __name__ == "__main__":
    main()