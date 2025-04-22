# tests/test_plotter.py
import sys
import os

# 获取当前测试文件所在目录（test目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径（假设test和src同级）
project_root = os.path.dirname(current_dir)
# 将src目录加入Python路径
sys.path.insert(0, os.path.join(project_root, "src"))


import numpy as np
from plotter import RRTPlotter

class TestRRTPlotter(unittest.TestCase):
    def test_node_label_visibility(self):
        plotter = RRTPlotter()
        node, _ = plotter.add_circle_node((0,0), label_visible=True)
        self.assertIsNotNone(node.text)
        
    def test_edge_connection(self):
        plotter = RRTPlotter()
        node1 = plotter.add_circle_node((0,0))[0]
        node2 = plotter.add_circle_node((5,5))[0]
        arrow = plotter.connect_nodes(node1, node2)[0]
        self.assertIsInstance(arrow, Annotation)