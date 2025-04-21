import numpy as np
import matplotlib.pyplot as plt
from plotter import RRTPlotter

# 创建字体字典
font_label = {'fontname': 'Times New Roman', 'fontsize': '14'}
title_label = {'fontname': 'Times New Roman', 'fontsize': '18', 'fontweight': 'bold'}
# 初始化绘图器
plotter = RRTPlotter(figsize=(12, 8))

# 添加障碍物
obs1 = np.array([[2, 2], [3, 2], [3, 4], [1.5, 4]])
plotter.add_obstacle(obs1, color='black')
plotter.add_obstacle(np.array([[5, 1], [7, 1], [6, 3]]), color='black')

# 添加节点（起点、终点、中间节点）
start_node = (1, 1, 0.3)
goal_node = (8, 6, 0.3)
nodes = [
    (1, 1, 0.3), (2.5, 3, 0.2), (4, 4, 0.2),
    (5.5, 5, 0.2), (7, 5.5, 0.2), (8, 6, 0.3)
]

for idx, (x, y, r) in enumerate(nodes):
    color = '#e74c3c' if (x,y,r)==start_node else '#2ecc71' if (x,y,r)==goal_node else '#3498db'
    plotter.add_circle((x, y), radius=r, color=color, label=f'q{idx}', fontdict=font_label)

# 连接节点（模拟RRT扩展路径）
plotter.connect_circles(nodes[0], nodes[1], label='Step 1', color='#2980b9', fontdict=font_label)
plotter.connect_circles(nodes[1], nodes[2], label='Step 2', color='#2980b9', fontdict=font_label)
plotter.connect_circles( nodes[3], nodes[4], label='Step 4', color='#2980b9', fontdict=font_label)

# 添加特殊箭头（显示随机方向采样）
plotter.radial_arrow(nodes[2], direction_angle=60, length=1.5, 
            color='#f39c12', label='Random\nDirection', fontdict=font_label)

# 添加全局标注
plotter.add_annotation((4, 7), "RRT Path Search Process", 
                      fontsize=14, color='#2c3e50', 
                      box_style='round,pad=0.3', fontdict=title_label)
plt.show()

# 导出矢量图
# plt.savefig('rrt_diagram.png', bbox_inches='tight', dpi=300)
# plt.close()