import numpy as np
import matplotlib.pyplot as plt
from plotter import RRTPlotter


# 创建字体字典
font_label = {'fontname': 'Times New Roman', 'fontsize': '18', 'color': 'black'}
title_label = {'fontname': 'Times New Roman', 'fontsize': '18', 'fontweight': 'bold'}

# 初始化全局属性
node_color = 'black'
line_color = 'black'
obstacle_color = 'black'

# 线宽
line_width = 2

edge_length = 1.5
# 箭头类型
arrowprops =  dict(
                arrowstyle='-|>', 
                color=line_color,
                lw=line_width,                
                shrinkA=0,  # 起点不收缩
                shrinkB=0,  # 终点不收缩
                mutation_scale = 20, # 用于缩放箭头样式属性（例如 head_length）的值。
                )

# 节点大小
node_size = 0.15

# 绘图画布的大小
figsize = [12, 8]

# 初始化绘图器
plotter = RRTPlotter(figsize=figsize)


# 在左上角绘制一条长度为edge_length,倾斜角为30度的带节点的箭头
# edge1 = plotter.add_arrow_with_angle(start=(0.5, figsize[1]-0.5), angle=np.radians(-30), length=edge_length, 
#                                 label_offset=0.2, fontdict=font_label, arrowprops=arrowprops)

# # 在指定edge的末端绘制一个指定大小的圆
# node1 = plotter.add_circle_after_arrow(edge1, radius=node_size, color=node_color, fontdict=font_label)

edge1, node1 = plotter.add_arrow_with_node(start=(0.5, figsize[1]-0.5),  angle=np.radians(-30), length=edge_length, radius = node_size,
                                label_offset=0.2, fontdict=font_label, arrowprops=arrowprops, color = node_color)

edge2, node2 = plotter.add_arrow_with_node_after_circle(node1,  angle=np.radians(-30), length=edge_length, radius = node_size,
                                label_offset=0.2, fontdict=font_label, arrowprops=arrowprops, color = node_color)






# # 添加障碍物
# obs1 = np.array([[2, 2], [3, 2], [3, 4], [1.5, 4]])
# plotter.add_obstacle(obs1, color='black')
# plotter.add_obstacle(np.array([[5, 1], [7, 1], [6, 3]]), color='black')



# # 添加节点（起点、终点、中间节点）
# start_node = (1, 1, 0.3)
# goal_node = (8, 6, 0.3)
# nodes = [
#     (1, 1, 0.3), (2.5, 3, 0.2), (4, 4, 0.2),
#     (5.5, 5, 0.2), (7, 5.5, 0.2), (8, 6, 0.3)
# ]

# for idx, (x, y, r) in enumerate(nodes):
#     plotter.add_circle((x, y), radius=r, color=node_color,  fontdict=font_label)

# # 连接节点（模拟RRT扩展路径）
# plotter.connect_circles(nodes[0], nodes[1],  color=line_color, fontdict=font_label)
# plotter.connect_circles(nodes[1], nodes[2],  color=line_color, fontdict=font_label)
# plotter.connect_circles( nodes[3], nodes[4],  color=line_color, fontdict=font_label)

# # 添加特殊箭头（显示随机方向采样）
# plotter.radial_arrow(nodes[2], direction_angle=60, length=1.5, 
#             color='#f39c12', label='Random\nDirection', fontdict=font_label)

# # 添加全局标注
# plotter.add_annotation((4, 7), "RRT Path Search Process", 
#                       fontsize=14, color='#2c3e50', 
#                       box_style='round,pad=0.3', fontdict=title_label)
plt.show()

# # 导出矢量图
# # plt.savefig('rrt_diagram.png', bbox_inches='tight', dpi=300)
# # plt.close()