import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.text import Annotation

class RRTPlotter:
    def __init__(self, figsize=(10, 8), dpi=100):
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.ax.set_aspect('equal')
        self.ax.axis('on')  # 隐藏坐标轴
        # 设置坐标轴大小为figsize的0.8倍
        self.ax.set_xlim(0, figsize[0]*0.8)
        self.ax.set_ylim(0, figsize[1]*0.8)

    def add_circle(self, center, radius=0.2, color='blue', alpha=1.0, label=None):
        """绘制实心圆点（支持标签自动避让）"""
        circle = Circle(center, radius=radius, color=color, alpha=alpha)
        self.ax.add_patch(circle)
        if label is not None:
            self.ax.text(center[0], center[1]+radius*1.2, str(label), 
                        ha='center', va='bottom', fontsize=8)

    def add_arrow(self, start, end, color='black', linewidth=1.5, 
                 arrow_style='->', head_width=0.1, label=None, label_offset=0.2):
        """绘制带标签的有向箭头"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        # 计算箭头方向避免遮挡圆点
        length = np.hypot(dx, dy)
        u = dx / length * 0.9  # 缩短10%避免覆盖起点
        v = dy / length * 0.9
        self.ax.arrow(start[0]+u*0.1, start[1]+v*0.1, u, v, 
                     shape='full', lw=linewidth, color=color, 
                     head_width=head_width, head_length=head_width*1.5)
        # 使用 annotate 实现
        # self.ax.annotate(
        #     text='',  # 无文本
        #     xy=end,   
        #     xytext=start,  # 箭头起点
        #     arrowprops=dict(
        #         arrowstyle='->', 
        #         color=color,
        #         lw=linewidth,
        #         shrinkA=0,  # 起点不收缩
        #         shrinkB=0,  # 终点不收缩
        #         headwidth=0.3,
        #         headlength=0.3
        #         )
        #     )

        # 添加标签
        if label is not None:
            label_x = start[0] + dx*0.5 + np.sign(dx)*label_offset
            label_y = start[1] + dy*0.5 + np.sign(dy)*label_offset
            self.ax.text(label_x, label_y, str(label), 
                        color=color, fontsize=9, ha='center', va='center')

    def add_obstacle(self, vertices, color='gray', alpha=0.6):
        """添加多边形障碍物（自动闭合路径）"""
        if not np.allclose(vertices[0], vertices[-1]): # 确保路径闭合
            vertices = np.vstack([vertices, vertices[0]])
        obstacle = Polygon(vertices, closed=True, 
                          color=color, alpha=alpha)
        self.ax.add_patch(obstacle)

    def add_annotation(self, position, text, color='red', fontsize=12, 
                       rotation=0, box_style=None):
        """添加文字标注（支持旋转和文本框样式）"""
        self.ax.text(position[0], position[1], text, 
                    color=color, fontsize=fontsize, rotation=rotation,
                    bbox=dict(boxstyle=box_style, color='white', alpha=0.8))
        
    def radial_arrow(plotter, circle, direction_angle, length=1.0, 
                 color='purple', label=None):
        """从圆心沿指定角度方向绘制箭头"""
        x, y, r = circle
        end_x = x + (r + length) * np.cos(np.radians(direction_angle))
        end_y = y + (r + length) * np.sin(np.radians(direction_angle))
        plotter.add_arrow((x, y), (end_x, end_y), color=color, label=label)

    def connect_circles(plotter, circle1, circle2, color='blue', 
                    label=None, **kwargs):
        """连接两个圆对象（自动计算切线方向）"""
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2
        
        # 计算方向向量
        dx = x2 - x1
        dy = y2 - y1
        angle = np.arctan2(dy, dx)
        
        # 计算起点和终点（圆边缘）
        start = x1 + r1 * np.cos(angle), (y1 + r1 * np.sin(angle))
        end = x2 - r2 * np.cos(angle), (y2 - r2 * np.sin(angle))
        
        plotter.add_arrow(start, end, color=color, label=label, **kwargs)