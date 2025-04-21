import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.text import Annotation

class RRTPlotter:    
    def __init__(self, figsize=(10, 8), dpi=100, boundary_view=True):
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.ax.set_aspect('equal')
        self.ax.axis('off')  # 隐藏坐标轴

        # 为图窗绘制边框
        if boundary_view:
            self.ax.add_patch(Rectangle((0, 0), figsize[0], figsize[1], 
                                    edgecolor='black', linewidth=1, fill=False))

        # 设置坐标轴大小为figsize的0.8倍
        self.ax.set_xlim(0, 1.1*figsize[0])
        self.ax.set_ylim(0, 1.1*figsize[1])
        self._arrow_num = 0  # 初始化arrow_num为0
        self._obstacle_num = 0 # 初始化obstacle_num为0
        self._node_num = 0 # 初始化node_num为0

        self.stype = RRTStyleConfig() # 初始化stype为RRTStyleConfig()

    ##############################################
    # 第一层：基础绘制层（返回原生matplotlib对象）
    ##############################################
    def draw_base_circle(self, center, **kwargs) -> Circle:
        """基础圆形绘制（无标签）"""
        params = {**self.style.node, **kwargs}
        circle = Circle(center, radius=params['radius'], 
                       color=params['color'], alpha=params['alpha'])
        self.ax.add_patch(circle)
        return circle
    
    def _draw_base_arrow(self, start, end, **kwargs) -> Annotation:
        """基础箭头绘制（无标签）"""
        params = {**self.style.arrow, **kwargs}
        return self.ax.annotate(
            '', xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle=params['arrow_style'],
                color=params['color'],
                lw=params['linewidth']
            )
        )
    
    def _draw_base_obstacle(self, vertices, **kwargs) -> Polygon:
        """基础障碍物绘制"""
        params = {**self.style.obstacle, **kwargs}
        
        if not np.allclose(vertices[0], vertices[-1]): # 确保路径闭合
            vertices = np.vstack([vertices, vertices[0]])
        
        obstacle = Polygon(vertices, closed=True, 
                          color=params['color'], alpha=params['alpoha'])
        return obstacle

    ##############################################
    # 第二层：组合元素层（基础元素+标签）
    ##############################################
    def add_circle(self, center, label=None, **kwargs) -> tuple[Circle, Annotation]:
        """带标签的圆形节点"""
        circle = self.draw_base_circle(center, **kwargs)
        text = None
        if label is not None:
            text = self.ax.text(
                center[0], center[1]+circle.radius*1.2,
                str(label),
                **self.style.node['fontdict']
            )
        return circle, text
    
    def add_circle(self, center, radius=0.2, color='blue', alpha=1.0, label=None, fontdict={}, **kwargs):
        """绘制实心圆点（支持标签自动避让）"""
        circle = Circle(center, radius=radius, color=color, alpha=alpha)
        self.ax.add_patch(circle)
        if label is not None:
            self.ax.text(center[0], center[1]+radius*1.2, str(label), **fontdict,
                        ha='center', va='bottom')
        return circle

    def add_arrow(self, start, end, 
                 label=None, label_offset=0.2, label_visual = True, fontdict={}, arrowprops = {}, **kwargs):
        """绘制带标签的有向箭头"""
        self._arrow_num += 1  # 每添加一个箭头，arrow_num加1
        if label is None:
            label = self._arrow_num  # 如果label为None，则使用当前arrow_num作为默认值    

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        # 计算箭头方向避免遮挡圆点
        length = np.hypot(dx, dy)
        # 使用 annotate 实现
        arrow_annotation = self.ax.annotate(
                    text='',  # 无文本
                    xy=end,   
                    xytext=start,  # 箭头起点
                    arrowprops= arrowprops
                    )
        # 添加标签
        if label_visual:
            # 分别解算x y的箭头的偏移方向，与箭头方向垂直，顺时针90°方向
            off_x, off_y = np.array([-dy, dx]) / np.linalg.norm([-dy, dx])
            label_x = start[0] + dx*0.5 + off_x*label_offset
            label_y = start[1] + dy*0.5 + off_y*label_offset            
            self.ax.text(label_x, label_y, str(label), **fontdict,
                        ha= 'center', va='center')
        
        return arrow_annotation


    def add_obstacle(self, vertices, color='black', alpha=1):
        """添加多边形障碍物（自动闭合路径）"""
        if not np.allclose(vertices[0], vertices[-1]): # 确保路径闭合
            vertices = np.vstack([vertices, vertices[0]])
        obstacle = Polygon(vertices, closed=True, 
                          color=color, alpha=alpha)
        self.ax.add_patch(obstacle)

    def add_annotation(self, position, text, color='red', fontsize=12, 
                       rotation=0, box_style=None,  fontdict={}):
        """添加文字标注（支持旋转和文本框样式）"""
        self.ax.text(position[0], position[1], text, **fontdict,
                    color=color,  rotation=rotation,
                    bbox=dict(boxstyle=box_style, color='white', alpha=0.8))
        
    def radial_arrow(plotter, circle, direction_angle, length=1.0, 
                 color='purple', label=None, fontdict={}):
        """从圆心沿指定角度方向绘制箭头"""
        x, y, r = circle
        end_x = x + (r + length) * np.cos(np.radians(direction_angle))
        end_y = y + (r + length) * np.sin(np.radians(direction_angle))
        plotter.add_arrow((x, y), (end_x, end_y), color=color, label=label, fontdict = fontdict)

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

    def add_arrow_with_angle(self, start = (0,0), angle = 0, length = 1, **kwargs):
        """添加带角度的箭头\n
        可选参数列表： label, label_offset, label_visual, fontdict, arrowprops"""
        end = (start[0] + length * np.cos(angle), 
               start[1] + length * np.sin(angle))
        return self.add_arrow(start, end, **kwargs)
    
    def add_circle_after_arrow(self, arrow_annotation, radius = 0.2, **kwargs):
        """在箭头末端绘制圆点\n
        可选参数列表：color, alpha, label, fontdict"""
        arrow_start = np.array(arrow_annotation.xyann)
        arrow_end = np.array(arrow_annotation.xy)
        arrow_direction = arrow_end - arrow_start
        # 方向归一化
        arrow_direction /= np.linalg.norm(arrow_direction)
        # 计算圆心位置
        circle_center = arrow_end + arrow_direction * radius
        circle = self.add_circle(circle_center, radius=radius, **kwargs)
        return circle
    
    # 添加一个在末端带节点的箭头
    def add_arrow_with_node(self, start = (0,0), angle = 0, length = 1, radius = 0.2, **kwargs):
        """添加带节点的箭头\n
        可选参数列表:label, label_offset, label_visual, fontdict, arrowprops"""
        # 箭头的长度应减去圆的半径，保证到圆中心的长度等于设定长度
        length = length - radius
        end = (start[0] + length * np.cos(angle), 
               start[1] + length * np.sin(angle))
        arrow_annotation = self.add_arrow(start, end, **kwargs)

        # 如果**kwargs中有将其去除
        node_center = self.add_circle_after_arrow(arrow_annotation, radius,  **kwargs)
        return arrow_annotation, node_center
    
    # 在一个圆后面添加一个在末端带节点的箭头
    def add_arrow_with_node_after_circle(self, circle, angle = 0, length = 1, radius = 0.2, **kwargs):
        """在圆后面添加带节点的箭头\n
        可选参数列表:label, label_offset, label_visual, fontdict, arrowprops"""
        circle_center = circle.center
        circle_radius = circle.radius
        # 计算箭头起点
        start = (circle_center[0] + circle_radius * np.cos(angle), 
                 circle_center[1] + circle_radius * np.sin(angle))
        # 长度应减掉起始圆的半径
        length = length - circle_radius
        # 计算箭头终点
        end = (start[0] + length * np.cos(angle), 
               start[1] + length * np.sin(angle))
        # 调用函数绘制
        return self.add_arrow_with_node(start, angle, length, radius, **kwargs)
    
    # 在类中定义参数模板
    _DEFAULT_STYLE = {
        'node': {
            'color': 'blue',
            'radius': 0.2,
            'fontdict': {}
        },
        'arrow': {
            'color': 'black',
            'linewidth': 1.5,
            'arrow_style': '-|>'
        }
    }


class RRTStyleConfig:
    """全局绘图样式配置类"""
    def __init__(self):
        # 节点默认样式
        self.node = {
            'color': 'blue',
            'radius': 0.2,
            'alpha': 1.0,
            'fontdict': {}
        }
        
        # 箭头默认样式
        self.arrow = {
            'color': 'black',
            'linewidth': 1.5,
            'arrow_style': '-|>',
            'label_offset': 0.2,
            'fontdict': {}
        }
        
        # 障碍物默认样式
        self.obstacle = {
            'color': 'black',
            'alpha': 0.6
        }
        
        # 标注默认样式
        self.annotation = {
            'color': 'red',
            'fontsize': 12,
            'box_style': None,
            'fontdict': {}
        }

    def update_style(self, element_type: str, **kwargs):
        """更新指定元素的全局样式"""
        if hasattr(self, element_type):
            getattr(self, element_type).update(kwargs)
        else:
            raise ValueError(f"未知的元素类型: {element_type}")