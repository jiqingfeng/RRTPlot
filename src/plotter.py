import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Annotation  
from matplotlib.patches import Circle, Rectangle, Polygon, Arc
from typing import Tuple, Optional, Dict, Any
from functools import wraps

class RRTStyleConfig:
    """全局绘图样式配置类"""
    def __init__(self):
        # 节点默认样式
        self.node = {
            'color': 'black',
            'radius': 0.2,
            'alpha': 1.0,
            'fontdict': {'fontsize': 14,'fontfamily': 'Times New Roman'}
        }
        
        # 箭头默认样式
        self.arrow = {
            'color': 'black',
            'linewidth': 1.5,
            'arrowstyle': '-|>',
            'mutation_scale': 20,
            'label_offset': 0.2,
            'fontdict': {'fontsize': 14,'fontfamily': 'Times New Roman'}
        }
        
        # 障碍物默认样式
        self.obstacle = {
            'color': 'black',
            'alpha': 1,
            'fontdict': {'fontsize': 14,'fontfamily': 'Times New Roman'}
        }
        
        # 标注默认样式
        self.annotation = {
            'color': 'black',
            'fontsize': 16,
            'fontdict': {'fontweight': 'bold', 
                         'fontfamily': 'Times New Roman',
                         }
        }

        # 新增角度标注样式配置
        self.angle = {
            'color': '#e67e22',
            'linewidth': 1.2,
            'arc_radius': 1.5,
            'label_offset': 0.8,
            'arrowprops': {
                'arrowstyle': '->',
                'connectionstyle': 'arc3,rad=0.2',
                'linestyle': 'dashed',
                'alpha': 0.6
            },
            'fontdict': {
                'fontsize': 12,
                'fontweight': 'bold',
                'fontfamily': 'Times New Roman'
            }
        }

    def update_style(self, element_type: str, **kwargs):
        """更新指定元素的全局样式"""
        if hasattr(self, element_type):
            getattr(self, element_type).update(kwargs)
        else:
            raise ValueError(f"未知的元素类型: {element_type}")

def validate_params(valid_keys: list):
    """参数验证装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            filtered = {k:v for k,v in kwargs.items() if k in valid_keys}
            return func(self, *args, **filtered)
        return wrapper
    return decorator

class RRTPlotter:
    def __init__(self, 
                 figsize: Tuple[float, float] = (10, 8),
                 dpi: int = 100,
                 boundary_view: bool = True,
                 style_config: Optional[RRTStyleConfig] = None):
        
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # 初始化样式配置
        self.style = style_config if style_config else RRTStyleConfig()
        
        # 绘制画布边框
        if boundary_view:
            border = Rectangle((0, 0), figsize[0], figsize[1], 
                             edgecolor='black', linewidth=1, fill=False)
            self.ax.add_patch(border)
        
        # 设置坐标范围
        self.ax.set_xlim(0, figsize[0]*1.1)
        self.ax.set_ylim(0, figsize[1]*1.1)
        
        # 初始化计数器
        self._element_counters = {'node': 0, 'edge': 0, 'obstacle': 0} # 记录节点数、边数和障碍物数

    ##############################################
    # 第一层：基础绘制方法
    ##############################################
    @validate_params(['color', 'radius', 'alpha'])
    def base_circle(self, center: Tuple[float, float], **kwargs) -> Circle:
        """绘制基础圆形"""
        params = {**self.style.node, **kwargs}
        # 去掉字体参数
        params.pop('fontdict', None)
        circle = Circle(center, **params)
        self.ax.add_patch(circle)
        return circle

    @validate_params(['color', 'linewidth', 'arrowstyle'])
    def base_arrow(self, 
                  start: Tuple[float, float], 
                  end: Tuple[float, float], 
                  **kwargs) -> Annotation:
        """绘制基础箭头"""
        params = {**self.style.arrow, **kwargs}
        # 去掉字体参数
        params.pop('fontdict', None)
        arrow = self.ax.annotate(
            '', xy=end, xytext=start,
            arrowprops={k: v for k, v in params.items() 
                      if k in ['arrowstyle', 'color', 'linewidth', 'mutation_scale']}
        )
        return arrow

    @validate_params(['color', 'alpha'])
    def base_obstacle(self, vertices: np.ndarray, **kwargs) -> Polygon:
        """绘制基础障碍物"""
        params = {**self.style.obstacle, **kwargs}
        if not np.allclose(vertices[0], vertices[-1]):
            vertices = np.vstack([vertices, vertices[0]])
        # 去掉字体参数
        params.pop('fontdict', None)
        obstacle = Polygon(vertices, closed=True, **params)
        self.ax.add_patch(obstacle)
        return obstacle

    ##############################################
    # 第二层：组合元素（基础元素+标签）
    ##############################################
    def add_circle_node(self, 
                 center: Tuple[float, float],
                 label: Optional[str] = None,
                 label_visible: bool = False,
                 **kwargs) -> Tuple[Circle, Optional[Annotation]]:
        """添加带标签的圆形节点"""
        # 更新编号
        self._element_counters['node'] += 1
        # 获取自动编号
        
        if label is None:
            label = str(self._element_counters['node'])
        
        # 绘制基础圆形
        circle = self.base_circle(center, **kwargs)
        
        # 添加标签
        text = None
        if label_visible:
            text = self.ax.text(
                center[0], center[1] - circle.radius * 2,
                label,
                **self.style.node['fontdict']
            )
        return circle, text

    def add_edge(self,
                start: Tuple[float, float],
                end: Tuple[float, float],
                label: Optional[str] = None,
                label_visible: bool = False,
                **kwargs) -> Tuple[Annotation, Optional[Annotation]]:
        """添加带标签的箭头"""
        # 更新编号
        self._element_counters['edge'] += 1
        # 获取自动编号
        if label is None:
            label = str(self._element_counters['edge'])
        
        # 绘制基础箭头
        arrow = self.base_arrow(start, end, **kwargs)
        
        # 计算标签位置
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        off_x, off_y = np.array([-dy, dx]) / np.linalg.norm([-dy, dx])
        
        label_x = start[0] + dx*0.5 + off_x * self.style.arrow['label_offset']
        label_y = start[1] + dy*0.5 + off_y * self.style.arrow['label_offset']
        
        # 添加标签
        text = None
        if label_visible:
            text = self.ax.text(label_x, label_y, label, **self.style.arrow['fontdict'])
        return arrow, text
    
    def add_labeled_obstacle(self,
                           vertices: np.ndarray,
                           label: Optional[str] = None,
                           label_visible: bool = False,
                           **kwargs) -> Tuple[Polygon, Optional[Annotation]]:
        # 更新障碍物计数器
        self._element_counters['obstacle'] += 1        
        # 自动生成标签编号
        if label is None:
            label = f"{self._element_counters['obstacle']}"
        
        # 绘制基础障碍物
        obstacle = self.base_obstacle(vertices, **kwargs)
                
        # 添加标签
        text = None
        if label_visible:
            # 计算精确几何中心
            closed_vertices = vertices if np.allclose(vertices[0], vertices[-1]) else np.vstack([vertices, vertices[0]])
            centroid = np.mean(closed_vertices[:-1], axis=0)  # 排除重复的闭合点

             # 获取障碍物实际颜色（处理颜色继承逻辑）
            obstacle_color = obstacle.get_facecolor()[:3]  # 获取RGB值，忽略alpha
            
            # 计算反色（最大对比度）
            inverse_color = tuple(1 - val for val in obstacle_color)

            text = self.ax.text(
            centroid[0], centroid[1],
            label,
            color=inverse_color,
            **{
                **self.style.obstacle['fontdict'],
                'ha': 'center',  # 水平居中
                'va': 'center'   # 垂直居中
            }
        )          
        return obstacle, text

    ##############################################
    # 第三层：高级逻辑方法
    ##############################################
    def connect_nodes(self,
                    start_node: Circle,
                    end_node: Circle,
                    **kwargs) -> Dict[str, Any]:
        """连接两个节点"""
        # 计算连接路径
        angle = np.arctan2(end_node.center[1]-start_node.center[1],
                         end_node.center[0]-start_node.center[0])
        
        start = (
            start_node.center[0] + start_node.radius * np.cos(angle),
            start_node.center[1] + start_node.radius * np.sin(angle)
        )
        end = (
            end_node.center[0] - end_node.radius * np.cos(angle),
            end_node.center[1] - end_node.radius * np.sin(angle)
        )
        
        # 绘制箭头
        arrow, arrow_text = self.add_edge(start, end, **kwargs)
        
        return {
            'start_node': start_node,
            'end_node': end_node,
            'arrow': arrow,
            'arrow_text': arrow_text
        }

    def add_path_segment(self,
                       start: Tuple[float, float],
                       angle: float,
                       length: float,
                       **kwargs) -> Dict[str, Any]:
        """添加带末端节点的路径段"""
        # 计算终点
        # 实际路径长度应减去圆节点的半径
        # 如果没有传入半径，则获取全局配置半径
        radius = kwargs.get('radius', self.style.node['radius'])
        segment_end = (
            start[0] + length * np.cos(angle),
            start[1] + length * np.sin(angle)
        )        
        length = length - radius
        end_circle_center = (
            segment_end[0] + radius * np.cos(angle),
            segment_end[1] + radius * np.sin(angle)
        )
        
        
        # 绘制箭头
        arrow, arrow_text = self.add_edge(start, segment_end, **kwargs)
        
        # 绘制末端节点
        node, node_text = self.add_circle_node(end_circle_center, **kwargs)
        
        return {
            'arrow': arrow,
            'arrow_text': arrow_text,
            'node': node,
            'node_text': node_text
        }

    def radial_extension(self,
                       base_node: Circle,
                       direction_angle: float,
                       length: float,
                       **kwargs) -> Dict[str, Any]:
        """从现有节点径向扩展"""
        # 计算起点
        start = (
            base_node.center[0] + base_node.radius * np.cos(direction_angle),
            base_node.center[1] + base_node.radius * np.sin(direction_angle)
        )
        # 修正长度，长度减去起点端圆的半径
        length = length - base_node.radius
        
        # 添加路径段
        return self.add_path_segment(
            start,
            np.radians(direction_angle),
            length - base_node.radius,
            **kwargs
        )
    
    def add_angle_annotation(self,
                           edge1: Annotation,
                           edge2: Annotation,
                           label: Optional[str] = None,
                           extension_length: float = 50,
                           **kwargs) -> Dict[str, Any]:
        """
        标注两个有向边之间的夹角
        :param edge1: 第一个边的Annotation对象
        :param edge2: 第二个边的Annotation对象
        :param label: 自定义标签文本（默认显示角度值）
        :param extension_length: 延长线基准长度
        :return: 包含标注元素的字典
        """
        # 合并样式参数
        style_params = {**self.style.angle, **kwargs}
        
        # 获取边坐标
        def get_edge_coords(edge):
            return np.array([edge.xyann, edge.xy])
        
        line1 = get_edge_coords(edge1)
        line2 = get_edge_coords(edge2)

        # 计算原始交点
        intersect_point, need_extension = self._find_intersection(line1, line2)
        # need_extension = False
        
        # 情况1：存在直接交点
        if intersect_point is not None:
            # 检查交点两侧长度
            d1_before = np.linalg.norm(line1[0] - intersect_point)
            d1_after = np.linalg.norm(line1[1] - intersect_point)
            d2_before = np.linalg.norm(line2[0] - intersect_point)
            d2_after = np.linalg.norm(line2[1] - intersect_point)
            
            if min(d1_before, d1_after, d2_before, d2_after) < style_params['arc_radius']:
                need_extension = True
        else:
            need_extension = True

        # 情况2：需要延长处理
        if need_extension:
            # 延长第一条边

            extended_line1 = self._extend_line(line1, extension_length)
            intersect_point = self._find_intersection(extended_line1, line2)
            
            # 绘制延长线
            extend_arrow, _ = self.add_edge(
                line1[1], extended_line1[1],
                linestyle='--',
                alpha=0.3,
                color=style_params['color']
            )
        else:
            extend_arrow = None

        # 计算角度参数
        v1 = line1[1] - line1[0] if not need_extension else extended_line1[1] - line1[0]
        v2 = line2[1] - line2[0]
        
        angle, angle_deg, is_acute = self._calculate_angle(v1, v2)
        
        # 确定标注位置
        arc_radius = style_params['arc_radius']

        label_pos_radius =  (arc_radius * style_params['label_offset'])
        label_pos_direction = self._get_label_offset(v1, v2)

        label_pos = (intersect_point[0][0] + label_pos_radius * label_pos_direction[0],
                     intersect_point[0][1] + label_pos_radius * label_pos_direction[1])
        a = intersect_point[0][0]
        b = intersect_point[0][1]
        arc_center = (a,b)
        # 绘制圆弧
        arc = Arc(arc_center, 
                2*arc_radius, 2*arc_radius,
                theta1=np.degrees(angle[0]), 
                theta2=np.degrees(angle[1]),
                color=style_params['color'],
                linewidth=style_params['linewidth'])
        self.ax.add_patch(arc)

        # 添加标注文本
        label_text = label if label else f"{angle_deg:.1f}°"
        text = self.ax.annotate(
            label_text,
            xy=label_pos,
            xytext=label_pos,
            **style_params['fontdict'],
            arrowprops=style_params['arrowprops'] if is_acute else None
        )

        return {
            'arc': arc,
            'text': text,
            'extend_arrow': extend_arrow,
            'intersection_point': intersect_point
        }


    ##############################################
    # 工具方法
    ##############################################
    def add_annotation(self,
                     position: Tuple[float, float],
                     text: str,
                     **kwargs) -> Annotation:
        """添加文字标注"""
        params = {**self.style.annotation, **kwargs}
        return self.ax.text(
            position[0], position[1], text,
            color=params['color'],
            fontsize=params['fontsize'],
            **params['fontdict']
        )
    


    def show(self):
        """显示绘图"""
        plt.show()

    def save(self, filename: str, dpi: int = 300):
        """保存图像"""
        self.fig.savefig(filename, bbox_inches='tight', dpi=dpi)

    # 几何计算辅助方法
    def _find_intersection(self, line1, line2):
        """
        计算两条线段的交点及是否实际相交
        :return: (交点坐标, 是否实际相交)
        """
        # 将输入转换为numpy数组
        p1, p2 = np.array(line1)
        p3, p4 = np.array(line2)

        # 计算分母项
        denominator = (p4[1]-p3[1])*(p2[0]-p1[0]) - (p4[0]-p3[0])*(p2[1]-p1[1])

        # 处理平行情况
        if np.isclose(denominator, 0):
            # 检查是否共线
            if np.allclose(np.cross(p3-p1, p2-p1), 0):
                # 在共线时返回第一个交点（如果有）
                t_values = []
                for p in [p3, p4]:
                    t = (p[0]-p1[0])/(p2[0]-p1[0]) if not np.isclose(p2[0], p1[0]) else \
                        (p[1]-p1[1])/(p2[1]-p1[1])
                    if 0 <= t <= 1:
                        t_values.append(t)
                if t_values:
                    t = np.mean(t_values)
                    intersect = p1 + t*(p2-p1)
                    return tuple(intersect), True
            # 完全平行不相交
            return None, False

        # 计算参数t和u
        numerator_t = (p4[0]-p3[0])*(p1[1]-p3[1]) - (p4[1]-p3[1])*(p1[0]-p3[0])
        numerator_u = (p2[0]-p1[0])*(p1[1]-p3[1]) - (p2[1]-p1[1])*(p1[0]-p3[0])
        t = numerator_t / denominator
        u = numerator_u / denominator

        # 计算交点坐标
        intersect = p1 + t*(p2 - p1)

        # 判断是否在线段范围内
        is_intersecting = (0 <= t <= 1) and (0 <= u <= 1)

        return (float(intersect[0]), float(intersect[1])), is_intersecting


    def _extend_line(self, line, length):
        """按方向延长线段"""
        direction = line[1] - line[0]
        unit_vector = direction / np.linalg.norm(direction)
        new_end = line[1] + unit_vector * length
        return np.array([line[0], new_end])

    def _calculate_angle(self, v1, v2):
        """计算转向角度"""
        angle_v1 = np.arctan2(v1[1], v1[0])
        angle_v2 = np.arctan2(v2[1], v2[0])
        angle = angle_v2 - angle_v1
        angle_deg = np.degrees(angle) % 360
        
        # 确定绘制方向
        if angle_deg > 180:
            start_angle = angle_v1
            end_angle = angle_v2
            is_acute = False
        else:
            start_angle = angle_v2
            end_angle = angle_v1
            is_acute = True
        
        return (start_angle, end_angle), angle_deg, is_acute

    def _get_label_offset(self, v1, v2):
        """计算标签偏移方向"""
        normal = np.array([-v1[1], v1[0]])
        if np.dot(normal, v2) > 0:
            return normal / np.linalg.norm(normal)
        return -normal / np.linalg.norm(normal)