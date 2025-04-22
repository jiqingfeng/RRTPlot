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
            'arc_radius': 0.5,
            'label_offset': 1.2,
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

        # 辅助线样式
        self.guideline = {
            'color': '#999999',
            'linewidth': 5.0,
            'linestyle': '--',
            'alpha': 0.6
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
    
    # 在RRTPlotter类中添加基础线绘制方法
    @validate_params(['color', 'linewidth', 'linestyle', 'alpha'])
    def add_line(self,
                start: Tuple[float, float],
                end: Tuple[float, float],
                **kwargs) -> plt.Line2D:
        """绘制基础线段"""
        params = {**self.style.guideline, **kwargs}
        line = self.ax.add_line(plt.Line2D(
                    [start[0], end[0]], 
                    [start[1], end[1]],
                    **{k: v for k, v in params.items() if k in ['color', 'linewidth', 'linestyle', 'alpha']}
                ))
        return line

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
        
        # 获取线段空间关系
        intersection, (dist1, dist2), angle_diff = self.calculate_segment_relations(edge1, edge2)
        
        # 情况1：直线无交点（平行或重合）
        if intersection is None:
            return None
        
        # 情况2：需要处理线段延长
        line1_orig = np.array([edge1.xyann, edge1.xy])
        line2_orig = np.array([edge2.xyann, edge2.xy])
        

        # 处理第一条线段的延长
        line1_ext, extend_arrow1 = self._extend_to_intersection(
            line1_orig, dist1, intersection,
            style_params['arc_radius']*1.5, 
            style_params
        )
        
        # 处理第二条线段的延长
        line2_ext, extend_arrow2 = self._extend_to_intersection(
            line2_orig, dist2, intersection,
            style_params['arc_radius']*1.5, 
            style_params
        )
        
        # 获取延长后的向量
        v1 = line1_ext[1] - line1_ext[0]
        v2 = line2_ext[1] - line2_ext[0]
        
        # 计算标注角度参数
        (start_angle, end_angle), angle_rad, is_acute = self._calculate_angle(v1, v2)
        

        # 确定标注位置
        arc_radius = style_params['arc_radius']
        label_pos = self._calculate_label_position(
            intersection, 
            v1, v2, 
            arc_radius * style_params['label_offset']
        )
        
        # 绘制圆弧
        arc = Arc(intersection, 
                2*arc_radius, 2*arc_radius,
                theta1=np.degrees(start_angle), 
                theta2=np.degrees(end_angle),
                color=style_params['color'],
                linewidth=style_params['linewidth'])
        self.ax.add_patch(arc)
        
        # 添加标注文本
        label_text = label if label else f"{abs(np.rad2deg(angle_rad)):.1f}°"
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
            'extend_arrows': [extend_arrow1, extend_arrow2],
            'intersection': intersection
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

    # 几何计算辅助方法.

    def _extend_to_intersection(self, 
                            original_line: np.ndarray,
                            signed_dist: float, 
                            intersection: np.ndarray,
                            min_radius: float,
                            style: dict) -> Tuple[np.ndarray, Optional[Annotation]]:
        """
        根据符号距离延长线段
        返回：(延长后的线段, 延长线箭头对象)
        """
        start, end = original_line[0], original_line[1]
        direction = end - start
        
        # 情况1：交点在线段内部
        if signed_dist == 0:
            # 检查终点到交点的剩余长度
            remaining_length = np.linalg.norm(end - intersection)
            if remaining_length < min_radius:
                new_end = end + direction / np.linalg.norm(direction) * (min_radius - remaining_length)
                arrow = self._draw_extension(end, new_end, style)
                return np.array([start, new_end]), arrow
            return original_line, None
        
        # 情况2：需要反向延长（起点之前）
        if signed_dist < 0:
            extend_length = abs(signed_dist) + min_radius
            new_start = start - direction / np.linalg.norm(direction) * extend_length
            arrow = self._draw_extension(start, new_start, style)
            return np.array([new_start, end]), arrow
        
        # 情况3：需要正向延长（终点之后）
        extend_length = signed_dist + min_radius
        new_end = end + direction / np.linalg.norm(direction) * extend_length
        arrow = self._draw_extension(end, new_end, style)
        return np.array([start, new_end]), arrow

    # 修改_extend_to_intersection中的绘图调用
    def _draw_extension(self, start: np.ndarray, end: np.ndarray, style: dict) -> plt.Line2D:
        """绘制延长线（使用基础线段）"""
        return self.add_line(
            tuple(start), tuple(end),
            color=style.get('color', self.style.guideline['color']),
            linewidth=style.get('linewidth', self.style.guideline['linewidth']),
            linestyle=style.get('linestyle', self.style.guideline['linestyle']),
            alpha=style.get('alpha', self.style.guideline['alpha'])
        )

    def _calculate_label_position(self, 
                                center: Tuple[float, float],
                                v1: np.ndarray, 
                                v2: np.ndarray,
                                offset: float) -> Tuple[float, float]:
        """计算标签位置"""
        # 计算夹角平分线方向
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        bisect_angle = (angle1 + angle2) / 2
        
        # 添加垂直于平分线的偏移
        dx = offset * np.cos(bisect_angle)
        dy = offset * np.sin(bisect_angle)
        
        return (center[0] + dx, center[1] + dy)
    
    def calculate_segment_relations(self, edge1: Annotation, edge2: Annotation) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], float]:
        """
        计算两个有向线段的空间关系和角度差
        返回：(交点坐标， (线段1距离, 线段2距离)，旋转角度)
        """
        # 获取线段坐标
        line1 = np.array([edge1.xyann, edge1.xy])
        line2 = np.array([edge2.xyann, edge2.xy])

        # 计算直线交点
        intersection = self._line_intersection(line1, line2)
        
        # 计算距离参数
        dist_params = (None, None)
        if intersection:
            dist_params = (
                self._calculate_signed_distance(line1, intersection),
                self._calculate_signed_distance(line2, intersection)
            )

        # 计算有向角度差
        angle_diff = self._calculate_angle_diff(
            self._get_direction_vector(line1),
            self._get_direction_vector(line2)
        )
        
        return intersection, dist_params, angle_diff

    def _calculate_signed_distance(self, line: np.ndarray, point: Tuple[float, float]) -> float:
        """
        计算点到线段的符号距离
        返回：负值-起点前，0-线段内，正值-终点后
        """
        start, end = line[0], line[1]
        vec_line = end - start
        vec_point = np.array(point) - start
        
        # 计算投影参数t
        t = np.dot(vec_point, vec_line) / np.dot(vec_line, vec_line)
        
        if t < 0:
            # 起点之前的距离（返回负值）
            return -np.linalg.norm(vec_point)
        elif t > 1:
            # 终点之后的距离（返回正值）
            return np.linalg.norm(np.array(point) - end)
        else:
            # 在线段内部
            return 0.0

    def _line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Optional[Tuple[float, float]]:
        """计算无限长直线的交点"""
        # 解线性方程组 [p + t*u = q + s*v]
        p, u = line1[0], line1[1] - line1[0]
        q, v = line2[0], line2[1] - line2[0]
        
        cross_uv = np.cross(u, v)
        if np.isclose(cross_uv, 0):
            return None  # 平行或重合
        
        w = q - p
        t = np.cross(w, v) / cross_uv
        return tuple(p + t * u)

    def _get_direction_vector(self, line: np.ndarray) -> np.ndarray:
        """获取线段方向向量"""
        return line[1] - line[0]

    def _calculate_angle_diff(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算有向角度差"""
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        angle_diff = (angle2 - angle1 + np.pi) % (2 * np.pi) - np.pi
        return angle_diff
    def _calculate_angle(self, v1, v2):
        """计算转向角度（返回弧度值）"""
        # 计算原始角度差并规范化到(-π, π]
        angle_v1 = np.arctan2(v1[1], v1[0])
        angle_v2 = np.arctan2(v2[1], v2[0])
        angle_diff = (angle_v2 - angle_v1 + np.pi) % (2 * np.pi) - np.pi

        start_angle = angle_v1
        end_angle = angle_v2
        is_acute = angle_diff > 0

        # 圆弧的绘制规则为从start_angle逆时针到end_angle
        # 确保圆弧的绘制方向与角度差一致
        if angle_diff < 0:
            start_angle, end_angle = end_angle, start_angle        
        return (start_angle, end_angle), angle_diff, is_acute

    def _get_label_offset(self, v1, v2):
        """计算标签偏移方向"""
        normal = np.array([-v1[1], v1[0]])
        if np.dot(normal, v2) > 0:
            return normal / np.linalg.norm(normal)
        return -normal / np.linalg.norm(normal)