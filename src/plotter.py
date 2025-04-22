from functools import wraps
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Polygon, Rectangle
from matplotlib.text import Annotation


class RRTStyleConfig:
    """RRT算法可视化工具的全局样式配置容器

    Attributes:
        node (dict): 节点绘图样式配置
        arrow (dict): 箭头绘图样式配置
        obstacle (dict): 障碍物绘图样式配置
        annotation (dict): 文字标注样式配置
        angle (dict): 角度标注样式配置
        guideline (dict): 辅助线样式配置

    Example:
        >>> style = RRTStyleConfig()
        >>> style.update_style('node', color='red', radius=0.3)
    """

    def __init__(self):
        """初始化RRT可视化元素的默认样式配置

        Attributes:
            node (dict): 节点样式配置，包含：
                - color: 节点颜色（默认'black'）
                - radius: 节点半径（默认0.2）
                - alpha: 透明度（默认1.0）
                - fontdict: 标签字体配置（默认Times New Roman 14号字）

            arrow (dict): 箭头样式配置，包含：
                - color: 箭头颜色（默认'black'）
                - linewidth: 线宽（默认1.5）
                - arrowstyle: 箭头样式（默认'-|>'）
                - mutation_scale: 箭头大小（默认20）
                - label_offset: 标签偏移量（默认0.2）

            obstacle (dict): 障碍物样式配置，包含：
                - color: 填充颜色（默认'black'）
                - alpha: 透明度（默认1.0）

            annotation (dict): 文字标注样式配置，包含：
                - color: 文字颜色（默认'black'）
                - fontsize: 字号（默认16）
                - fontdict: 字体配置（默认加粗Times New Roman）

            angle (dict): 角度标注样式配置，包含：
                - color: 圆弧颜色（默认'#e67e22'）
                - linewidth: 线宽（默认1.2）
                - arc_radius: 圆弧半径（默认0.5）
                - label_offset: 标签偏移系数（默认1.2）

            guideline (dict): 辅助线样式配置，包含：
                - color: 线条颜色（默认'#999999'）
                - linewidth: 线宽（默认5.0）
                - linestyle: 线型（默认虚线'--'）
                - alpha: 透明度（默认0.6）

        Example:
            >>> style = RRTStyleConfig()
            >>> print(style.node['color'])  # 输出: black
        """

        # 节点默认样式
        self.node = {
            "color": "black",
            "radius": 0.2,
            "alpha": 1.0,
            "fontdict": {"fontsize": 14, "fontfamily": "Times New Roman"},
        }

        # 箭头默认样式
        self.arrow = {
            "color": "black",
            "linewidth": 1.5,
            "arrowstyle": "-|>",
            "mutation_scale": 20,
            "label_offset": 0.2,
            "fontdict": {"fontsize": 14, "fontfamily": "Times New Roman"},
        }

        # 障碍物默认样式
        self.obstacle = {
            "color": "black",
            "alpha": 1,
            "fontdict": {"fontsize": 14, "fontfamily": "Times New Roman"},
        }

        # 标注默认样式
        self.annotation = {
            "color": "black",
            "fontsize": 16,
            "fontdict": {
                "fontweight": "bold",
                "fontfamily": "Times New Roman",
            },
        }

        # 新增角度标注样式配置
        self.angle = {
            "color": "#e67e22",
            "linewidth": 1.2,
            "arc_radius": 0.5,
            "label_offset": 1.2,
            "fontdict": {
                "fontsize": 12,
                "fontweight": "bold",
                "fontfamily": "Times New Roman",
            },
        }

        # 辅助线样式
        self.guideline = {
            "color": "#999999",
            "linewidth": 5.0,
            "linestyle": "--",
            "alpha": 0.6,
        }

    def update_style(self, element_type: str, **kwargs):
        """动态更新指定类型元素的绘图样式

        Args:
            element_type (str): 需要更新的元素类型，可选值：
                'node', 'arrow', 'obstacle', 'annotation', 'angle', 'guideline'
            **kwargs: 需要更新的样式键值对

        Raises:
            ValueError: 当传入无效的元素类型时抛出

        Example:
            >>> style_config = RRTStyleConfig()
            >>> style_config.update_style('node', color='#FF5733', alpha=0.8)
        """
        if hasattr(self, element_type):
            getattr(self, element_type).update(kwargs)
        else:
            raise ValueError(f"未知的元素类型: {element_type}")


def validate_params(valid_keys: list):
    """参数验证装饰器工厂函数

    用于验证被装饰方法的输入参数，自动过滤非白名单参数

    Args:
        valid_keys (list): 允许的参数名列表

    Returns:
        function: 参数验证装饰器

    Example:
        >>> @validate_params(['color', 'linewidth'])
        >>> def draw_line(...):
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
            return func(self, *args, **filtered)

        return wrapper

    return decorator


class RRTPlotter:
    """RRT算法可视化工具核心类

    提供RRT算法各要素的可视化绘制功能

    Attributes:
        fig (Figure): matplotlib图形对象
        ax (Axes): matplotlib坐标轴对象
        style (RRTStyleConfig): 样式配置对象
        _element_counters (dict): 各类型图形元素计数器
    """

    def __init__(
        self,
        figsize: Tuple[float, float] = (10, 8),
        dpi: int = 100,
        boundary_view: bool = True,
        style_config: Optional[RRTStyleConfig] = None,
    ):
        """初始化RRT可视化画布

        Args:
            figsize (Tuple[float, float]): 画布物理尺寸 (宽度, 高度)，单位：英寸，默认(10,8)
            dpi (int): 输出分辨率（每英寸点数），默认100
            boundary_view (bool): 是否显示画布边界框，默认True
            style_config (Optional[RRTStyleConfig]): 自定义样式配置对象，默认使用内置配置

        Attributes:
            fig (Figure): matplotlib图形对象
            ax (Axes): matplotlib坐标系对象
            style (RRTStyleConfig): 实际生效的样式配置
            _element_counters (dict): 各类型元素计数器 {'node': 0, 'edge': 0, ...}

        Example:
            >>> plotter = RRTPlotter(figsize=(12, 10), dpi=150, boundary_view=False)
        """

        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # 初始化样式配置
        self.style = style_config if style_config else RRTStyleConfig()

        # 绘制画布边框
        if boundary_view:
            border = Rectangle(
                (0, 0),
                figsize[0],
                figsize[1],
                edgecolor="black",
                linewidth=1,
                fill=False,
            )
            self.ax.add_patch(border)

        # 设置坐标范围
        self.ax.set_xlim(0, figsize[0] * 1.1)
        self.ax.set_ylim(0, figsize[1] * 1.1)

        # 初始化计数器
        self._element_counters = {
            "node": 0,
            "edge": 0,
            "obstacle": 0,
        }  # 记录节点数、边数和障碍物数

    ##############################################
    # 第一层：基础绘制方法
    ##############################################
    @validate_params(["color", "radius", "alpha"])
    def base_circle(self, center: Tuple[float, float], **kwargs) -> Circle:
        """绘制基础圆形元素（无标签）

        Args:
            center (Tuple[float, float]): 圆心坐标 (x, y)
            **kwargs: 可覆盖的样式参数（color, radius, alpha）

        Returns:
            Circle: 创建的圆形Patch对象

        Example:
            >>> circle = plotter.base_circle((5,5), color='red', radius=0.3)
        """
        params = {**self.style.node, **kwargs}
        # 去掉字体参数
        params.pop("fontdict", None)
        circle = Circle(center, **params)
        self.ax.add_patch(circle)
        return circle

    @validate_params(["color", "linewidth", "arrowstyle"])
    def base_arrow(
        self, start: Tuple[float, float], end: Tuple[float, float], **kwargs
    ) -> Annotation:
        """绘制基础箭头元素（无标签）

        Args:
            start (Tuple[float, float]): 起点坐标 (x, y)
            end (Tuple[float, float]): 终点坐标 (x, y)
            **kwargs: 可覆盖的样式参数（color, linewidth, arrowstyle）

        Returns:
            Annotation: 创建的箭头Annotation对象

        Example:
            >>> arrow = plotter.base_arrow((0,0), (5,5), linewidth=2)
        """
        params = {**self.style.arrow, **kwargs}
        # 去掉字体参数
        params.pop("fontdict", None)
        arrow = self.ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                k: v
                for k, v in params.items()
                if k in ["arrowstyle", "color", "linewidth", "mutation_scale"]
            },
        )
        return arrow

    @validate_params(["color", "alpha"])
    def base_obstacle(self, vertices: np.ndarray, **kwargs) -> Polygon:
        """绘制基础多边形障碍物（无标签）

        Args:
            vertices (np.ndarray): 多边形顶点坐标数组，形状为(N,2)
            **kwargs: 可覆盖的样式参数（color, alpha）

        Returns:
            Polygon: 创建的多边形Patch对象

        Note:
            会自动闭合未闭合的多边形顶点

        Example:
            >>> vertices = np.array([[0,0], [2,0], [1,1]])
            >>> obstacle = plotter.base_obstacle(vertices, color='gray')
        """

        params = {**self.style.obstacle, **kwargs}
        if not np.allclose(vertices[0], vertices[-1]):
            vertices = np.vstack([vertices, vertices[0]])
        # 去掉字体参数
        params.pop("fontdict", None)
        obstacle = Polygon(vertices, closed=True, **params)
        self.ax.add_patch(obstacle)
        return obstacle

    # 在RRTPlotter类中添加基础线绘制方法
    @validate_params(["color", "linewidth", "linestyle", "alpha"])
    def add_line(
        self, start: Tuple[float, float], end: Tuple[float, float], **kwargs
    ) -> plt.Line2D:
        """绘制辅助线段

        Args:
            start (Tuple[float, float]): 起点坐标 (x, y)
            end (Tuple[float, float]): 终点坐标 (x, y)
            **kwargs: 可覆盖的样式参数：
                - color: 线条颜色
                - linewidth: 线宽
                - linestyle: 线型（如'--'表示虚线）
                - alpha: 透明度

        Returns:
            plt.Line2D: 创建的线段对象

        Example:
            >>> line = plotter.add_line((0,0), (5,5), linestyle='--', color='#999')
        """
        params = {**self.style.guideline, **kwargs}
        line = self.ax.add_line(
            plt.Line2D(
                [start[0], end[0]],
                [start[1], end[1]],
                **{
                    k: v
                    for k, v in params.items()
                    if k in ["color", "linewidth", "linestyle", "alpha"]
                },
            )
        )
        return line

    ##############################################
    # 第二层：组合元素（基础元素+标签）
    ##############################################
    def add_circle_node(
        self,
        center: Tuple[float, float],
        label: Optional[str] = None,
        label_visible: bool = False,
        **kwargs,
    ) -> Tuple[Circle, Optional[Annotation]]:
        """添加带自动编号的圆形节点

        Args:
            center (Tuple[float, float]): 圆心坐标 (x, y)
            label (Optional[str]): 自定义标签文本，默认自动编号
            label_visible (bool): 是否显示标签，默认False
            **kwargs: 节点样式参数（继承自base_circle）

        Returns:
            Tuple[Circle, Optional[Annotation]]:
                圆形节点对象和文本标签对象（当label_visible为True时非空）

        Example:
            >>> node, label = plotter.add_circle_node((2,3), label_visible=True)
        """
        # 更新编号
        self._element_counters["node"] += 1
        # 获取自动编号

        if label is None:
            label = str(self._element_counters["node"])

        # 绘制基础圆形
        circle = self.base_circle(center, **kwargs)

        # 添加标签
        text = None
        if label_visible:
            text = self.ax.text(
                center[0],
                center[1] - circle.radius * 2,
                label,
                **self.style.node["fontdict"],
            )
        return circle, text

    def add_edge(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        label: Optional[str] = None,
        label_visible: bool = False,
        **kwargs,
    ) -> Tuple[Annotation, Optional[Annotation]]:
        """添加带自动编号的箭头连接

        Args:
            start (Tuple[float, float]): 起点坐标 (x, y)
            end (Tuple[float, float]): 终点坐标 (x, y)
            label (Optional[str]): 自定义标签文本，默认自动生成序号
            label_visible (bool): 是否显示边标签，默认False
            **kwargs: 箭头样式参数，可覆盖：
                - color: 箭头颜色
                - linewidth: 线宽
                - arrowstyle: 箭头样式
                - mutation_scale: 箭头大小

        Returns:
            Tuple[Annotation, Optional[Annotation]]:
                (箭头对象, 文本标签对象) 元组，当label_visible=False时第二个元素为None

        Example:
            >>> arrow, label = plotter.add_edge((0,0), (5,5),
            >>>                                color='blue',
            >>>                                label_visible=True)
        """
        # 更新编号
        self._element_counters["edge"] += 1
        # 获取自动编号
        if label is None:
            label = str(self._element_counters["edge"])

        # 绘制基础箭头
        arrow = self.base_arrow(start, end, **kwargs)

        # 计算标签位置
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        off_x, off_y = np.array([-dy, dx]) / np.linalg.norm([-dy, dx])

        label_x = start[0] + dx * 0.5 + off_x * self.style.arrow["label_offset"]
        label_y = start[1] + dy * 0.5 + off_y * self.style.arrow["label_offset"]

        # 添加标签
        text = None
        if label_visible:
            text = self.ax.text(label_x, label_y, label, **self.style.arrow["fontdict"])
        return arrow, text

    def add_labeled_obstacle(
        self,
        vertices: np.ndarray,
        label: Optional[str] = None,
        label_visible: bool = False,
        **kwargs,
    ) -> Tuple[Polygon, Optional[Annotation]]:
        """添加带标签编号的障碍物多边形

        Args:
            vertices (np.ndarray): 障碍物顶点坐标数组，形状为(N,2)
            label (Optional[str]): 自定义标签文本，默认自动编号
            label_visible (bool): 是否显示标签，默认False
            **kwargs: 障碍物样式参数（继承自base_obstacle）

        Returns:
            Tuple[Polygon, Optional[Annotation]]:
                障碍物多边形对象和文本标签对象（当label_visible为True时非空）

        Raises:
            ValueError: 当顶点数小于3时可能抛出异常

        Example:
            >>> vertices = np.array([[0,0], [1,0], [1,1]])
            >>> obstacle, label = plotter.add_labeled_obstacle(vertices)
        """
        # 更新障碍物计数器
        self._element_counters["obstacle"] += 1
        # 自动生成标签编号
        if label is None:
            label = f"{self._element_counters['obstacle']}"

        # 绘制基础障碍物
        obstacle = self.base_obstacle(vertices, **kwargs)

        # 添加标签
        text = None
        if label_visible:
            # 计算精确几何中心
            closed_vertices = (
                vertices
                if np.allclose(vertices[0], vertices[-1])
                else np.vstack([vertices, vertices[0]])
            )
            centroid = np.mean(closed_vertices[:-1], axis=0)  # 排除重复的闭合点

            # 获取障碍物实际颜色（处理颜色继承逻辑）
            obstacle_color = obstacle.get_facecolor()[:3]  # 获取RGB值，忽略alpha

            # 计算反色（最大对比度）
            inverse_color = tuple(1 - val for val in obstacle_color)

            text = self.ax.text(
                centroid[0],
                centroid[1],
                label,
                color=inverse_color,
                **{
                    **self.style.obstacle["fontdict"],
                    "ha": "center",  # 水平居中
                    "va": "center",  # 垂直居中
                },
            )
        return obstacle, text

    ##############################################
    # 第三层：高级逻辑方法
    ##############################################
    def connect_nodes(
        self, start_node: Circle, end_node: Circle, **kwargs
    ) -> Tuple[Polygon, Optional[Annotation]]:
        """
        连接两个RRT节点并绘制箭头

        Args:
            start_node (Circle): 起始节点对象
            end_node (Circle): 终止节点对象
            **kwargs: 箭头样式参数

        Returns:
            Tuple[Annotation, Optional[Annotation]]: (箭头对象, 标签文本对象)

        Examples:
            >>> node1 = plotter.add_circle_node((0,0))
            >>> node2 = plotter.add_circle_node((5,5))
            >>> arrow, text = plotter.connect_nodes(node1, node2)
        """
        # 计算连接路径
        angle = np.arctan2(
            end_node.center[1] - start_node.center[1],
            end_node.center[0] - start_node.center[0],
        )

        start = (
            start_node.center[0] + start_node.radius * np.cos(angle),
            start_node.center[1] + start_node.radius * np.sin(angle),
        )
        end = (
            end_node.center[0] - end_node.radius * np.cos(angle),
            end_node.center[1] - end_node.radius * np.sin(angle),
        )

        # 绘制箭头
        arrow, arrow_text = self.add_edge(start, end, **kwargs)

        return arrow, arrow_text

    def add_path_segment(
        self, start: Tuple[float, float], angle: float, length: float, **kwargs
    ) -> Dict[str, Any]:
        """添加带末端节点的完整路径段

        Args:
            start (Tuple[float, float]): 路径起点坐标 (x, y)
            angle (float): 扩展方向角度（单位：弧度）
            length (float): 路径总长度（自动扣除末端节点半径）
            **kwargs: 路径样式参数，包括：
                - radius: 末端节点半径（默认使用全局配置）
                - 其他箭头/节点样式参数

        Returns:
            Dict[str, Any]: 包含以下键的字典：
                - 'arrow': 路径箭头对象
                - 'arrow_text': 箭头标签对象
                - 'node': 末端节点对象
                - 'node_text': 节点标签对象

        Note:
            实际路径长度 = length - 末端节点半径

        Example:
            >>> path = plotter.add_path_segment((2,3), np.pi/4, 5,
            >>>                                 radius=0.3,
            >>>                                 color='green')
        """
        # 计算终点
        # 实际路径长度应减去圆节点的半径
        # 如果没有传入半径，则获取全局配置半径
        radius = kwargs.get("radius", self.style.node["radius"])
        segment_end = (
            start[0] + length * np.cos(angle),
            start[1] + length * np.sin(angle),
        )
        length = length - radius
        end_circle_center = (
            segment_end[0] + radius * np.cos(angle),
            segment_end[1] + radius * np.sin(angle),
        )

        # 绘制箭头
        arrow, arrow_text = self.add_edge(start, segment_end, **kwargs)

        # 绘制末端节点
        node, node_text = self.add_circle_node(end_circle_center, **kwargs)

        return {
            "arrow": arrow,
            "arrow_text": arrow_text,
            "node": node,
            "node_text": node_text,
        }

    def radial_extension(
        self, base_node: Circle, direction_angle: float, length: float, **kwargs
    ) -> Annotation:
        """从现有节点沿指定角度方向扩展路径段

        Args:
            base_node (Circle): 基准节点对象
            direction_angle (float): 扩展方向角度（单位：度）
            length (float): 扩展总长度（需大于节点半径）
            **kwargs: 路径段样式参数

        Returns:
            Annotation: 包含路径段元素的字典

        Raises:
            ValueError: 当扩展长度小于节点半径时抛出

        Example:
            >>> node = plotter.add_circle_node((2,3))
            >>> path = plotter.radial_extension(node, 45, 5)
        """
        # 计算起点
        start = (
            base_node.center[0] + base_node.radius * np.cos(direction_angle),
            base_node.center[1] + base_node.radius * np.sin(direction_angle),
        )
        # 修正长度，长度减去起点端圆的半径
        length = length - base_node.radius

        # 添加路径段
        return self.add_path_segment(
            start, np.radians(direction_angle), length - base_node.radius, **kwargs
        )

    def inward_connection(
        self, target_node: Circle, direction_angle: float, length: float, **kwargs
    ) -> Annotation:
        """创建指向目标节点的外部连接边

        Args:
            target_node (Circle): 目标节点对象
            direction_angle (float): 连接方向角度（单位：度）
            length (float): 连接边总长度（需大于节点半径）
            **kwargs: 箭头样式参数

        Returns:
            Dict[str, Any]: 包含箭头要素的字典，包含以下键：
                - arrow: 箭头对象
                - arrow_text: 标签对象
                - start_point: 起点坐标
                - end_point: 终点坐标

        Raises:
            ValueError: 当长度小于节点半径时抛出

        Example:
            >>> target = plotter.add_circle_node((5,5))
            >>> conn = plotter.inward_connection(target, 30, 6)
        """
        # 计算起点位置（目标节点外延反方向）
        radius = target_node.radius
        angle_rad = np.radians(direction_angle)

        if length < radius:
            raise ValueError("Length must be greater than the radius.")

        # 计算箭头终点（目标节点边缘）
        end_point = (
            target_node.center[0] - radius * np.cos(angle_rad),
            target_node.center[1] - radius * np.sin(angle_rad),
        )

        # 计算起点位置（考虑长度和半径偏移）
        start_point = (
            end_point[0] - (length - radius) * np.cos(angle_rad),
            end_point[1] - (length - radius) * np.sin(angle_rad),
        )
        # 绘制箭头
        arrow, arrow_text = self.add_edge(start_point, end_point, **kwargs)

        return {
            "arrow": arrow,
            "arrow_text": arrow_text,
            "start_point": start_point,
            "end_point": end_point,
        }

    def add_angle_annotation(
        self,
        edge1: Annotation,
        edge2: Annotation,
        label: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """标注两有向边之间的夹角

        Args:
            edge1 (Annotation): 第一条边的Annotation对象
            edge2 (Annotation): 第二条边的Annotation对象
            label (Optional[str]): 自定义标签文本，默认显示角度值
            **kwargs: 角度标注样式参数

        Returns:
            Dict[str, Any]: 包含以下元素的字典：
                - arc: 圆弧对象
                - text: 标签文本对象
                - extend_arrows: 延长线对象列表
                - intersection: 交点坐标

        Note:
            当两直线平行或重合时返回None

        Example:
            >>> edge1 = plotter.add_edge((0,0), (5,0))
            >>> edge2 = plotter.add_edge((0,0), (0,5))
            >>> angle_anno = plotter.add_angle_annotation(edge1, edge2)
        """
        # 合并样式参数
        style_params = {**self.style.angle, **kwargs}

        # 获取线段空间关系
        intersection, (dist1, dist2), angle_diff = self.calculate_segment_relations(
            edge1, edge2
        )

        # 情况1：直线无交点（平行或重合）
        if intersection is None:
            return None

        # 情况2：需要处理线段延长
        line1_orig = np.array([edge1.xyann, edge1.xy])
        line2_orig = np.array([edge2.xyann, edge2.xy])

        # 处理第一条线段的延长
        line1_ext, extend_arrow1 = self._extend_to_intersection(
            line1_orig,
            dist1,
            intersection,
            style_params["arc_radius"] * 1.5,
            style_params,
        )

        # 处理第二条线段的延长
        line2_ext, extend_arrow2 = self._extend_to_intersection(
            line2_orig,
            dist2,
            intersection,
            style_params["arc_radius"] * 1.5,
            style_params,
        )

        # 获取延长后的向量
        v1 = line1_ext[1] - line1_ext[0]
        v2 = line2_ext[1] - line2_ext[0]

        # 计算标注角度参数
        (start_angle, end_angle), angle_rad, is_acute = self._calculate_angle(v1, v2)

        # 确定标注位置
        arc_radius = style_params["arc_radius"]
        label_pos = self._calculate_label_position(
            intersection, v1, v2, arc_radius * style_params["label_offset"]
        )

        # 绘制圆弧
        arc = Arc(
            intersection,
            2 * arc_radius,
            2 * arc_radius,
            theta1=np.degrees(start_angle),
            theta2=np.degrees(end_angle),
            color=style_params["color"],
            linewidth=style_params["linewidth"],
        )
        self.ax.add_patch(arc)

        # 添加标注文本
        label_text = label if label else f"{abs(np.rad2deg(angle_rad)):.1f}°"
        text = self.ax.annotate(
            label_text,
            xy=label_pos,
            horizontalalignment="center",
            verticalalignment="center",
            **style_params["fontdict"],
        )

        return {
            "arc": arc,
            "text": text,
            "extend_arrows": [extend_arrow1, extend_arrow2],
            "intersection": intersection,
        }

    ##############################################
    # 第四层：更高级的方法
    ##############################################
    @validate_params(
        [
            " tree_type",
            "start_angle",
            "start_length",
            "node_color",
            "edge_color",
            "root_radius",
            "node_label",
            "edge_label",
        ]
    )
    def draw_rrt_tree(
        self,
        tree: Any,
        tree_type: str = "classic",
        start_angle: float = None,
        start_length: float = 1,
        node_label: bool = False,  # 节点标签控制
        edge_label: bool = False,  # 边标签控制
        **kwargs,
    ) -> Dict[str, Any]:
        """自动绘制RRT树结构

        Args:
            tree (Any): 树结构数据，类型取决于tree_type参数
            tree_type (str): 树结构类型，可选：
                'classic' - 经典树结构（需包含parent/position属性）
                'networkx' - NetworkX的Graph对象（待调试与测试）
                'ompl' - OMPL库的RRT树结构（待调试与测试）
            start_angle (float): 根节点起始角度（弧度），默认None
            start_length (float): 根节点起始长度，默认1
            node_label (bool): 是否显示节点编号，默认False
            edge_label (bool): 是否显示边编号，默认False
            **kwargs: 样式覆盖参数（node_color, edge_color等）

        Returns:
            Dict[str, Any]: 包含所有绘图元素的字典，键为'nodes'和'edges'

        Raises:
            ValueError: 当传入不支持的tree_type时抛出

        Example:
            >>> elements = plotter.draw_rrt_tree(tree, node_label=True)
        """
        # 样式参数合并
        params = {
            "node_color": self.style.node["color"],
            "edge_color": self.style.arrow["color"],
            "root_radius": self.style.node["radius"] * 1.5,
            "show_labels": False,
            "node_label": node_label,  # 参数传递
            "edge_label": edge_label,  # 参数传递
            **kwargs,
        }

        elements = {"nodes": [], "edges": []}

        # 分类型处理
        if tree_type == "classic":
            self._draw_classic_rrt(tree, params, elements, start_angle, start_length)
        elif tree_type == "networkx":
            self._draw_networkx_rrt(tree, params, elements)
        elif tree_type == "ompl":
            self._draw_ompl_rrt(tree, params, elements)
        else:
            raise ValueError(f"不支持的树结构类型: {tree_type}")

        return elements

    def _draw_classic_rrt(self, root_node, params, elements, start_angle, start_length):
        """经典RRT树结构绘制内部实现

        Args:
            root_node: 根节点对象，需包含position和children属性
            params (dict): 渲染参数配置
            elements (dict): 元素存储字典
            start_angle (float): 初始角度（弧度）
            start_length (float): 初始长度

        Implementation:
            使用广度优先搜索遍历树结构，自动处理：
            - 节点半径差异（根节点特殊样式）
            - 边标签可见性
            - 根节点方向连接
        """
        from collections import deque

        queue = deque([root_node])

        while queue:
            node = queue.popleft()

            # 绘制当前节点
            node_style = {
                "color": params["node_color"],
                "radius": (
                    params["root_radius"]
                    if node.parent is None
                    else self.style.node["radius"]
                ),
            }
            circle, _ = self.add_circle_node(
                tuple(node.position), label_visible=params["node_label"], **node_style
            )
            elements["nodes"].append(circle)
            # 记录当前节点的元素
            node.element = circle

            # 绘制到父节点的边
            if node.parent is not None:
                edge, _ = self.connect_nodes(
                    node.parent.element,
                    node.element,
                    label_visible=params["edge_label"],
                )
                elements["edges"].append(edge)
                node.edge_element = edge
            elif start_angle is not None:
                # 根节点存在接入方向
                dic = self.inward_connection(
                    node.element,
                    start_angle,
                    start_length,
                    label_visible=params["edge_label"],
                )
                elements["edges"].append(dic["arrow"])
                node.edge_element = dic["arrow"]

            # 添加子节点到队列
            queue.extend(node.children)

    def _draw_networkx_rrt(self, G, params, elements):
        """处理NetworkX图结构（实验性功能）

        Args:
            G (networkx.Graph): NetworkX图对象，节点需包含'position'属性
            params (dict): 绘图参数配置
            elements (dict): 存储绘图元素的字典

        Note:
            - 当前实现为初步版本，需进一步测试验证
            - 要求图中节点必须包含'position'属性
            - 暂不支持动态更新图结构

        Warning:
             此功能尚未经过测试验证
        """
        import networkx as nx

        for edge in nx.dfs_edges(G):
            # 绘制边
            start_pos = G.nodes[edge[0]]["position"]
            end_pos = G.nodes[edge[1]]["position"]
            edge_obj, _ = self.add_edge(
                tuple(start_pos), tuple(end_pos), color=params["edge_color"]
            )
            elements["edges"].append(edge_obj)

        for node in G.nodes:
            # 绘制节点
            is_root = G.in_degree(node) == 0
            radius = params["root_radius"] if is_root else self.style.node["radius"]
            circle, _ = self.add_circle_node(
                tuple(G.nodes[node]["position"]),
                label_visible=params["show_labels"],
                color=params["node_color"],
                radius=radius,
            )
            elements["nodes"].append(circle)

    def _draw_ompl_rrt(self, ss, params, elements):
        """处理OMPL状态空间树结构（实验性功能）

        Args:
            ss (ompl.base.StateSpace): OMPL状态空间对象
            params (dict): 绘图参数配置
            elements (dict): 存储绘图元素的字典

        Raises:
            RuntimeError: 未安装OMPL库时抛出
            NotImplementedError: 遇到不支持的状态空间类型时抛出

        Note:
            - 需要预先安装OMPL库(pip install ompl)
            - 当前仅支持RealVectorStateSpace类型
            - 状态空间维度需与绘图空间维度匹配

        Warning:
            此功能尚未经过完整测试验证
        """
        try:
            from ompl import base as ob
        except ImportError:
            raise RuntimeError("需要安装OMPL库")

        def add_state(state):
            # OMPL状态转换为坐标
            if isinstance(ss, ob.RealVectorStateSpace):
                return [state[i] for i in range(ss.getDimension())]
            # 其他状态空间类型可在此扩展
            raise NotImplementedError("暂不支持该状态空间类型")

        # 遍历状态空间中的状态
        for state in ss.getStates():
            parent = state.getParent()
            if parent is not None:
                # 绘制边
                start_pos = add_state(parent)
                end_pos = add_state(state)
                edge, _ = self.add_edge(
                    tuple(start_pos), tuple(end_pos), color=params["edge_color"]
                )
                elements["edges"].append(edge)

            # 绘制节点
            is_root = state.getParent() is None
            radius = params["root_radius"] if is_root else self.style.node["radius"]
            circle, _ = self.add_circle_node(
                tuple(add_state(state)),
                label_visible=params["show_labels"],
                color=params["node_color"],
                radius=radius,
            )
            elements["nodes"].append(circle)

    ##############################################
    # 工具方法
    ##############################################
    def add_annotation(
        self, position: Tuple[float, float], text: str, **kwargs
    ) -> Annotation:
        """添加自由文本标注

        Args:
            position (Tuple[float, float]): 文本锚点坐标 (x, y)
            text (str): 标注文本内容
            **kwargs: 文本样式参数，可覆盖：
                - color: 文本颜色
                - fontsize: 字号
                - fontweight: 字重（如'bold'）
                - ha/va: 水平/垂直对齐方式

        Returns:
            Annotation: 创建的文本标注对象

        Example:
            >>> anno = plotter.add_annotation((5,5), "重要节点",
            >>>                              fontsize=18,
            >>>                              color='red',
            >>>                              ha='center')
        """
        params = {**self.style.annotation, **kwargs}
        return self.ax.text(
            position[0],
            position[1],
            text,
            color=params["color"],
            fontsize=params["fontsize"],
            **params["fontdict"],
        )

    def show(self):
        """显示可视化窗口

        Note:
            阻塞式方法，会暂停程序执行直到关闭窗口
            建议在所有绘图操作完成后调用
        """

        plt.show()

    def save(self, filename: str, dpi: int = 300):
        """保存当前可视化结果到文件

        Args:
            filename (str): 保存路径（支持格式：png/jpg/pdf/svg等）
            dpi (int): 输出分辨率（每英寸点数），默认300

        Example:
            >>> plotter.save('rrt_tree.png', dpi=600)
        """

        self.fig.savefig(filename, bbox_inches="tight", dpi=dpi)

    # 几何计算辅助方法.
    def calculate_segment_relations(
        self, edge1: Annotation, edge2: Annotation
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], float]:
        """计算线段空间关系（几何核心方法）

        Algorithm:
            1. 计算无限长直线交点
            2. 计算符号距离判断相对位置
            3. 计算有向角度差

        Returns:
            tuple: 包含三个元素的元组：
                0: 交点坐标 (x,y) 或 None
                1: (到线段1的符号距离, 到线段2的符号距离)
                2: 从线段1到线段2的旋转角度（弧度）

        Note:
            符号距离说明：负值表示在起点延长线方向，正值表示在终点延长线方向
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
                self._calculate_signed_distance(line2, intersection),
            )

        # 计算有向角度差
        angle_diff = self._calculate_angle_diff(
            self._get_direction_vector(line1), self._get_direction_vector(line2)
        )

        return intersection, dist_params, angle_diff

    def _extend_to_intersection(
        self,
        original_line: np.ndarray,
        signed_dist: float,
        intersection: np.ndarray,
        min_radius: float,
        style: dict,
    ) -> Tuple[np.ndarray, Optional[Annotation]]:
        """【内部方法】根据符号距离延长线段到交点

        Args:
            original_line (np.ndarray): 原始线段坐标数组 [起点, 终点]
            signed_dist (float): 符号距离（负数表示需要反向延长）
            intersection (np.ndarray): 理论交点坐标
            min_radius (float): 最小延长半径阈值
            style (dict): 延长线样式配置

        Returns:
            Tuple[np.ndarray, Optional[Annotation]]:
                (延长后的线段坐标数组, 延长线图形对象)

        Algorithm:
            1. 根据符号距离判断延长方向
            2. 计算需要延长的长度
            3. 绘制延长线并返回新线段
        """
        start, end = original_line[0], original_line[1]
        direction = end - start

        # 情况1：交点在线段内部
        if signed_dist == 0:
            # 检查终点到交点的剩余长度
            remaining_length = np.linalg.norm(end - intersection)
            if remaining_length < min_radius:
                new_end = end + direction / np.linalg.norm(direction) * (
                    min_radius - remaining_length
                )
                arrow = self._draw_extension(end, new_end, style)
                return np.array([start, new_end]), arrow
            return original_line, None

        # 情况2：需要反向延长（起点之前）
        if signed_dist < 0:
            extend_length = abs(signed_dist)
            new_start = start - direction / np.linalg.norm(direction) * extend_length
            arrow = self._draw_extension(start, new_start, style)
            return np.array([new_start, end]), arrow

        # 情况3：需要正向延长（终点之后）
        extend_length = signed_dist + min_radius
        new_end = end + direction / np.linalg.norm(direction) * extend_length
        arrow = self._draw_extension(end, new_end, style)
        return np.array([start, new_end]), arrow

    # 修改_extend_to_intersection中的绘图调用
    def _draw_extension(
        self, start: np.ndarray, end: np.ndarray, style: dict
    ) -> plt.Line2D:
        """绘制辅助线段

        Args:
            start (Tuple[float, float]): 起点坐标 (x, y)
            end (Tuple[float, float]): 终点坐标 (x, y)
            **kwargs: 线段样式参数（color, linewidth, linestyle, alpha）

        Returns:
            plt.Line2D: 创建的线段对象

        Example:
            >>> line = plotter.add_line((0,0), (5,5), linestyle='--')
        """
        return self.add_line(
            tuple(start),
            tuple(end),
            color=style.get("color", self.style.guideline["color"]),
            linewidth=style.get("linewidth", self.style.guideline["linewidth"]),
            linestyle=style.get("linestyle", self.style.guideline["linestyle"]),
            alpha=style.get("alpha", self.style.guideline["alpha"]),
        )

    def _calculate_label_position(
        self, center: Tuple[float, float], v1: np.ndarray, v2: np.ndarray, offset: float
    ) -> Tuple[float, float]:
        """【内部方法】计算角度标注标签位置

        Args:
            center (Tuple[float, float]): 夹角顶点坐标
            v1 (np.ndarray): 第一条边方向向量
            v2 (np.ndarray): 第二条边方向向量
            offset (float): 偏移距离系数

        Returns:
            Tuple[float, float]: 标签位置坐标 (x, y)

        Math:
            标签位置 = 夹角平分线方向 * offset + 顶点坐标
        """
        # 计算夹角平分线方向
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        bisect_angle = (angle1 + angle2) / 2

        # 添加垂直于平分线的偏移
        dx = offset * np.cos(bisect_angle)
        dy = offset * np.sin(bisect_angle)

        return (center[0] + dx, center[1] + dy)

    def _calculate_signed_distance(
        self, line: np.ndarray, point: Tuple[float, float]
    ) -> float:
        """【内部方法】计算点到线段的符号距离

        Args:
            line (np.ndarray): 线段坐标数组 [[x0,y0], [x1,y1]]
            point (Tuple[float, float]): 目标点坐标

        Returns:
            float: 符号距离值：
                - 负值：点在起点延长线方向
                - 0：点在线段内部
                - 正值：点在终点延长线方向
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

    def _line_intersection(
        self, line1: np.ndarray, line2: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """直线交点计算（几何底层方法）

        Math:
            使用线性代数方法求解两直线交点：
            设直线1参数方程：p + t*u
            直线2参数方程：q + s*v
            解方程组得到t值

        Returns:
            Optional[Tuple]: 交点坐标或None（当直线平行时）

        Reference:
            https://stackoverflow.com/a/565282
        """
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
        """【内部方法】获取线段方向向量

        Args:
            line (np.ndarray): 线段坐标数组，形状为(2,2)的numpy数组
                [[x_start, y_start], [x_end, y_end]]

        Returns:
            np.ndarray: 方向向量 [dx, dy]

        Example:
            >>> line = np.array([[0,0], [2,3]])
            >>> vec = plotter._get_direction_vector(line)  # 返回 [2,3]
        """
        return line[1] - line[0]

    def _calculate_angle_diff(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """【内部方法】计算有向角度差（弧度）

        Args:
            v1 (np.ndarray): 第一个方向向量 [dx1, dy1]
            v2 (np.ndarray): 第二个方向向量 [dx2, dy2]

        Returns:
            float: 从v1到v2的旋转角度（弧度），范围(-π, π]
                - 正值表示逆时针旋转角度
                - 负值表示顺时针旋转角度

        Math:
            计算公式：
            angle_diff = (angle2 - angle1 + π) % (2π) - π
            其中 angle1 = arctan2(v1.y, v1.x)
                angle2 = arctan2(v2.y, v2.x)

        Example:
            >>> v1 = np.array([1, 0])  # 0度方向
            >>> v2 = np.array([0, 1])  # 90度方向
            >>> diff = plotter._calculate_angle_diff(v1, v2)  # 返回 π/2
        """
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        angle_diff = (angle2 - angle1 + np.pi) % (2 * np.pi) - np.pi
        return angle_diff

    def _calculate_angle(self, v1, v2):
        """【内部方法】计算转向角度参数

        Args:
            v1 (np.ndarray): 基准方向向量 [dx1, dy1]
            v2 (np.ndarray): 目标方向向量 [dx2, dy2]

        Returns:
            tuple: 包含三个元素的元组：
                (start_angle, end_angle): 圆弧起止角度（弧度）
                angle_diff: 实际角度差（弧度）
                is_acute: 是否为锐角（<90度）

        Note:
            返回值用于绘制角度标注圆弧，保证圆弧总是绘制较小的夹角

        Example:
            >>> v1 = np.array([1, 0])
            >>> v2 = np.array([0, 1])
            >>> angles, diff, acute = plotter._calculate_angle(v1, v2)
            # 返回 (0.0, 1.5708), 1.5708, True
        """
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
        """【内部方法】计算标签偏移方向向量

        Args:
            v1 (np.ndarray): 第一条边方向向量
            v2 (np.ndarray): 第二条边方向向量

        Returns:
            np.ndarray: 单位化的垂直偏移方向向量

        """
        normal = np.array([-v1[1], v1[0]])
        if np.dot(normal, v2) > 0:
            return normal / np.linalg.norm(normal)
        return -normal / np.linalg.norm(normal)
