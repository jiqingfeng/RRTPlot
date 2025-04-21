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