





### 环境配置
``` python
# 环境配置
# 安装项目所需库（示例）
pip install matplotlib numpy shapely pyyaml
#将依赖列表导出到requirements.txt
pip freeze > requirements.txt
# 其他机器恢复环境时只需：
pip install -r requirements.txt
```

### 进入虚拟环境
``` bash
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process # 临时允许执行脚本
.\venv\Scripts\activate # Windows激活
source venv/bin/activate # Linux激活
```