# 备注
核心代码：
- vllm_start.sh 启动 vllm 服务，在跑 LangGraph 的系统前，需要启动 vllm 服务，也就是本地部署的大模型。可以看作是 Ollama 的替代品。
- chart_coder_agent.py 是核心代码，定义了多智能体系统中的各个智能体，包括代码生成智能体、视觉智能体和代码执行智能体。


# MultiTurnChartCoder (下面是 Cursor 自动生成的，可以大致看看)

基于 LangGraph 的多智能体数据图表代码生成系统。该系统通过多轮交互的方式，结合视觉语言模型（VLM）和大语言模型（LLM）来生成高质量的数据可视化代码。

## 系统架构

系统由以下核心组件构成：

1. **Code Agent (VLM)**
   - 使用 Qwen2.5-VL-32B-Instruct 作为视觉语言模型
   - 负责根据用户输入和反馈生成图表代码
   - 支持多模态输入（文本描述和参考图像）

2. **Visual Agent (LLM)**
   - 分析生成的代码和执行结果
   - 提供改进建议和反馈
   - 确保图表符合最佳实践

3. **代码执行环境**
   - 安全的代码执行沙箱
   - 实时图表生成和验证
   - 错误捕获和处理

## 项目结构

```
.
├── chart_coder_agent.py    # 核心实现代码
├── utils/                  # 工具函数
├── figs/                   # 生成的图表存储
├── requirements.txt        # 项目依赖
└── vllm_start.sh          # VLLM 服务启动脚本
```

## 环境要求

- Python 3.8+
- CUDA 支持（用于 VLM 推理）
- 足够的 GPU 内存（建议 32GB+）

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/MultiTurnChartCoder.git
cd MultiTurnChartCoder
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 配置

1. 创建 `.env` 文件：
```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=http://your_api_base_url
MAX_ITERATIONS=3
TEMPERATURE=0
```

2. 启动 VLLM 服务：
```bash
bash vllm_start.sh
```

## 使用示例

```python
from chart_coder_agent import generate_chart_code

# 基本使用
result = generate_chart_code(
    prompt="生成一个散点图，展示x和y两个变量之间的关系，并添加趋势线"
)

# 使用参考图像
result = generate_chart_code(
    prompt="根据这个图片生成类似的图表",
    reference_image="path/to/image.png"
)

# 获取结果
print(result["code"])  # 生成的代码
print(result["feedback"])  # 反馈历史
```

## 开发计划

- [ ] 支持更多图表类型
- [ ] 添加代码质量评估
- [ ] 优化多轮对话策略
- [ ] 增加更多数据源支持
- [ ] 改进错误处理机制

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 备注

<!-- 在这里添加您的个人备注 -->



