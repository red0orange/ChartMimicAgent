# 数据图表代码生成 Agent

我想实现一个基于 LangGraph 的 Multi-agent 的系统去完成数据图表代码生成任务，输入是数据图表和 prompt，输出是画出图表的 python 代码。这个系统主要包括一个代码生成的 VLM 和一个负责提供 feedback 的 LLM，然后 multi-turn 地循环直到限定次数或满足条件。

我使用的 LLM 是满足 OpenAI API 标准的 Qwen2.5-VL-32B-Instruct，如下，但不支持 function calling。
```
    llm = ChatOpenAI(
        api_key="xxx",  # guiji
        model="Qwen/Qwen2.5-VL-32B-Instruct",

        base_url="http://35.220.164.252:3888/v1",
        temperature=0,
        max_tokens=None,
        timeout=None,
        # max_retries=2,
        # organization="...",
        # other params...
    )
```

初步的设计思路如下：
- Code Agent (VLM): 使用 LangChain 的多模态 LLM 接口（如 ChatOpenAI with gpt-4-vision-preview 或其他 VLM 封装）作为 Code Agent。它的输入是用户查询和上一轮的反馈（如果有）。
- Visual Agent (LLM): 使用一个标准的 LLM（如 ChatOpenAI with gpt-4）作为 Visual Agent。它的任务是接收 Code Agent 生成的代码。
- Code Execution Tool: 创建一个 LangChain Tool，它使用 Python REPL (或更安全的沙箱环境) 来执行代码，并捕获生成的图表（可以保存为文件或 base64 编码）和任何错误。
- Visual Agent Logic: Visual Agent 调用 Code Execution Tool。如果成功生成图表，它可以（理想情况下，如果它也是 VLM）"查看"图表并结合 "Principles" 给出反馈。如果它只是 LLM，它可以分析代码结构、参数，或依赖于图表的文本描述（如果能生成）来提供反馈。
- LangGraph State: 定义一个状态对象，包含当前代码、生成的图表路径/数据、反馈历史、迭代次数等。
- Loop: 使用 LangGraph 定义节点（Code Generation, Code Execution & Visual Analysis, Feedback Generation）和边，实现循环，直到满足退出条件（如最大迭代次数、Visual Agent 判断满意）。

## 项目结构

- `chart_coder_agent.py`: 主要的代码生成系统实现
- `test_chart_coder.py`: 测试脚本
- `requirements.txt`: 项目依赖

## 系统架构

该系统基于LangGraph实现多智能体交互：

1. **Code Agent (VLM)**: 使用多模态大语言模型生成图表代码
2. **Visual Agent (LLM)**: 分析代码和执行结果，提供反馈
3. **代码执行工具**: 执行生成的代码，捕获图表输出

系统工作流程：
```
用户输入 → Code Agent生成代码 → 代码执行 → Visual Agent提供反馈 → 迭代改进 → 最终代码输出
```

## 安装与使用

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置API

1. 创建`.env`文件，包含以下内容：
```
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=http://35.220.164.252:3888/v1
MAX_ITERATIONS=3  # 最大迭代次数
TEMPERATURE=0     # 生成的随机性
```

### 命令行使用

系统提供了简单的命令行界面：

```bash
# 基本用法
python chart_coder_cli.py "生成一个散点图，展示x和y两个变量之间的关系，并添加趋势线"

# 使用参考图像
python chart_coder_cli.py "根据这个图片生成类似的图表" --image reference.png

# 显示反馈历史
python chart_coder_cli.py "生成一个柱状图" --show-feedback

# 指定输出路径和保存代码
python chart_coder_cli.py "生成一个折线图" --output myfigure.png --save-code chart_code.py

# 交互式模式
python chart_coder_cli.py --interactive
```

### 运行测试

```bash
python test_chart_coder.py
```

### 在代码中使用

```python
from chart_coder_agent import generate_chart_code

# 生成简单图表代码
result = generate_chart_code("生成一个散点图，展示x和y两个变量之间的关系，并添加趋势线")

# 查看生成的代码
print(result["code"])

# 获取生成的图表（如果成功）
if result.get("image_data"):
    with open("chart.png", "wb") as f:
        f.write(base64.b64decode(result["image_data"]))

# 查看反馈历史
print(result["feedback"])
```

## 扩展与改进

- 支持更多图表类型
- 添加更多评估指标
- 支持更复杂的数据输入
- 优化代码执行安全性



