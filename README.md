# MultiTurnChartCoder

## 项目说明
本项目主要实现了以下内容：

1. 复现 ChartMimic Benchmark
   - 实现数据集的读取和评估（仅包含 Low-level 评估，因为 High-level 评估需要调用 GPT-4 API，成本较高）
   - 基于 vLLM 本地部署 Qwen2.5-VL 模型进行测评
   - 结果显示 Qwen2.5-VL 全面超越其他开源模型

2. 复现 MatPlotAgent 方案
   - 使用两个 VLM 模型（均采用 Qwen2.5-VL）
   - 一个模型负责代码生成，另一个提供反馈
   - 通过多轮迭代优化图表代码
   - 实验效果：相比 Qwen2.5-VL Baseline 有所下降

3. 复现 MageBench 方案
   - 使用单个 VLM 模型
   - 将执行后的代码结果反馈给模型
   - 通过多轮迭代优化图表代码
   - 实验效果：相比 Qwen2.5-VL Baseline 有明显提升

cite:
- [1] Yang, Cheng, et al. "Chartmimic: Evaluating lmm's cross-modal reasoning capability via chart-to-code generation." arXiv preprint arXiv:2406.09961 (2024).
- [2] Zhao, Xuanle, et al. "ChartCoder: Advancing Multimodal Large Language Model for Chart-to-Code Generation." arXiv preprint arXiv:2501.06598 (2025).
- [3] Yang, Zhiyu, et al. "Matplotagent: Method and evaluation for llm-based agentic scientific data visualization." arXiv preprint arXiv:2402.11453 (2024).
- [4] Zhang, Miaosen, et al. "MageBench: Bridging Large Multimodal Models to Agents." arXiv preprint arXiv:2412.04531 (2024).


## 实验结果
![实验结果](./当前结果.png)

## 代码文件说明
- `vllm_start.sh`: 启动 vLLM 服务，用于本地部署大模型（可视为 Ollama 的替代品）
- `chart_coder_agent.py`: MatPlotAgent 方案实现
- `chart_coder_agent_multi_process.py`: MatPlotAgent 方案的多进程版本
- `chart_coder_agent_single_vlm.py`: MageBench 方案实现
- `chart_coder_agent_single_vlm_multi_process.py`: MageBench 方案的多进程版本
- `chartmimic_benchmark.py`: ChartMimic 数据集的读取和评估

## 快速开始（并没有封装好，代码里面需要手动修改路径之类的）
1. 启动 vLLM 服务：
```bash
bash vllm_start.sh
```

2. 运行推理：
```bash
python chart_coder_agent_multi_process.py   # 在里面选择 MatPlotAgent 或者 MageBench 方案
```

3. 运行评估：
```bash
python chartmimic_benchmark.py
```

## 还可以补充的实验
- [ ] Qwen2.5-VL 的 Multi-turn 问答效果可能不是最好的，尝试让每次迭代解耦，即每次迭代输入上一轮生成的代码、对应执行结果、和要求的 GT 图像，然后生成新的代码，这样可以使用 Qwen2.5-VL 的单轮问答能力，而不是 Multi-turn 问答能力
- [ ] 当前 MatplotAgent 和 MageBench 方案，我只测试了迭代一轮的情况，可以补充迭代多轮的情况
- [ ] 分析为什么 MatplotAgent 相比于 Baseline 效果下降