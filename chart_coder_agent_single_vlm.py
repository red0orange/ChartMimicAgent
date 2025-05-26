"""
数据图表代码生成单智能体系统

该系统使用LangGraph实现，包含以下组件：
- Code Agent (VLM): 负责生成图表代码并根据执行结果优化代码
- Code Execution Tool: 执行生成的代码并捕获结果
"""

import os
import base64
from typing import Annotated, Optional, List, Dict, Any
from typing_extensions import TypedDict

import tempfile
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
matplotlib.use('Agg')  # 设置非交互式后端
from pdf2image import convert_from_path
import io

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from chartmimic_benchmark import ChartMimicBenchmark
from utils.my_0_response_processor import process_llm_response, execute_llm_code

cur_file_dir = os.path.dirname(os.path.abspath(__file__))

# 加载环境变量(如果存在.env文件)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("提示: 安装 python-dotenv 可以从.env文件加载配置")

# 配置环境变量（优先使用.env中的配置，否则使用默认值）
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "2"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0"))


class ChartCoderState(TypedDict):
    """图表代码生成系统的状态"""
    user_input: str
    image_data_base64: str
    gen_image_data_base64: str
    code: Optional[str]
    iterations: int
    max_iterations: int
    execution_error: Optional[str]
    messages: Annotated[list, add_messages]  # 使用LangGraph的消息历史管理

def image_path_to_base64(image_path: str) -> str:
    """将图片文件转换为base64编码的字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        raise Exception(f"转换图片失败: {str(e)}")

def image_to_base64(image) -> str:
    """将PIL图像对象转换为base64编码的字符串"""
    buf = io.BytesIO()
    image.save(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_str

def base64_to_plt_image(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return img

# 代码执行工具
def execute_code(code: str) -> Dict[str, Any]:
    """执行Python代码并捕获生成的图表"""
    return_dict = {}

    output_dir = os.path.join(cur_file_dir, "figs")
    success, output_pdf_file, output_png_file, err_msg = execute_llm_code(code, output_dir)

    if success:
        return_dict["success"] = True
        return_dict["image_path"] = output_png_file
    else:
        return_dict["success"] = False
        return_dict["error"] = err_msg
    
    return return_dict

# 设置代码执行工具
code_execution_tool = Tool(
    name="execute_python_code",
    description="执行Python代码生成图表",
    func=execute_code
)

def create_code_agent():
    """创建代码生成智能体"""
    llm = ChatOpenAI(
        model="Qwen25_VL",
        api_key="dummy",
        base_url="http://localhost:8000/v1",
    )
    return llm

# 节点函数
def code_generation(state: ChartCoderState) -> ChartCoderState:
    """代码生成节点"""
    code_agent = create_code_agent()
    
    # 构建系统消息
    system_message = """"""
    
    # 如果是第一次迭代，添加系统消息和初始用户消息
    if state["iterations"] == 0:
        print("First Iteration !!!!!!!!!!!!!!!")
        state["messages"] = [SystemMessage(content=system_message)]
        user_message = HumanMessage(
            content=[
                {"type": "text", "text": state["user_input"]},
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/png;base64,{state['image_data_base64']}",
                        "detail": "high"
                    }
                }
            ]
        )
        state["messages"].append(user_message)
    else:
        # 如果有执行错误，优先处理错误
        if not "gen_image_data_base64" in state:
            print("Execution Error Input !!!!!!!!!!!!!!!")
            print(f"执行错误: {state['execution_error']}")
            feedback_message = HumanMessage(
                content=[
                    {"type": "text", "text": f"The code execution encountered an error: {state['execution_error']}\nPlease fix the code to resolve this error."},
                ]
            )
            state["messages"].append(feedback_message)
        # 如果有生成的图像，将其作为反馈
        else:
            print("Visual Feedback Input !!!!!!!!!!!!!!!")
            feedback_message = HumanMessage(
                content=[
                    {"type": "text", "text": "Please optimize the code based on the differences between the generated chart and the reference chart."},
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{state['image_data_base64']}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{state['gen_image_data_base64']}",
                            "detail": "high"
                        }
                    }
                ]
            )
            state["messages"].append(feedback_message)
    
    # 调用LLM获取响应
    response = code_agent.invoke(state["messages"])
    
    # 更新状态
    new_state = state.copy()
    new_state["code"] = process_llm_response(response.content)
    new_state["messages"] = state["messages"] + [response]
    new_state["iterations"] = state["iterations"] + 1
    
    return new_state

def code_execution(state: ChartCoderState) -> ChartCoderState:
    """代码执行节点"""
    code = state["code"]
    if not code:
        return state
    
    # 执行代码
    result = code_execution_tool.invoke(code)
    
    # 更新状态
    new_state = state.copy()
    print(result)
    if result["success"]:
        new_state["gen_image_data_base64"] = image_path_to_base64(result["image_path"])
    else:
        new_state["execution_error"] = result["error"]
    
    return new_state

def should_continue(state: ChartCoderState) -> str:
    """决定是否继续迭代"""
    print(f"Iteration: {state['iterations']}")
    if state["iterations"] >= state["max_iterations"]:
        return "end"
    
    return "continue"

# 构建工作流图
def build_chart_coder_graph():
    """构建图表代码生成工作流图"""
    workflow = StateGraph(ChartCoderState)
    
    # 添加节点
    workflow.add_node("code_generation", code_generation)
    workflow.add_node("code_execution", code_execution)
    
    # 添加条件边
    workflow.add_conditional_edges(
        "code_generation",
        should_continue,
        {
            "continue": "code_execution",
            "end": END
        }
    )
    
    # 设置边
    workflow.add_edge("code_execution", "code_generation")
    
    # 设置入口点
    workflow.set_entry_point("code_generation")
    
    return workflow.compile()

# 初始化图表代码生成器
chart_coder_app = build_chart_coder_graph()

def generate_chart_code(user_input: str, image_path: str, max_iterations: int = MAX_ITERATIONS):
    """生成图表代码的主函数"""
    # 初始化状态
    if isinstance(image_path, str):
        image_data_base64 = image_path_to_base64(image_path)
    else:
        image_data_base64 = image_to_base64(image_path)
    initial_state: ChartCoderState = {
        "user_input": user_input,
        "image_data_base64": image_data_base64,
        "code": None,
        "iterations": 0,
        "max_iterations": max_iterations,
        "execution_error": None,
        "messages": []
    }
    
    # 执行工作流
    result = chart_coder_app.invoke(initial_state)

    # 提取消息内容
    for i in range(len(result["messages"])):
        result["messages"][i] = result["messages"][i].content
    
    return result

if __name__ == "__main__":
    chart_mimic_benchmark = ChartMimicBenchmark()
    dataset = chart_mimic_benchmark.get_dataset()

    from tqdm import tqdm
    for i,data in tqdm(enumerate(dataset), total=len(dataset)):
        if i <= 5:
            continue

        image = data["images"][0]
        user_query = data["problem"]
        gt_code = data["answer"]

        result = generate_chart_code(user_query, image_path=image)
        
    print(f"生成的代码:\n{result['code']}")
    print(f"迭代次数: {result['iterations']}") 