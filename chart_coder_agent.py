"""
数据图表代码生成多智能体系统

该系统使用LangGraph实现，包含以下组件：
- Code Agent (VLM): 负责生成图表代码
- Visual Agent (LLM): 负责分析代码并提供反馈
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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from utils.my_0_response_processor import process_llm_response, execute_llm_code

cur_file_dir = os.path.dirname(os.path.abspath(__file__))

# langsmith
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "tutor"

# 加载环境变量(如果存在.env文件)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("提示: 安装 python-dotenv 可以从.env文件加载配置")

# 配置环境变量（优先使用.env中的配置，否则使用默认值）
API_KEY = os.environ.get("OPENAI_API_KEY", "sk-gKHgXqGOLol2jNZ2aTbVJwUVUoWPNB4nOpZn8AUsSVspbiXZ")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://35.220.164.252:3888/v1")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "2"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0"))

# 设置API环境变量
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_BASE_URL"] = BASE_URL

class ChartCoderState(TypedDict):
    """图表代码生成系统的状态"""
    user_input: str
    image_data_base64: str
    gen_image_data_base64: str
    code: Optional[str]
    feedback: List[str]
    iterations: int
    max_iterations: int
    execution_error: Optional[str]
    messages: Annotated[list, add_messages]  # 使用LangGraph的消息历史管理

def image_path_to_base64(image_path: str) -> str:
    """
    将图片文件转换为base64编码的字符串
    
    Args:
        image_path: 图片文件的路径
        
    Returns:
        str: base64编码的图片数据字符串
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        raise Exception(f"转换图片失败: {str(e)}")

def image_to_base64(image) -> str:
    """
    将PIL图像对象转换为base64编码的字符串
    
    Args:
        image: PIL图像对象
    """
    buf = io.BytesIO()
    image.save(buf, format="png")  # 保存为完整的图像格式
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_str

def base64_to_plt_image(base64_str):
    # 如果base64字符串包含头部信息（如'data:image/jpeg;base64,'），则需要移除它
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    # 解码base64字符串为二进制数据
    img_data = base64.b64decode(base64_str)
    
    # 使用PIL打开图像
    img = Image.open(io.BytesIO(img_data))
    
    return img  # 如果需要返回PIL图像对象以便进一步处理

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

# 设置Code Agent (VLM)
def create_code_agent():
    """创建代码生成智能体"""
    llm = ChatOpenAI(
        model="Qwen/Qwen2.5-VL-32B-Instruct",
        base_url=BASE_URL,
        temperature=TEMPERATURE,
    )
    
    return llm

# 设置Visual Agent (LLM)
def create_visual_agent():
    """创建视觉分析智能体"""
    llm = ChatOpenAI(
        model="Qwen/Qwen2.5-VL-32B-Instruct",  # 使用纯文本LLM
        base_url=BASE_URL,
        temperature=TEMPERATURE,
    )
    
    return llm

# 节点函数
def code_generation(state: ChartCoderState) -> ChartCoderState:
    """代码生成节点"""
    code_agent = create_code_agent()
    
    # 构建系统消息
    # system_message = """你是一位专业的数据图表代码生成助手。根据提供的参考图表图片和用户的要求，生成的绘制图表的 Python 代码。使用matplotlib、pandas 和 numpy 库。"""
    system_message = """You are a professional data chart code generation assistant. According to the provided reference chart images and user requirements, generate Python code to draw charts. Use matplotlib, pandas and numpy libraries."""
    
    # 如果是第一次迭代，添加系统消息和初始用户消息
    if state["iterations"] == 0:
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
    
    # 如果有反馈，添加最新的反馈作为用户消息
    if state["feedback"]:
        feedback_message = f"根据以下反馈改进代码：\n{state['feedback'][-1]}"
        state["messages"].append(HumanMessage(content=feedback_message))
    
    # 调用LLM获取响应
    response = code_agent.invoke(state["messages"])
    
    # 更新状态
    new_state = state.copy()
    new_state["code"] = process_llm_response(response.content)  # 提取代码
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
    if result["success"]:
        new_state["gen_image_data_base64"] = image_path_to_base64(result["image_path"])
    else:
        new_state["execution_error"] = result["error"]
    
    return new_state

def feedback_generation(state: ChartCoderState) -> ChartCoderState:
    """反馈生成节点"""
    visual_agent = create_visual_agent()
    
    # 构建系统消息
    # system_message = """你是一位专业的数据图表可视化评估助手。分析输入的参考图表（左）和生成的图表（右），简要提出能使得生成的图表更接近参考图表的改进建议。"""
    system_message = """You are a data visualization assessment assistant. A image is provided, which contains the input reference chart (left) and the generated chart (right). Please briefly provide improvement suggestions to make the generated chart more similar to the reference chart."""
    
    # 构建消息列表
    messages = [SystemMessage(content=system_message)]

    # 拼接参考图表和生成的图表
    image_data = state["image_data_base64"]
    gen_image_data = state["gen_image_data_base64"]
    image = base64_to_plt_image(image_data)
    gen_image = base64_to_plt_image(gen_image_data)
    # 保持图像比例不变，将 gen_image 缩放到与 image 相同的高度
    target_height = image.height
    gen_image = gen_image.resize((int(gen_image.width * target_height / gen_image.height), target_height))

    width = image.width + gen_image.width
    height = max(image.height, gen_image.height)
    cat_image = Image.new('RGB', (width, height), 'white')
    cat_image.paste(image, (0, 0))
    cat_image.paste(gen_image, (image.width, 0))
    cat_image_base64 = image_to_base64(cat_image)

    # 准备用户消息内容
    user_message = HumanMessage(
        content=[
            {
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/png;base64,{cat_image_base64}",
                    "detail": "high"
                }
            },
        ]
    )
    
    messages.append(user_message)
    
    # 调用LLM获取响应
    response = visual_agent.invoke(messages)
    
    # 更新状态
    new_state = state.copy()
    new_state["feedback"] = state["feedback"] + [response.content]
    
    return new_state

def error_feedback_generation(state: ChartCoderState) -> ChartCoderState:
    """错误反馈生成节点"""
    # 直接使用执行错误作为反馈
    new_state = state.copy()
    error_message = f"The code execution encountered an error: {state['execution_error']}\nPlease fix it."
    new_state["feedback"] = state["feedback"] + [error_message]
    return new_state

def should_continue(state: ChartCoderState) -> str:
    """决定是否继续迭代"""
    # 如果达到最大迭代次数，则结束
    if state["iterations"] >= state["max_iterations"]:
        return "end"
    
    # # 如果代码执行有错误，继续迭代
    # if state.get("execution_error"):
    #     return "continue"
    
    # # 如果没有图像数据，继续迭代
    # if not state["image_data_base64"]:
    #     return "continue"
    
    # 检查最后一次反馈是否表明需要进一步改进
    if state["feedback"] and any(keyword in state["feedback"][-1].lower() 
                             for keyword in ["improvement", "optimization", "fix", "problem", "suggestion"]):
        return "continue"
    
    # 默认结束
    return "end"

# 构建工作流图
def build_chart_coder_graph():
    """构建图表代码生成工作流图"""
    # 创建状态图
    workflow = StateGraph(ChartCoderState)
    
    # 添加节点
    workflow.add_node("code_generation", code_generation)
    workflow.add_node("code_execution", code_execution)
    workflow.add_node("feedback_generation", feedback_generation)
    workflow.add_node("error_feedback_generation", error_feedback_generation)
    
    # 设置边
    workflow.add_edge("code_generation", "code_execution")
    
    # 添加条件边：根据代码执行结果决定下一步
    workflow.add_conditional_edges(
        "code_execution",
        lambda state: "end" if state["iterations"] >= state["max_iterations"] else ("error_feedback" if state.get("execution_error") else "normal_feedback"),
        {
            "error_feedback": "error_feedback_generation",
            "normal_feedback": "feedback_generation",
            "end": END
        }
    )
    
    # 条件边：决定是否继续迭代
    workflow.add_conditional_edges(
        "feedback_generation",
        should_continue,
        {
            "continue": "code_generation",
            "end": END
        }
    )
    workflow.add_edge("error_feedback_generation", "code_generation")
    
    # 设置入口点
    workflow.set_entry_point("code_generation")
    
    return workflow.compile()

# 初始化图表代码生成器
chart_coder_app = build_chart_coder_graph()

def generate_chart_code(user_input: str, image_path: str, max_iterations: int = MAX_ITERATIONS):
    """生成图表代码的主函数"""
    # 初始化状态
    image_data_base64 = image_path_to_base64(image_path)
    initial_state: ChartCoderState = {
        "user_input": user_input,
        "image_data_base64": image_data_base64,
        "code": None,
        "feedback": [],
        "iterations": 0,
        "max_iterations": max_iterations,
        "execution_error": None,
        "messages": []  # 初始化消息列表
    }
    
    # 执行工作流
    result = chart_coder_app.invoke(initial_state)
    
    return result

def pdf_to_png(pdf_path: str) -> str:
    try:
        # 将 PDF 转换为图片
        save_path = pdf_path.replace(".pdf", ".png")
        images = convert_from_path(pdf_path)
        
        # 如果 PDF 有多页，只取第一页
        if images:
            # 将图片转换为字节流
            images[0].save(save_path, "PNG")
            
            # 转换为 base64
            return image_path_to_base64(save_path)
        else:
            raise Exception("PDF 文件为空或无法转换")
            
    except Exception as e:
        raise Exception(f"PDF 转换失败: {str(e)}")

if __name__ == "__main__":
    # 示例用法
    user_query = "You are an expert Python developer who specializes in writing matplotlib code based on a given picture. I found a very nice picture in a STEM paper, but there is no corresponding source code available. I need your help to generate the Python code that can reproduce the picture based on the picture I provide.\nNote that it is necessary to use figsize=(7.0, 7.0) to set the image size to match the original size.\nNow, please give me the matplotlib code that reproduces the picture below"
    image_path = "/home/hdh/github_projects/cmr_benchmark/9_thanks_methods/ChartPlotRL/dataset/ori_500/3d_1.png"
    result = generate_chart_code(user_query, image_path=image_path)
    
    print(f"生成的代码:\n{result['code']}")
    print(f"迭代次数: {result['iterations']}")
    print(f"反馈历史:\n{result['feedback']}")