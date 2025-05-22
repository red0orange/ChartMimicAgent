import re
import json
import os
import uuid
import subprocess


def extract_code(text):
    """
    从markdown文本中提取Python代码块
    
    Args:
        text (str): 包含代码块的markdown文本
    
    Returns:
        list: 提取出的代码块列表
    """
    code = re.findall(r"```python(.*?)```", text, re.DOTALL)
    if len(code) == 0:
        code = [""]
    return code


def process_llm_response(response):
    """
    处理LLM响应并提取代码
    
    Args:
        response: LLM的响应内容
        model_type (str): 模型类型，可选 "gpt", "claude", "idefics2"
    
    Returns:
        str: 提取并处理后的代码
    """
    # 根据模型类型提取代码
    code = response
    
    # 处理不同模型的特殊格式
    code = extract_code(code)[0]
    
    if code == "":
        return ""
    
    # 清理代码中的保存和显示命令
    code = re.sub(r"plt\.savefig\(.*\n*", "", code, flags=re.S)
    code = re.sub(r"plt\.show\(.*\n*", "", code, flags=re.S)
    
    return code.strip()


def execute_llm_code(code, output_dir):
    """执行Python代码并捕获生成的图表"""
    # 生成随机文件名
    random_filename = str(uuid.uuid4())[:8]
    output_file = f"temp_{random_filename}.py"
    output_file = os.path.join(output_dir, output_file)
    output_pdf_file = output_file.replace(".py", ".pdf")
    output_png_file = output_file.replace(".py", ".png")

    # 为代码添加适当的缩进
    indented_code = '\n    '.join(code.strip().split('\n'))
    code = (
        "try:\n    "
        + indented_code
        + '\nexcept Exception as e:\n    exit(100)\nplt.savefig("{}")\nplt.savefig("{}")'.format(
            output_pdf_file, output_png_file
        )
    )
    
    err_msg = ""
    try:
        with open(output_file, "w") as f:
            f.write(code)
        
        result = subprocess.run(["python3", output_file], capture_output=True, text=True)
        if result.returncode != 0:
            success = False
            err_msg = result.stderr
        else:
            success = True
    except Exception as e:
        success = False
        print(f"Exception occurred: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(output_file):
            os.remove(output_file)
        # if os.path.exists(output_pdf_file):
        #     os.remove(output_pdf_file)

    return success, output_pdf_file, output_png_file, err_msg


def process_and_execute_code(code, model_type="qwen", variable_code=None):
    """
    处理并执行代码
    
    Args:
        code: 原始代码或模型响应
        model_type (str): 模型类型，可选 "gpt", "claude", "idefics2"
        variable_code (str): 变量代码（可选）
    
    Returns:
        bool: 执行是否成功
    """
    # 生成随机文件名
    random_filename = str(uuid.uuid4())[:8]
    output_file = f"temp_{random_filename}.py"
    
    # 首先处理LLM响应
    code = process_llm_response(code, model_type)
    
    # 添加变量代码（如果提供）
    if variable_code:
        variable_code = variable_code.replace("\n", "\n    ")
        code = variable_code + "\n    " + code
    
    # 为代码添加适当的缩进
    indented_code = '\n    '.join(code.strip().split('\n'))
    code = (
        "try:\n    "
        + indented_code
        + '\nexcept Exception as e:\n    exit(100)\nplt.savefig("{}")'.format(
            output_file.replace(".py", f".pdf")
        )
    )
    
    try:
        with open(output_file, "w") as f:
            f.write(code)
        
        result = subprocess.run(["python3", output_file], capture_output=True, text=True)
        if result.returncode != 0:
            success = False
        else:
            success = True
    except Exception as e:
        success = False
        print(f"Exception occurred: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(output_file):
            os.remove(output_file)
        pdf_file = output_file.replace(".py", ".pdf")
        if os.path.exists(pdf_file):
            os.remove(pdf_file)

    return success, code


if __name__ == "__main__":
    test_input = """To reproduce the given matplotlib plots, we need to follow these steps:\n\n1. **Create the scatter plot for "Luxury Brand Popularity vs Price"**:\n   - Use `plt.scatter` to plot the data points.\n   - Label the axes and add a title.\n\n2. **Create the box plot for "Customer Satisfaction Distribution Across Brands"**:\n   - Use `plt.boxplot` to create the box plots.\n   - Label the axes and add a title.\n\n3. **Set the figure size and display the plots**:\n   - Use `plt.figure(figsize=(10.0, 6.0))` to set the figure size.\n   - Display the plots using `plt.show()`.\n\nHere is the Python code to reproduce the given picture:\n\n```python\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# Data for the scatter plot\ndata_scatter = {\n    \'Gucci\': [1100, 8.5],\n    \'Chanel\': [1800, 9.0],\n    \'Louis Vuitton\': [2000, 9.2],\n    \'Dior\': [1600, 8.0],\n    \'Prada\': [1000, 7.5],\n}\n\n# Data for the box plot\ndata_box = {\n    \'Gucci\': np.random.normal(loc=8, scale=1, size=100),\n    \'Prada\': np.random.normal(loc=8, scale=1, size=100),\n    \'Louis Vuitton\': np.random.normal(loc=5, scale=1, size=100),\n    \'Chanel\': np.random.normal(loc=4, scale=1, size=100),\n    \'Dior\': np.random.normal(loc=9, scale=1, size=100),\n}\n\n# Create the scatter plot\nplt.figure(figsize=(10.0, 6.0))\n\n# Scatter plot\nplt.subplot(1, 2, 1)\nplt.scatter([d[0] for d in data_scatter.values()], [d[1] for d in data_scatter.values()], s=100)\nplt.title(\'Luxury Brand Popularity vs Price\')\nplt.xlabel(\'Average Price ($)\')\nplt.ylabel(\'Popularity Index\')\nplt.grid(True)\n\n# Box plot\nplt.subplot(1, 2, 2)\nplt.boxplot(list(data_box.values()), labels=list(data_box.keys()))\nplt.title(\'Customer Satisfaction Distribution Across Brands\')\nplt.xlabel(\'Brands\')\nplt.ylabel(\'Customer Satisfaction\')\nplt.grid(True)\n\n# Show the plots\nplt.tight_layout()\nplt.show()\n```\n\nThis code will generate the two plots as shown in the picture. The `plt.tight_layout()` function is used to adjust the layout to avoid overlapping labels."""
    process_and_execute_code(test_input)
