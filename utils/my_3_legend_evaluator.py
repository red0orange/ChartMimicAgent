from typing import List, Tuple, Dict
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import numpy as np

import matplotlib.pyplot as plt
import re
import uuid


def evaluate_legend_matching(generation_code: str, golden_code: str, use_position: bool = True, temp_save_dir_name: str = None) -> Dict[str, float]:
    """评估生成的图表代码和标准代码之间的图例匹配程度
    
    Args:
        generation_code: 生成的代码字符串
        golden_code: 标准代码字符串
        use_position: 是否考虑图例位置信息
        
    Returns:
        dict: 包含precision、recall和f1的字典
    """
    # 在savefig后添加删除文件的代码
    def add_delete_file_after_savefig(code):
        pattern = r'(plt\.savefig\(.*?\))'
        def replace_func(match):
            full_savefig = match.group(1)
            file_path = re.search(r'plt\.savefig\([\'"](.*?)[\'"]', full_savefig).group(1)
            return f'{full_savefig}\nos.remove("{file_path}")'
        return re.sub(pattern, replace_func, code, flags=re.DOTALL)
    generation_code = add_delete_file_after_savefig(generation_code)
    golden_code = add_delete_file_after_savefig(golden_code)
    
    if temp_save_dir_name is None:
        temp_save_dir = os.environ["PROJECT_PATH"]
    else:
        temp_save_dir = os.path.join(os.environ["PROJECT_PATH"], temp_save_dir_name)
    os.makedirs(temp_save_dir, exist_ok=True)

    generation_code_temp_file = os.path.join(temp_save_dir, "temp_{}.py".format(str(uuid.uuid4())[:8]))
    golden_code_temp_file = os.path.join(temp_save_dir, "temp_{}.py".format(str(uuid.uuid4())[:8]))
    with open(generation_code_temp_file, 'w') as f:
        f.write(generation_code)
    with open(golden_code_temp_file, 'w') as f:
        f.write(golden_code)

    def log_legends(code_file):
        """获取代码中的图例对象"""
        with open(code_file, 'r') as f:
            lines = f.readlines()
        code = ''.join(lines)

        prefix = get_prefix()
        output_file = code_file.replace(".py", ".txt")
        suffix = get_suffix(output_file)
        code = prefix + code + suffix

        code_log_texts_file = code_file.replace(".py", "_log_legends.py")
        with open(code_log_texts_file, 'w') as f:
            f.write(code)
        
        os.system(f"python3 {code_log_texts_file}")

        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                texts = f.read()
                texts = eval(texts)
            os.remove(output_file)
        else:
            texts = []

        os.remove(code_log_texts_file)
        
        return texts

    def calculate_metrics(generation_texts: List[Tuple], golden_texts: List[Tuple]) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {"precision": 0, "recall": 0, "f1": 0}
        
        if len(generation_texts) == 0 or len(golden_texts) == 0:
            metrics["precision"] = 1
            metrics["recall"] = 1
            metrics["f1"] = 1
            return metrics

        len_generation = len(generation_texts)
        len_golden = len(golden_texts)

        if not use_position:
            generation_texts = [t[-1] for t in generation_texts]
            golden_texts = [t[-1] for t in golden_texts]

            n_correct = 0
            for t in golden_texts:
                if t in generation_texts:
                    n_correct += 1
                    generation_texts.remove(t)
        else:
            generation_texts = [t[2:] for t in generation_texts]
            golden_texts = [t[2:] for t in golden_texts]

            n_correct = 0
            for t1 in golden_texts:
                for t2 in generation_texts:
                    if t1[-1] == t2[-1] and abs(t1[0] - t2[0]) <= 10 and abs(t1[1] - t2[1]) <= 10:
                        n_correct += 1
                        generation_texts.remove(t2)
                        break

        metrics["precision"] = n_correct / len_generation
        metrics["recall"] = n_correct / len_golden
        if metrics["precision"] + metrics["recall"] == 0:
            metrics["f1"] = 0
        else:
            metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])

        return metrics

    def get_prefix():
        return f"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
lib_path = os.path.join(os.environ["PROJECT_PATH"], "utils")
sys.path.insert(0, lib_path)
import global_config
global_config.reset_texts()
from matplotlib.backends.backend_pdf import RendererPdf

drawed_legend_texts = []
drawed_texts = []

def log_function(func):
    def wrapper(*args, **kwargs):
        global drawed_texts

        object = args[0]
        x = args[2]
        y = args[3]
        x_rel = ( x / object.width / 72 ) * 100
        y_rel = ( y / object.height / 72 ) * 100
        s = args[4]

        drawed_texts.append( (x, y, x_rel, y_rel, s) )
        return func(*args, **kwargs)

    return wrapper

RendererPdf.draw_text = log_function(RendererPdf.draw_text)
"""
    
    def get_suffix(output_file):
        return f"""
all_axes = plt.gcf().get_axes()
legends = [ax.get_legend() for ax in all_axes if ax.get_legend() is not None]
for legend in legends:
    for t in legend.get_texts():
        drawed_legend_texts.append(t.get_text())

new_drawed_legend_texts = []
for t1 in drawed_legend_texts:
    for t2 in drawed_texts:
        if t1 == t2[-1]:
            new_drawed_legend_texts.append(t2)
            break
drawed_legend_texts = new_drawed_legend_texts

with open('{output_file}', 'w') as f:
    f.write(str(drawed_legend_texts))
"""

    # 主要评估逻辑
    generation_texts = log_legends(generation_code_temp_file)
    golden_texts = log_legends(golden_code_temp_file)
    
    metrics = calculate_metrics(generation_texts, golden_texts)
    
    # 清理临时文件
    redunant_file = os.path.basename(golden_code_temp_file).replace(".py", ".pdf")
    if os.path.exists(redunant_file):
        os.remove(redunant_file)
    if os.path.exists(generation_code_temp_file):
        os.remove(generation_code_temp_file)
    if os.path.exists(golden_code_temp_file):
        os.remove(golden_code_temp_file)
        
    return metrics

if __name__ == "__main__":
    pass