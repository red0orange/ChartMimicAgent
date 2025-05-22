from typing import List, Tuple, Dict
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import uuid

import matplotlib.pyplot as plt
import re

def evaluate_text_matching(generation_code: str, golden_code: str, use_position: bool = False, use_axs: bool = True) -> Dict[str, float]:
    """评估生成的图表代码和标准代码之间的文本匹配程度
    
    Args:
        generation_code_file: 生成的代码文件路径
        golden_code_file: 标准代码文件路径
        use_position: 是否考虑文本位置信息
        use_axs: 是否使用坐标轴信息
        
    Returns:
        dict: 包含precision、recall和f1的字典
    """
    # 在savefig后添加删除文件的代码
    def add_delete_file_after_savefig(code):
        pattern = r'(plt\.savefig\(.*?\))'
        def replace_func(match):
            full_savefig = match.group(1)
            # 提取文件名
            file_path = re.search(r'plt\.savefig\([\'"](.*?)[\'"]', full_savefig).group(1)
            return f'{full_savefig}\nos.remove("{file_path}")'
        return re.sub(pattern, replace_func, code, flags=re.DOTALL)
    
    generation_code = add_delete_file_after_savefig(generation_code)
    golden_code = add_delete_file_after_savefig(golden_code)
    
    generation_code_temp_file = "temp_{}.py".format(str(uuid.uuid4())[:8])
    golden_code_temp_file = "temp_{}.py".format(str(uuid.uuid4())[:8])
    with open(generation_code_temp_file, 'w') as f:
        f.write(generation_code)
    with open(golden_code_temp_file, 'w') as f:
        f.write(golden_code)

    def log_texts(code_file):
        """获取代码中的文本对象"""
        with open(code_file, 'r') as f:
            lines = f.readlines()
        code = ''.join(lines)

        prefix = get_prefix()
        output_file = code_file.replace(".py", "_log_texts.txt")
        suffix = get_suffix(output_file)
        code = prefix + code + suffix

        if not use_axs:
            savefig_idx = code.find("plt.savefig")
            ax_ticks_deletion_code = get_ax_ticks_deletion_code()
            code = code[:savefig_idx] + ax_ticks_deletion_code + code[savefig_idx:]

        code_log_texts_file = code_file.replace(".py", "_log_texts.py")
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
lib_path = os.path.join(os.environ["PROJECT_PATH"], "examples/reward_function/chartmimic_evaluator")
sys.path.insert(0, lib_path)
import global_config
global_config.reset_texts()
from matplotlib.backends.backend_pdf import RendererPdf

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
with open('{output_file}', 'w') as f:
    f.write(str(drawed_texts))
"""

    def get_ax_ticks_deletion_code():
        return """
all_axes = plt.gcf().get_axes()
for ax in all_axes:
    ax.set_xticks([])
    ax.set_yticks([])
"""

    # 主要评估逻辑
    generation_texts = log_texts(generation_code_temp_file)
    golden_texts = log_texts(golden_code_temp_file)
    
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