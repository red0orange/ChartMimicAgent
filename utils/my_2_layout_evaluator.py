from typing import List, Tuple, Dict
from dotenv import load_dotenv
load_dotenv()

import os
import re
import sys
import uuid

import matplotlib.pyplot as plt

def evaluate_layout_matching(generation_code: str, golden_code: str, temp_save_dir_name: str = None) -> Dict[str, float]:
    """评估生成的图表代码和标准代码之间的布局匹配程度
    
    Args:
        generation_code: 生成的代码字符串
        golden_code: 标准代码字符串
        
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

    def log_layouts(code_file):
        """获取代码中的布局对象"""
        with open(code_file, 'r') as f:
            lines = f.readlines()
        code = ''.join(lines)

        prefix = get_prefix()
        output_file = code_file.replace(".py", "_log_layouts.txt")
        if "/graph" in code_file:
            suffix = get_suffix_special_for_graph(output_file)
        else:
            suffix = get_suffix(output_file)

        code = prefix + code + suffix

        code_log_texts_file = code_file.replace(".py", "_log_layouts.py")
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

    def calculate_metrics(generation_layouts: List[Tuple], golden_layouts: List[Tuple]) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {"precision": 0, "recall": 0, "f1": 0}
        
        if len(generation_layouts) == 0 or len(golden_layouts) == 0:
            return metrics

        len_generation = len(generation_layouts)
        len_golden = len(golden_layouts)

        n_correct = 0
        for t in golden_layouts:
            if t in generation_layouts:
                n_correct += 1
                generation_layouts.remove(t)

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
"""
    
    def get_suffix(output_file):
        return f"""

def get_gridspec_layout_info(fig):
    layout_info = {{}}
    for ax in fig.axes:
        spec = ax.get_subplotspec()
        if spec is None:
            continue
        gs = spec.get_gridspec()
        nrows, ncols = gs.get_geometry()
        row_start, row_end = spec.rowspan.start, spec.rowspan.stop - 1  # Zero-based and inclusive
        col_start, col_end = spec.colspan.start, spec.colspan.stop - 1  # Zero-based and inclusive
        layout_info[ax] = dict(nrows=nrows, ncols=ncols, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)
    # print(layout_info)
    layout_info = list(layout_info.values())
    return layout_info

layout_info = get_gridspec_layout_info(fig=plt.gcf())
with open('{output_file}', 'w') as f:
    f.write(str(layout_info))
"""
    
    def get_suffix_special_for_graph(output_file):
        return f"""
def get_gridspec_layout_info(fig):
    layout_info = {{}}
    for ax in fig.axes:
        layout_info[ax] = dict(nrows=1, ncols=1, row_start=0, row_end=1, col_start=0, col_end=1)
    # print(layout_info)
    layout_info = list(layout_info.values())
    return layout_info

layout_info = get_gridspec_layout_info(fig=plt.gcf())
with open('{output_file}', 'w') as f:
    f.write(str(layout_info))
"""

    # 主要评估逻辑
    generation_layouts = log_layouts(generation_code_temp_file)
    golden_layouts = log_layouts(golden_code_temp_file)
    
    metrics = calculate_metrics(generation_layouts, golden_layouts)
    
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