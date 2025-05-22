from typing import List, Tuple, Dict
from dotenv import load_dotenv
load_dotenv()

import os
import sys
cur_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cur_file_path)

import re
import uuid
from itertools import permutations
from multiprocessing import Pool, cpu_count
from color_utils import group_color, calculate_similarity_single


def calculate_similarity_for_permutation(args):
    shorter, perm = args
    current_similarity = sum(calculate_similarity_single(c1, c2) for c1, c2 in zip(shorter, perm))
    return current_similarity

def evaluate_color_matching(generation_code: str, golden_code: str) -> Dict[str, float]:
    """评估生成的图表代码和标准代码之间的颜色匹配程度
    
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
            file_path = re.search(r'plt\.savefig\([\'"](.*?)[\'"]', full_savefig).group(1)
            return f'{full_savefig}\nos.remove("{file_path}")'
        return re.sub(pattern, replace_func, code, flags=re.DOTALL)
    generation_code = add_delete_file_after_savefig(generation_code)
    golden_code = add_delete_file_after_savefig(golden_code)

    # 创建临时文件
    generation_code_temp_file = "temp_{}.py".format(str(uuid.uuid4())[:8])
    golden_code_temp_file = "temp_{}.py".format(str(uuid.uuid4())[:8])
    with open(generation_code_temp_file, 'w') as f:
        f.write(generation_code)
    with open(golden_code_temp_file, 'w') as f:
        f.write(golden_code)

    def log_colors(code_file):
        """获取代码中的颜色对象"""
        with open(code_file, 'r') as f:
            lines = f.readlines()
        code = ''.join(lines)

        prefix = get_prefix()
        output_file = code_file.replace(".py", "_log_colors.txt")
        suffix = get_suffix(output_file)
        code = prefix + code + suffix

        code_log_colors_file = code_file.replace(".py", "_log_colors.py")
        with open(code_log_colors_file, 'w') as f:
            f.write(code)
        
        os.system(f"python3 {code_log_colors_file}")

        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                colors = f.read()
                colors = eval(colors)
            os.remove(output_file)
        else:
            colors = []

        os.remove(code_log_colors_file)
        return colors

    def calculate_metrics(generation_colors: List[Tuple], golden_colors: List[Tuple]) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {"precision": 0, "recall": 0, "f1": 0}
        
        if len(generation_colors) == 0 or len(golden_colors) == 0:
            return metrics

        generation_colors = list(generation_colors)
        golden_colors = list(golden_colors)

        group_generation_colors = group_color(generation_colors)
        group_golden_colors = group_color(golden_colors)

        # 合并颜色组
        merged_color_group = list(set(list(group_generation_colors.keys()) + list(group_golden_colors.keys())))
        for color in merged_color_group:
            if color not in group_generation_colors:
                group_generation_colors[color] = []
            if color not in group_golden_colors:
                group_golden_colors[color] = []

        max_set_similarity = 0
        for color in merged_color_group:
            max_set_similarity += calculate_similarity_parallel(group_generation_colors[color], group_golden_colors[color])

        metrics["precision"] = max_set_similarity / len(generation_colors) if len(generation_colors) != 0 else 0
        metrics["recall"] = max_set_similarity / len(golden_colors) if len(golden_colors) != 0 else 0
        
        if metrics["precision"] + metrics["recall"] == 0:
            metrics["f1"] = 0
        else:
            metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])

        return metrics

    def calculate_similarity_parallel(lst1, lst2):
        if len(lst1) == 0 or len(lst2) == 0:
            return 0

        shorter, longer = (lst1, lst2) if len(lst1) <= len(lst2) else (lst2, lst1)
        perms = permutations(longer, len(shorter))

        with Pool(processes=cpu_count()) as pool:
            similarities = pool.map(calculate_similarity_for_permutation, [(shorter, perm) for perm in perms])

        return max(similarities)

    def get_prefix():
        with open(os.environ["PROJECT_PATH"]+"/examples/reward_function/chartmimic_evaluator/color_evaluator_prefix.py", "r") as f:
            prefix = f.read()
        return prefix

    def get_suffix(output_file):
        return f"""
drawed_colors = list(set(drawed_colors))
drawed_colors = update_drawed_colors(drawed_objects)
if len(drawed_colors) > 10:
    drawed_colors = filter_color(drawed_colors)
with open('{output_file}', 'w') as f:
    f.write(str(drawed_colors))
"""

    # 主要评估逻辑
    generation_colors = log_colors(generation_code_temp_file)
    golden_colors = log_colors(golden_code_temp_file)
    
    metrics = calculate_metrics(generation_colors, golden_colors)
    
    # 清理临时文件
    redunant_file = os.path.basename(golden_code_temp_file).replace(".py", ".pdf")
    if os.path.exists(redunant_file):
        os.remove(redunant_file)
    if os.path.exists(generation_code_temp_file):
        os.remove(generation_code_temp_file)
    if os.path.exists(golden_code_temp_file):
        os.remove(golden_code_temp_file)
        
    return metrics