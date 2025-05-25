# 放置数据集读取、结果评估、可视化等代码
import os

import json
import uuid
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets
import concurrent.futures
from typing import Dict, Any

from utils.my_0_response_processor import process_and_execute_code
from utils.my_1_text_evaluator import evaluate_text_matching
from utils.my_2_layout_evaluator import evaluate_layout_matching
from utils.my_3_legend_evaluator import evaluate_legend_matching
from utils.my_4_color_evaluator import evaluate_color_matching

cur_file_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_PATH"] = cur_file_dir


class ChartMimicBenchmark:
    def __init__(self):
        self.data_dir = os.path.join(cur_file_dir, "datasets")
        self.train_dir = os.path.join(self.data_dir, "train")
        self.test_dir = os.path.join(self.data_dir, "validation")
        
        self.train_data = load_from_disk(self.train_dir)
        self.test_data = load_from_disk(self.test_dir)
        self.dataset = concatenate_datasets([self.train_data, self.test_data])
        pass

    def get_dataset(self):
        return self.dataset

    def evaluate(self, results):
        # 输入和 dataset 中
        pass


def process_single_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """处理单个结果的函数"""
    cur_result = result
    
    pred_success = cur_result["success"]
    if "data" not in cur_result:
        return {
            "text_matching_result": {"precision": 0, "recall": 0, "f1": 0},
            "layout_matching_result": {"precision": 0, "recall": 0, "f1": 0},
            "legend_matching_result": {"precision": 0, "recall": 0, "f1": 0},
            "color_matching_result": {"precision": 0, "recall": 0, "f1": 0}
        }
    
    pred_code = cur_result["data"]["code"]
    gt_code = cur_result["gt_code"]

    # 做一点预处理
    pred_code = pred_code.replace("w_xaxis", "xaxis")
    pred_code = pred_code.replace("w_yaxis", "yaxis")
    pred_code = pred_code.replace("w_zaxis", "zaxis")
    pred_code = pred_code.replace("tick.label", "tick._label")

    # 是否能执行
    excute_success, _ = process_and_execute_code(pred_code, process=False)
    if not excute_success:
        return {
            "excute_success": False,
            "text_matching_result": {"precision": 0, "recall": 0, "f1": 0},
            "layout_matching_result": {"precision": 0, "recall": 0, "f1": 0},
            "legend_matching_result": {"precision": 0, "recall": 0, "f1": 0},
            "color_matching_result": {"precision": 0, "recall": 0, "f1": 0}
        }

    # 在最后添加 plt.savefig("temp.png")
    random_str = str(uuid.uuid4())[:8]
    pred_code = pred_code + "\nplt.savefig('temp_{}.pdf')".format(random_str)

    # 文本匹配
    text_matching_result = evaluate_text_matching(pred_code, gt_code, temp_save_dir_name="figs")
    
    # 布局匹配
    layout_matching_result = evaluate_layout_matching(pred_code, gt_code, temp_save_dir_name="figs")
    
    # 图例匹配
    legend_matching_result = evaluate_legend_matching(pred_code, gt_code, temp_save_dir_name="figs")
    
    # 颜色匹配
    color_matching_result = evaluate_color_matching(pred_code, gt_code, temp_save_dir_name="figs")

    return {
        "excute_success": excute_success,
        "text_matching_result": text_matching_result,
        "layout_matching_result": layout_matching_result,
        "legend_matching_result": legend_matching_result,
        "color_matching_result": color_matching_result
    }

def eval_json_results(results_path: str, max_workers: int = 4):
    """
    多线程版本的评估函数
    
    Args:
        results_path: 结果文件路径
        max_workers: 最大工作线程数，默认为4
    """
    results_dir = os.path.dirname(results_path)
    results = json.load(open(results_path, "r"))
    
    all_eval_results = []
    
    # 使用线程池执行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_result = {executor.submit(process_single_result, result): result for result in results}
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(future_to_result), total=len(results)):
            try:
                result = future.result()
                print(result)
                all_eval_results.append(result)
            except Exception as e:
                print(f"处理结果时发生错误: {e}")
                all_eval_results.append({
                    "error": str(e),
                    "text_matching_result": {"precision": 0, "recall": 0, "f1": 0},
                    "layout_matching_result": {"precision": 0, "recall": 0, "f1": 0},
                    "legend_matching_result": {"precision": 0, "recall": 0, "f1": 0},
                    "color_matching_result": {"precision": 0, "recall": 0, "f1": 0}
                })

    # 保存结果
    json.dump(all_eval_results, open(os.path.join(results_dir, "all_eval_results.json"), "w"), indent=4)


def export_final_metrics(results_path: str):
    results_dir = os.path.dirname(results_path)
    results = json.load(open(results_path, "r"))

    # 统计 excute_success 为 True 的样本数量
    # excute_success = [result["excute_success"] for result in results]
    excute_success = [result["excute_success"] if "excute_success" in result else False for result in results]
    excute_success_count = sum(excute_success)
    print(f"excute_success_count: {excute_success_count}")
    print(f"excute_success_rate: {excute_success_count / len(results)}")

    text_matching_result = [result["text_matching_result"] for result in results]
    layout_matching_result = [result["layout_matching_result"] for result in results]
    legend_matching_result = [result["legend_matching_result"] for result in results]
    color_matching_result = [result["color_matching_result"] for result in results]

    # 计算 f1
    text_f1 = [result["text_matching_result"]["f1"] for result in results]
    layout_f1 = [result["layout_matching_result"]["f1"] for result in results]
    legend_f1 = [result["legend_matching_result"]["f1"] for result in results]
    color_f1 = [result["color_matching_result"]["f1"] for result in results]

    # 计算平均值
    text_f1_avg = sum(text_f1) / len(text_f1)   
    layout_f1_avg = sum(layout_f1) / len(layout_f1)
    legend_f1_avg = sum(legend_f1) / len(legend_f1)
    color_f1_avg = sum(color_f1) / len(color_f1)

    print(f"text_f1_avg: {text_f1_avg}")
    print(f"layout_f1_avg: {layout_f1_avg}")
    print(f"legend_f1_avg: {legend_f1_avg}")
    print(f"color_f1_avg: {color_f1_avg}")
    pass


if __name__ == "__main__":
    # # 评估
    # # results_path = "/home/hdh/Projects/MultiTurnChartCoder/results/res_20250524_232031/langgraph_states.json"
    # results_path = "/home/hdh/Projects/MultiTurnChartCoder/results/res_20250524_160704/baseline_states.json"
    # eval_json_results(results_path)
    # # 评估结果导出为 metrics
    # results_path = "/home/hdh/Projects/MultiTurnChartCoder/results/res_20250524_160704/all_eval_results.json"
    # export_final_metrics(results_path)

    # # 评估
    # results_path = "/home/hdh/Projects/MultiTurnChartCoder/results/res_20250524_232031/langgraph_states.json"
    # eval_json_results(results_path)
    # 评估结果导出为 metrics
    results_path = "/home/hdh/Projects/MultiTurnChartCoder/results/res_20250524_232031/all_eval_results.json"
    export_final_metrics(results_path)