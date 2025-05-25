# 放置数据集读取、结果评估、可视化等代码
import os

import json
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets

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


if __name__ == "__main__":
    benchmark = ChartMimicBenchmark()
    # print(benchmark.train_data)
    # print(benchmark.test_data)

    # results_path = "/home/hdh/Projects/MultiTurnChartCoder/results/res_20250523_103208/langgraph_states.json"
    results_path = "/home/hdh/Projects/MultiTurnChartCoder/results/res_20250524_160704/baseline_states.json"
    results_dir = os.path.dirname(results_path)
    results = json.load(open(results_path, "r"))

    # results_path = "/home/hdh/Projects/MultiTurnChartCoder/results/res_20250523_103208/langgraph_states.npy"
    # results = np.load(results_path, allow_pickle=True)

    all_eval_results = []
    for result, data in tqdm(zip(results, benchmark.dataset), total=len(results)):
        cur_result = result
        cur_data = data

        pred_success = cur_result["success"]
        if "data" in cur_result:
            pred_code = cur_result["data"]["code"]
        else:
            all_eval_results.append({
                "text_matching_result": {"precision": 0, "recall": 0, "f1": 0},
                "layout_matching_result": {"precision": 0, "recall": 0, "f1": 0},
                "legend_matching_result": {"precision": 0, "recall": 0, "f1": 0},
                "color_matching_result": {"precision": 0, "recall": 0, "f1": 0}
            })
            continue

        gt_code = data["answer"]

        # 做一点预处理
        pred_code = pred_code.replace("w_xaxis", "xaxis")
        pred_code = pred_code.replace("w_yaxis", "yaxis")
        pred_code = pred_code.replace("w_zaxis", "zaxis")
        pred_code = pred_code.replace("tick.label", "tick._label")

        # 是否能执行
        excute_success, _ = process_and_execute_code(pred_code, process=False)
        if not excute_success:
            all_eval_results.append({
                "excute_success": False,
                "text_matching_result": {"precision": 0, "recall": 0, "f1": 0},
                "layout_matching_result": {"precision": 0, "recall": 0, "f1": 0},
                "legend_matching_result": {"precision": 0, "recall": 0, "f1": 0},
                "color_matching_result": {"precision": 0, "recall": 0, "f1": 0}
            })
            continue

        # 文本匹配
        text_matching_result = evaluate_text_matching(pred_code, gt_code, temp_save_dir_name="figs")
        print(text_matching_result)

        # 布局匹配
        layout_matching_result = evaluate_layout_matching(pred_code, gt_code, temp_save_dir_name="figs")
        print(layout_matching_result)

        # 图例匹配
        legend_matching_result = evaluate_legend_matching(pred_code, gt_code, temp_save_dir_name="figs")
        print(legend_matching_result)

        # 颜色匹配
        color_matching_result = evaluate_color_matching(pred_code, gt_code, temp_save_dir_name="figs")
        print(color_matching_result)

        cur_eval_result = {
            "excute_success": excute_success,
            "text_matching_result": text_matching_result,
            "layout_matching_result": layout_matching_result,
            "legend_matching_result": legend_matching_result,
            "color_matching_result": color_matching_result
        }
        all_eval_results.append(cur_eval_result)

    json.dump(all_eval_results, open(os.path.join(results_dir, "all_eval_results.json"), "w"), indent=4)


        
        
        
        
        
        