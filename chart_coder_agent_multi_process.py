import os
import datetime
import multiprocessing as mp
from typing import List, Dict, Any
import traceback
import json

import numpy as np

from chart_coder_agent import generate_chart_code
from chartmimic_benchmark import ChartMimicBenchmark

cur_file_dir = os.path.dirname(os.path.abspath(__file__))

def process_single_item(data: Dict[str, Any]) -> Dict[str, Any]:
    """处理单个数据项的函数"""
    try:
        image = data["images"][0]
        user_query = data["problem"]
        gt_code = data["answer"]

        result_state = generate_chart_code(user_query, image_path=image)
        return {
            "success": True,
            "data": result_state,
            "error": None,
            "gt_code": gt_code
        }
    except Exception as e:
        error_msg = f"处理数据时发生错误: {str(e)}\n{traceback.format_exc()}"
        return {
            "success": False,
            "data": None,
            "error": error_msg,
            "gt_code": gt_code
        }

def save_results(results: List[Dict[str, Any]], save_path: str):
    """保存结果到文件"""
    # 保存为numpy文件
    np.save(save_path, results)
    
    # 同时保存一个可读的JSON文件
    json_path = save_path.replace('.npy', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 创建保存目录
    save_dir = os.path.join(cur_file_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成保存文件名
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = "res_{}".format(time_str)
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, "langgraph_states.npy")

    # 获取数据集
    chart_mimic_benchmark = ChartMimicBenchmark()
    dataset = chart_mimic_benchmark.get_dataset()

    # 设置进程数（使用CPU核心数的75%）
    # num_processes = max(1, int(mp.cpu_count() * 0.75))
    num_processes = 4
    print(f"使用 {num_processes} 个进程进行处理")

    # 创建进程池
    with mp.Pool(processes=num_processes) as pool:
        results = []
        total_items = len(dataset)
        
        # 使用imap来处理数据，这样可以实时获取结果
        for i, result in enumerate(pool.imap(process_single_item, dataset)):
            results.append(result)
            
            # 每处理10个数据项保存一次结果
            if (i + 1) % 10 == 0 or (i + 1) == total_items:
                print(f"进度: {i + 1}/{total_items}")
                save_results(results, save_path)
                
                # 打印错误信息（如果有）
                if not result["success"]:
                    print(f"警告: 第 {i + 1} 个数据项处理失败")
                    print(result["error"])
        
        # 打印最终统计信息
        success_count = sum(1 for r in results if r["success"])
        print(f"\n处理完成:")
        print(f"总数据项: {total_items}")
        print(f"成功处理: {success_count}")
        print(f"失败数量: {total_items - success_count}")
        
        # 打印最后一个成功结果的信息（如果有）
        last_success = next((r["data"] for r in reversed(results) if r["success"]), None)
        if last_success:
            print(f"\n最后一个成功结果的代码:\n{last_success['code']}")
            print(f"迭代次数: {last_success['iterations']}")
            print(f"反馈历史:\n{last_success['feedback']}")