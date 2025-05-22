#!/usr/bin/env python
"""
数据图表代码生成系统 - 命令行界面

提供简单的命令行界面来使用图表代码生成系统。
"""

import argparse
import base64
import os
import sys
from chart_coder_agent import generate_chart_code

def main():
    parser = argparse.ArgumentParser(description="数据图表代码生成系统")
    parser.add_argument("prompt", nargs="?", help="图表生成提示词")
    parser.add_argument("--image", "-i", help="输入图像路径（可选）")
    parser.add_argument("--output", "-o", help="输出图表路径", default="output_chart.png")
    parser.add_argument("--max-iterations", "-m", type=int, default=3, help="最大迭代次数")
    parser.add_argument("--show-feedback", "-f", action="store_true", help="显示反馈历史")
    parser.add_argument("--interactive", "-I", action="store_true", help="交互式模式")
    parser.add_argument("--save-code", "-s", help="保存生成的代码到文件")
    
    args = parser.parse_args()
    
    # 交互式模式
    if args.interactive:
        interactive_mode(args)
        return
    
    # 直接命令行模式
    if not args.prompt:
        parser.print_help()
        return
    
    # 处理输入图像
    image_data = None
    if args.image and os.path.exists(args.image):
        with open(args.image, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")
    
    # 生成代码
    print(f"正在生成图表代码（最大迭代次数: {args.max_iterations}）...")
    result = generate_chart_code(args.prompt, image_data=image_data, max_iterations=args.max_iterations)
    
    # 输出结果
    print(f"\n迭代次数: {result['iterations']}")
    print(f"\n最终代码:\n\n{result['code']}")
    
    # 保存代码
    if args.save_code:
        with open(args.save_code, "w", encoding="utf-8") as f:
            f.write(result["code"])
        print(f"\n代码已保存到: {args.save_code}")
    
    # 输出反馈
    if args.show_feedback:
        print("\n反馈历史:")
        for i, feedback in enumerate(result["feedback"]):
            print(f"\n=== 反馈 {i+1} ===\n{feedback}")
    
    # 保存图表
    if result.get("image_data"):
        with open(args.output, "wb") as f:
            f.write(base64.b64decode(result["image_data"]))
        print(f"\n图表已保存到: {args.output}")
    else:
        print("\n未能生成图表")

def interactive_mode(args):
    """交互式模式，逐步指导用户生成图表"""
    print("===== 图表代码生成系统 - 交互式模式 =====")
    
    # 输入提示词
    prompt = input("请输入您想要生成的图表描述: ")
    if not prompt:
        print("需要提供图表描述，退出程序。")
        return
    
    # 询问是否使用参考图像
    use_image = input("是否使用参考图像? (y/n): ").lower() == 'y'
    image_data = None
    
    if use_image:
        image_path = input("请输入图像路径: ")
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode("utf-8")
            print("已加载图像。")
        else:
            print(f"无法找到图像: {image_path}")
            if input("是否继续而不使用图像? (y/n): ").lower() != 'y':
                return
    
    # 询问迭代次数
    try:
        max_iterations = int(input(f"请输入最大迭代次数 (默认: {args.max_iterations}): ") or args.max_iterations)
    except ValueError:
        max_iterations = args.max_iterations
        print(f"使用默认迭代次数: {max_iterations}")
    
    # 输出文件
    output_path = input(f"请输入图表保存路径 (默认: {args.output}): ") or args.output
    
    # 是否保存代码
    save_code = input("是否保存生成的代码? (y/n): ").lower() == 'y'
    code_path = None
    if save_code:
        code_path = input("请输入代码保存路径: ")
    
    # 执行生成
    print("\n开始生成图表代码...")
    result = generate_chart_code(prompt, image_data=image_data, max_iterations=max_iterations)
    
    # 输出结果
    print(f"\n完成！迭代次数: {result['iterations']}")
    print(f"\n最终代码:\n\n{result['code']}")
    
    # 保存代码
    if save_code and code_path:
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(result["code"])
        print(f"\n代码已保存到: {code_path}")
    
    # 显示反馈
    if input("\n是否显示反馈历史? (y/n): ").lower() == 'y':
        print("\n反馈历史:")
        for i, feedback in enumerate(result["feedback"]):
            print(f"\n=== 反馈 {i+1} ===\n{feedback}")
    
    # 保存图表
    if result.get("image_data"):
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(result["image_data"]))
        print(f"\n图表已保存到: {output_path}")
    else:
        print("\n未能生成图表")
    
    print("\n感谢使用图表代码生成系统！")

if __name__ == "__main__":
    main() 