import os
import re
import time
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import pandas as pd  # 导入 pandas 库
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
class EmotionAnalyzer:
    def __init__(self, num_workers=4):
        # Qwen 模型配置
        torch.cuda.empty_cache()  # 释放显存缓存

        print("正在加载Qwen模型...")
        model_id = "/data/huggingface_models/Qwen2.5-VL-72B-Instruct"

        # 设备检测
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"检测到设备: {self.device}")

        # 不使用BitsAndBytesConfig，尝试直接加载模型
        try:
            print("尝试使用torch_dtype加载模型...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16  # 使用fp16替代bfloat16
            ).eval()
        except Exception as e:
            print(f"使用torch_dtype加载失败: {e}")
            print("尝试使用默认配置加载模型...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                device_map="auto"
            ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id)
        print(f"模型加载完成，使用设备: {self.device}")

        # 初始化对话历史，为了兼容预置图片功能
        self.chat_history = []

        # ------ 预置图片和提示词模块 ------
        print("正在加载预置图片和提示词...")
        print("预置图片和提示词加载完成。\n")
        # ------ 预置图片和提示词模块结束 ------

        # 保存预置的chat_history，以便后续使用
        self.preset_chat_history = self.chat_history.copy()

        # 并行处理配置
        self.num_workers = num_workers

        # 设置请求重试参数
        self.max_retries = 3
        self.base_delay = 1
        self.max_delay = 10
        self.timeout = 60

        # 定义有效答案集合
        self.valid_answers = {
            "is_coordinated": ["是", "否"],
            "coordination_level": ["1", "2", "3", "4", "5", "6"], # 修改为包含6
            "emotion_polarity": ["积极情绪", "消极情绪"],
            "emotion_type": ["高兴", "惊讶", "伤心", "厌恶", "害怕", "愤怒"],
            "judgment_basis": ["上面部", "下面部"]
        }

        # 初始化错误收集列表
        self.parsing_errors = []

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """加载图像文件为PIL Image对象"""
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                return image
            else:
                print(f"图像文件不存在: {image_path}")
                return None
        except Exception as e:
            print(f"加载图像文件失败: {image_path}, 错误: {e}")
            return None

    def api_request(self, prompt: str, image_path: str) -> Optional[str]:
        """使用Qwen模型进行本地推理"""
        # 加载图像
        image = self.load_image(image_path)
        if image is None:
            return None

        # 复制预设的聊天历史作为基础
        messages = []
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        })

        print(f"正在处理图像: {image_path}")
        print(f"消息数量: {len(messages)}")

        for attempt in range(self.max_retries):
            try:
                # 应用聊天模板
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device)

                # 记录输入长度
                input_len = inputs["input_ids"].shape[-1]

                # 生成回应
                with torch.inference_mode():
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
			            top_p=1,
                        top_k=None,
			            temperature=1,
			            do_sample=True,
                    )
                    # 只提取新生成的部分
                    generation = generation[0][input_len:]

                # 解码回应
                response = self.processor.decode(generation, skip_special_tokens=True)

                # 将此次对话加入历史，便于下一次使用
                current_message = {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},  # 保存路径而不是对象
                        {"type": "text", "text": prompt},
                    ],
                }
                assistant_message = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}]
                }
                self.chat_history.append(current_message)
                self.chat_history.append(assistant_message)

                return response

            except Exception as e:
                print(f"模型推理异常 (第 {attempt + 1} 次尝试): {e}")
                import traceback
                traceback.print_exc()
                if attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    time.sleep(delay)
                else:
                    print("达到最大重试次数，放弃处理")
                    return None

        return None  # 所有重试都失败

    def parse_response(self, response: str) -> Optional[Dict]:
        """解析模型回复，提取各个问题的答案并进行严格校验，返回解析结果或 None"""
        result = {
            "is_coordinated": None,
            "coordination_level": None,
            "emotion_polarity": None,
            "emotion_type": None,
            "judgment_basis": None
        }
        valid = True # 假设初始状态是有效的

        # 提取是否协调
        match = re.search(r"1[\.\s]*图片中所表现出来的表情是否协调.*?回答：\s*(?:\[)?(是|否)(?:\])?", response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer in self.valid_answers["is_coordinated"]:
                result["is_coordinated"] = answer
            else:
                valid = False # 校验失败
                print(f"  校验失败: is_coordinated 的值为 '{answer}', 必须是 {self.valid_answers['is_coordinated']}") # Keep for debugging
        else:
            valid = False # 解析失败
            print(f"  解析失败: 未能提取 is_coordinated 的回答") # Keep for debugging

        # 提取协调性等级
        match = re.search(r"2[\.\s]*图片中表情的协调/不协调性等级.*?回答：\s*(?:\[)?([1-6])(?:\])?", response, re.DOTALL) # 修改为匹配1-6
        if match:
            answer = match.group(1).strip()
            if answer in self.valid_answers["coordination_level"]:
                result["coordination_level"] = answer
            else:
                valid = False # 校验失败
                print(f"  校验失败: coordination_level 的值为 '{answer}', 必须是 {self.valid_answers['coordination_level']}") # Keep for debugging
        else:
            valid = False # 解析失败
            print(f"  解析失败: 未能提取 coordination_level 的回答") # Keep for debugging

        # 提取情绪极性
        match = re.search(r"3[\.\s]*图片中的整体情绪类型.*?回答：\s*(?:\[)?(积极情绪|消极情绪)(?:\])?", response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer in self.valid_answers["emotion_polarity"]:
                result["emotion_polarity"] = answer
            else:
                valid = False # 校验失败
                print(f"  校验失败: emotion_polarity 的值为 '{answer}', 必须是 {self.valid_answers['emotion_polarity']}") # Keep for debugging
        else:
            valid = False # 解析失败
            print(f"  解析失败: 未能提取 emotion_polarity 的回答") # Keep for debugging

        # 提取情绪类型
        match = re.search(r"4[\.\s]*图片所表现的情绪.*?回答：\s*(?:\[)?(高兴|惊讶|伤心|厌恶|害怕|愤怒)(?:\])?", response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer in self.valid_answers["emotion_type"]:
                result["emotion_type"] = answer
            else:
                valid = False # 校验失败
                print(f"  校验失败: emotion_type 的值为 '{answer}', 必须是 {self.valid_answers['emotion_type']}") # Keep for debugging
        else:
            valid = False # 解析失败
            print(f"  解析失败: 未能提取 emotion_type 的回答") # Keep for debugging

        # 提取判断依据
        match = re.search(r"5[\.\s]*判断依据.*?回答：\s*(?:\[)?(上面部|下面部)(?:\])?", response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer in self.valid_answers["judgment_basis"]:
                result["judgment_basis"] = answer
            else:
                valid = False # 校验失败
                print(f"  校验失败: judgment_basis 的值为 '{answer}', 必须是 {self.valid_answers['judgment_basis']}") # Keep for debugging
        else:
            valid = False # 解析失败
            print(f"  解析失败: 未能提取 judgment_basis 的回答") # Keep for debugging

        if not valid:
            return None # 如果任何解析或校验失败，返回 None

        return result


    def process_single_image(self, image_path: str) -> Optional[Dict]:
        """处理单张图像的分析任务，包含重试机制和严格的结果校验"""
        prompt = """
请严格按照以下格式逐一回答问题：

1.图片中所表现出来的表情是否协调
你认为所呈现的面孔图片中的表情是否可能同时出现在一个人的面孔上，请用是否来进行回答（如无法进行判断，请说出一个你认为可能性较大的答案）
回答：[是/否]

2.图片中表情的协调/不协调性等级
你认为图片中表情的协调性等级是多少，请用1—6之间的数字来进行回答（例如，在上一个选择中你的答案为协调，则1代表"些许协调"，2代表"较为协调"，3代表"协调"，4代表"非常协调"，5代表"十分协调", 6代表"极其协调"；不协调如上）
回答：[1/2/3/4/5/6]

3.图片中的整体情绪类型
你认为图片中的表情属于积极情绪还是消极情绪，请用积极情绪或消极情绪来进行回答（如无法进行判断，请说出一个你认为可能性较大的答案）
回答：[积极情绪/消极情绪]

4.图片所表现的情绪
你认为图片中的情绪属于高兴、惊讶、伤心、厌恶、害怕、愤怒中的哪一个？请用以上六个词中的一个词来进行回答（如无法进行判断，请说出一个你认为可能性较大的答案）
回答：[高兴/惊讶/伤心/厌恶/害怕/愤怒] 
5.判断依据
你在判断图片所表达的情绪时，是通过面部哪个区域进行判断的？请用上面部或下面部来进行回答（如无法进行判断，请说出一个你认为可能性较大的答案）
回答：[上面部/下面部]
"""
        max_retries = 3
        parsed_result = None # 初始化 parsed_result 为 None
        for retry_attempt in range(max_retries):
            response = self.api_request(prompt, image_path)

            if response is None:
                print(f"API request failed for image: {image_path} on retry {retry_attempt + 1}/{max_retries}")
                continue  # api_request already handles its own retries, but if it fully fails, we retry the whole process

            print(f"回复内容: {response[:150]}...")  # 只打印前150个字符

            current_parsed_result = self.parse_response(response) # Use a temporary variable
            if current_parsed_result is None: # 检查 parse_response 返回值，为 None 表示解析或校验失败
                print(f"  回复内容解析或数据校验失败，正在重试 ({retry_attempt + 1}/{max_retries})")
                continue # 解析或校验失败，重试
            else:
                print(f"  回复内容解析和数据校验成功")
                parsed_result = current_parsed_result # If successful, assign to parsed_result and break
                break # Break out of the retry loop on success

        if parsed_result: # Only return if parsed_result is not None (meaning success in at least one retry)
            parsed_result.update({"pic": image_path, "full_response": response})
            return parsed_result
        else: # If loop completes without success, return None (final failure)
            print(f"达到最大重试次数，处理图像失败: {image_path}")
            return None  # Return None if all retries failed


    def process_images_parallel(self, image_paths: List[str]) -> List[Dict]:
        """并行处理多张图像"""
        results = []
        # 每次开始新的批次前清空解析错误列表
        self.parsing_errors = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_path = {
                executor.submit(self.process_single_image, path): path
                for path in image_paths
            }

            # 使用tqdm显示进度
            for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(image_paths),
                              desc="处理图像"):
                path = future_to_path[future]
                try:
                    result = future.result()
                    if result is None: # result 为 None，表示该图片最终处理失败
                        print(f"  警告: 图像 {path} 处理最终失败，已达到最大重试次数")
                        self.parsing_errors.append({ # Only add to error report on final failure
                            "pic": os.path.basename(path),
                            "error_type": "达到最大重试次数后处理失败"
                        })
                    else:
                        results.append(result) # Only append successful results
                except Exception as e:
                    print(f"处理图像 {path} 时出错: {e}")
                    self.parsing_errors.append({ # Keep error reporting for exceptions
                        "pic": os.path.basename(path),
                        "error_type": f"处理异常: {e}"
                    })


        if self.parsing_errors: # If there are any final errors, print summary
            print(f"\n警告: 在处理过程中，有 {len(self.parsing_errors)} 张图片处理失败或解析校验失败。详细信息请查看错误报告。")

        return results


    def find_image_paths(self, image_folder: str) -> List[str]:
        """查找所有符合条件的图像路径"""
        image_paths = []
        image_pattern = re.compile(r"image_(\d+)\.jpg$")

        for filename in os.listdir(image_folder):
            if image_pattern.match(filename):
                image_path = os.path.join(image_folder, filename)
                image_paths.append(image_path)

        def get_image_number(filepath):
            match = image_pattern.match(os.path.basename(filepath))
            return int(match.group(1)) if match else 0

        image_paths.sort(key=get_image_number)

        return image_paths


def main():
    # 设置输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results_qwen_1")
    os.makedirs(output_dir, exist_ok=True)

    # 创建并行处理的分析器实例
    analyzer = EmotionAnalyzer(num_workers=1)  # 使用单线程处理，避免并发问题

    # 指定图像文件夹路径，使用和Gemma代码相同的路径
    image_folder = "/home/majc/image_block"

    # 获取所有图像路径
    all_image_paths = analyzer.find_image_paths(image_folder)
    print(f"找到 {len(all_image_paths)} 个图像待处理")

    # 批量生成结果文件
    for file_num in range(1, 42):  
        # 记录开始时间
        start_time = time.time()
        print(f"\n开始生成第 {file_num} 个结果文件")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 并行处理所有图像
        results = analyzer.process_images_parallel(all_image_paths)

        # 过滤掉 None 结果，防止空行
        valid_results = [r for r in results if r is not None]


        if valid_results: # 只有当有有效结果时才保存
            print(f"已处理完 {len(valid_results)} 个图像，正在保存结果...")

            # 同时保存为Excel格式
            df = pd.DataFrame([{
                'pic': os.path.basename(r['pic']),
                'is_coordinated': r['is_coordinated'],
                'coordination_level': r['coordination_level'],
                'emotion_polarity': r['emotion_polarity'],
                'emotion_type': r['emotion_type'],
                'judgment_basis': r['judgment_basis']
            } for r in valid_results])

            excel_output_file = os.path.join(output_dir, f"qwen_analysis_results_{file_num:02d}.xlsx")
            df.to_excel(excel_output_file, index=False)
            print(f"结果已保存到Excel文件: {excel_output_file}")

        else:
            print(f"批次 {file_num:02d} 没有成功处理任何图像，没有有效结果保存。")

        if analyzer.parsing_errors: # 如果有错误，保存错误报告 Excel
            error_df = pd.DataFrame(analyzer.parsing_errors)
            excel_error_file = os.path.join(output_dir, f"qwen_analysis_errors_{file_num:02d}.xlsx")
            error_df.to_excel(excel_error_file, index=False)
            print(f"错误报告已保存到Excel文件: {excel_error_file}")


    print(f"\n所有批次处理完成！共生成了 {file_num} 个结果文件")
    print(f"所有结果文件保存在目录: {output_dir}")


if __name__ == "__main__":
    main()
