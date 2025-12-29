import os
import re
import time
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

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
        # 预置协调图片
        for i in range(1, 4):
            # 使用和Gemma代码相同的路径
            image_path = f"/home/majc/Image differentiation/coord_example{i}.png"
            prompt = f"""这是协调的面部表情示例{i}，请记住这张图片的特征。"""
            if os.path.exists(image_path):  # 检查图片文件是否存在
                try:
                    image = Image.open(image_path).convert("RGB")
                    self.chat_history.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    })
                    # 添加模型回复，确保对话顺序为user/assistant交替
                    self.chat_history.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"好的，我已记住这张协调面部表情示例{i}的特征作为判断标准。"}]
                    })
                    print(f"预置协调图片加载成功: {image_path}")
                except Exception as e:
                    print(f"警告: 预置协调图片加载失败: {image_path}, 错误: {e}")
            else:
                print(f"警告: 预置协调图片文件未找到: {image_path}")

        # 预置不协调图片
        for i in range(1, 4):
            # 使用和Gemma代码相同的路径
            image_path = f"/home/majc/Image differentiation/non_coord_example{i}.png"
            prompt = f"""这是不协调的面部表情示例{i}，请记住这张图片的特征作为判断标准。"""
            if os.path.exists(image_path):  # 检查图片文件是否存在
                try:
                    image = Image.open(image_path).convert("RGB")
                    self.chat_history.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    })
                    # 添加模型回复，确保对话顺序为user/assistant交替
                    self.chat_history.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"好的，我已记住这张不协调面部表情示例{i}的特征作为判断标准。"}]
                    })
                    print(f"预置不协调图片加载成功: {image_path}")
                except Exception as e:
                    print(f"警告: 预置不协调图片加载失败: {image_path}, 错误: {e}")
            else:
                print(f"警告: 预置不协调图片文件未找到: {image_path}")

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
            "coordination_level": ["1", "2", "3", "4", "5"],
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

        # 添加系统消息
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": "以下为我们对于面部是否协调的判断标准示意图，请学习示意图中的判断标准并按照此判断标准对接下来任务中的问题：面部表情是否协调等问题进行回答。"}]
        })

        # 添加所有预设的对话
        for msg in self.preset_chat_history:
            # 对于包含图片的消息，需要复制一个新的内容列表
            new_content = []
            for item in msg["content"]:
                if item["type"] == "image" and not isinstance(item["image"], Image.Image):
                    # 如果图片是路径而不是PIL对象，则加载图片
                    loaded_img = self.load_image(item["image"])
                    if loaded_img:
                        new_content.append({"type": "image", "image": loaded_img})
                else:
                    # 直接使用原始内容
                    new_content.append(item)

            if new_content:  # 只有当内容不为空时才添加消息
                messages.append({
                    "role": msg["role"],
                    "content": new_content
                })

        # 添加当前问题
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

    def parse_response(self, response: str) -> Dict:
        """解析模型回复，提取各个问题的答案"""
        result = {
            "is_coordinated": None,
            "coordination_level": None,
            "emotion_polarity": None,
            "emotion_type": None,
            "judgment_basis": None
        }

        # 提取是否协调
        match = re.search(r"1\..*?\[(.*?)\]", response, re.DOTALL)
        if match:
            result["is_coordinated"] = match.group(1).strip()

        # 提取协调性等级
        match = re.search(r"2\..*?\[(.*?)\]", response, re.DOTALL)
        if match:
            result["coordination_level"] = match.group(1).strip()

        # 提取情绪极性
        match = re.search(r"3\..*?\[(.*?)\]", response, re.DOTALL)
        if match:
            result["emotion_polarity"] = match.group(1).strip()

        # 提取情绪类型
        match = re.search(r"4\..*?\[(.*?)\]", response, re.DOTALL)
        if match:
            result["emotion_type"] = match.group(1).strip()

        # 提取判断依据
        match = re.search(r"5\..*?\[(.*?)\]", response, re.DOTALL)
        if match:
            result["judgment_basis"] = match.group(1).strip()

        return result

    def process_single_image(self, image_path: str) -> Optional[Dict]:
        """处理单张图像的分析任务，包含重试机制"""
        prompt = """
请逐一回答以下问题：
1.你认为所呈现的面孔图片中的表情是否可能同时出现在一个人的面孔上，请用是否来进行回答（如无法进行判断，请说出一个你认为可能性较大的答案）
回答：[是/否]

2.你认为图片中人脸面部表情的协调性等级是多少，请用1—5之间的数字来进行回答（例如，在上一个选择中你的答案为协调，则1代表"些许协调"，2代表"较为协调"，3代表"协调"，4代表"非常协调"，5代表"十分协调"；不协调如上）
回答：[1/2/3/4/5]

3.你认为图片中的人脸的面部表情属于积极情绪还是消极情绪，请用积极情绪或消极情绪来进行回答（如无法进行判断，请说出一个你认为可能性较大的答案）
回答：[积极情绪/消极情绪]

4.你认为图片中的人脸的表情所表达的情绪属于高兴、惊讶、伤心、厌恶、害怕、愤怒中的哪一个？请用以上六个词中的一个词来进行回答（如无法进行判断，请说出一个你认为可能性较大的答案）
回答：[高兴/惊讶/伤心/厌恶/害怕/愤怒] 

5.你在判断图片中的人脸所表达的情绪时，是通过面部哪个区域进行判断的？请用上面部或下面部来进行回答（如无法进行判断，请说出一个你认为可能性较大的答案）
回答：[上面部/下面部]


"""
        max_retries = 3
        for retry_attempt in range(max_retries):
            response = self.api_request(prompt, image_path)

            if response is None:
                print(f"API request failed for image: {image_path} on retry {retry_attempt + 1}/{max_retries}")
                continue  # api_request already handles its own retries, but if it fully fails, we retry the whole process

            print(f"回复内容: {response[:150]}...")  # 只打印前150个字符

            parsed_result = self.parse_response(response)
            is_valid_format = True
            for key, value in parsed_result.items():
                if value is None:
                    is_valid_format = False
                    print(f"警告: 解析结果格式不正确，{key} 的值为 None，正在重试 ({retry_attempt + 1}/{max_retries})")
                    # 记录解析错误信息，用于后续生成错误报告
                    self.parsing_errors.append({
                        "pic": image_path,
                        "missing_key": key,
                        "full_response": response
                    })
                    break
            if not is_valid_format:
                continue  # retry if format is invalid

            parsed_result.update({"pic": image_path, "full_response": response})
            return parsed_result  # return if format is valid

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
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"处理图像 {path} 时出错: {e}")

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

    def save_error_report(self, batch_num: int, output_dir: str):
        """保存解析错误报告到Wrong文件"""
        if not self.parsing_errors:
            print(f"批次 {batch_num:02d} 没有检测到解析错误，不生成错误报告")
            return

        # 创建错误报告文件
        error_file = os.path.join(output_dir, f"Wrong{batch_num:02d}.txt")
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(f"批次 {batch_num:02d} 解析错误报告\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for i, error in enumerate(self.parsing_errors):
                f.write(f"错误 #{i + 1}:\n")
                f.write(f"图像路径: {error['pic']}\n")
                f.write(f"缺失键值: {error['missing_key']}\n")
                f.write(f"完整回复:\n{error['full_response']}\n")
                f.write("-" * 80 + "\n\n")

        print(f"已生成错误报告: {error_file}")


def main():
    # 设置输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results_qwen_3")
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

        # 生成错误报告（如果有解析错误）
        analyzer.save_error_report(file_num, output_dir)

        if results:
            print(f"已处理完 {len(results)} 个图像，正在保存结果...")

            # 将结果保存到文件
            output_file = os.path.join(output_dir, f"qwen_analysis_results_{file_num:02d}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"批次 {file_num:02d} 分析结果\n")
                f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")

                for result in results:
                    f.write(f"图像: {result['pic']}\n")
                    f.write(f"是否协调: {result['is_coordinated']}\n")
                    f.write(f"协调性等级: {result['coordination_level']}\n")
                    f.write(f"情绪极性: {result['emotion_polarity']}\n")
                    f.write(f"情绪类型: {result['emotion_type']}\n")
                    f.write(f"判断依据: {result['judgment_basis']}\n")
                    f.write(f"完整回复: {result['full_response']}\n")
                    f.write("-" * 80 + "\n")
            print(f"结果已保存到文件: {output_file}")

            # 同时保存为Excel格式
            import pandas as pd
            df = pd.DataFrame([{
                'pic': os.path.basename(r['pic']),
                'is_coordinated': r['is_coordinated'],
                'coordination_level': r['coordination_level'],
                'emotion_polarity': r['emotion_polarity'],
                'emotion_type': r['emotion_type'],
                'judgment_basis': r['judgment_basis']
            } for r in results])

            excel_output_file = os.path.join(output_dir, f"qwen_analysis_results_{file_num:02d}.xlsx")
            df.to_excel(excel_output_file, index=False)
            print(f"结果已保存到Excel文件: {excel_output_file}")

            # 记录结束时间和耗时
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"批次 {file_num:02d} 处理完成，耗时: {elapsed_time:.2f} 秒")
            print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"批次 {file_num:02d} 没有成功处理任何图像")

    print(f"\n所有批次处理完成！共生成了 {file_num} 个结果文件")
    print(f"所有结果文件保存在目录: {output_dir}")


if __name__ == "__main__":
    main()
