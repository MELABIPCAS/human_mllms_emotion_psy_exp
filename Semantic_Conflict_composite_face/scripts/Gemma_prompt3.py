import os
import re
import time
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from PIL import Image

# Mac环境下不需要特别指定GPU，会自动使用可用资源
# 如果有多GPU配置，可以根据情况修改
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 如有需要再开启

class EmotionAnalyzer:
    def __init__(self, num_workers=4):
        # Gemma 模型配置
        torch.cuda.empty_cache()  # 释放显存缓存

        print("正在加载Gemma模型...")
        model_id = "/Users/wangsujing/programs/huggingface/google-gemma-3-27b-it"

        # 配置 4bit 量化
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        self.gemma_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            #load_in_8bit=True  # 启用 8-bit 量化
        ).eval()

        self.gemma_processor = AutoProcessor.from_pretrained(model_id)
        self.device = "mps" 
        print(f"模型加载完成，使用设备: {self.device}")

        # 初始化对话历史，为了兼容预置图片功能
        self.chat_history = []

        # ------ 预置图片和提示词模块 ------
        print("正在加载预置图片和提示词...")
        # 预置协调图片
        for i in range(1, 4):
            # 修改为Mac路径格式
            image_path = os.path.expanduser(f"/Users/wangsujing/programs/huggingface/Image_differentiation/coord_example{i}.png")
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
            # 修改为Mac路径格式
            image_path = os.path.expanduser(f"/Users/wangsujing/programs/huggingface/Image_differentiation/non_coord_example{i}.png")
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
        """使用Gemma模型进行本地推理"""
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
                inputs = self.gemma_processor.apply_chat_template(
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
                    generation = self.gemma_model.generate(
                        **inputs,
                        max_new_tokens=2048,  # 增加token数量
                        do_sample=True,  # 修改为False以获得确定性结果
                        temperature=1,
                    )
                    # 只提取新生成的部分
                    generation = generation[0][input_len:]

                # 解码回应
                response = self.gemma_processor.decode(generation, skip_special_tokens=True)

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
                traceback.print_exc()  # 打印详细错误信息

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
        for answer in self.valid_answers["is_coordinated"]:
            if answer in response:
                result["is_coordinated"] = answer
                break

        # 提取协调性等级
        level_pattern = re.compile(r'\b[1-5]\b')
        level_match = level_pattern.search(response)
        if level_match:
            result["coordination_level"] = level_match.group(0)

        # 提取情绪极性
        for answer in self.valid_answers["emotion_polarity"]:
            if answer in response:
                result["emotion_polarity"] = answer
                break

        # 提取情绪类型
        for answer in self.valid_answers["emotion_type"]:
            if answer in response:
                result["emotion_type"] = answer
                break

        # 提取判断依据
        for answer in self.valid_answers["judgment_basis"]:
            if answer in response:
                result["judgment_basis"] = answer
                break

        return result

    def process_single_image(self, image_path: str) -> Optional[Dict]:
        """处理单张图像的分析任务，包含重试机制"""
        prompt = """
请严格按照以下格式逐一回答问题：

1.图片中所表现出来的表情是否协调
你认为所呈现的面孔图片中的表情是否可能同时出现在一个人的面孔上，请用是否来进行回答（如无法进行判断，请说出一个你认为可能性较大的答案）
回答：[是/否]

2.图片中表情的协调/不协调性等级
你认为图片中表情的协调性等级是多少，请用1—5之间的数字来进行回答（例如，在上一个选择中你的答案为协调，则1代表"些许协调"，2代表"较为协调"，3代表"协调"，4代表"非常协调"，5代表"十分协调"；不协调如上）
回答：[1/2/3/4/5]

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
        max_retries = 5
        for retry_attempt in range(max_retries):
            response = self.api_request(prompt, image_path)

            if response is None:
                print(f"API request failed for image: {image_path} on retry {retry_attempt + 1}/{max_retries}")
                continue # api_request already handles its own retries, but if it fully fails, we retry the whole process

            #print(f"图像: {image_path}, 模型回复长度: {len(response)}")
            print(f"回复内容: {response[:100]}...")  # 只打印前100个字符

            parsed_result = self.parse_response(response)
            is_valid_format = True
            for key, value in parsed_result.items():
                if value is None:
                    is_valid_format = False
                    print(f"警告: 解析结果格式不正确，{key} 的值为 None，正在重试 ({retry_attempt + 1}/{max_retries})")
                    break
            if not is_valid_format:
                continue # retry if format is invalid

            parsed_result.update({"pic": image_path, "full_response": response})
            return parsed_result # return if format is valid

        print(f"达到最大重试次数，处理图像失败: {image_path}")
        return None # Return None if all retries failed


    def process_images_parallel(self, image_paths: List[str]) -> List[Dict]:
        """并行处理多张图像"""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_path = {
                executor.submit(self.process_single_image, path): path
                for path in image_paths
            }

            # 使用tqdm显示进度
            for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(image_paths), desc="处理图像"):
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


def main():
    # 创建并行处理的分析器实例
    analyzer = EmotionAnalyzer(num_workers=1)  # 使用单线程处理，避免并发问题

    # 修改为Mac路径格式
    image_folder = os.path.expanduser("/Users/wangsujing/programs/huggingface/Go2")  # 请替换成你的图像文件夹路径
    image_paths = analyzer.find_image_paths(image_folder)
    print(f"找到 {len(image_paths)} 个图像待处理")

    # 并行处理所有图像
    results = analyzer.process_images_parallel(image_paths)

    if results:
        print(f"已处理完 {len(results)} 个图像，结果已打印在控制台。")

        # 指定输出目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "results")
        os.makedirs(output_dir, exist_ok=True)

        # 将结果保存到文件
        output_file = os.path.join(output_dir, "gemma_analysis_results.txt")
        with open(output_file, "w", encoding="utf-8") as f:
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

        excel_output_file = os.path.join(output_dir, "gemma_analysis_results.xlsx")
        df.to_excel(excel_output_file, index=False)
        print(f"结果已保存到Excel文件: {excel_output_file}")
        print(f"所有结果文件保存在目录: {output_dir}")
    else:
        print("没有成功处理任何图像")


if __name__ == "__main__":
    main()