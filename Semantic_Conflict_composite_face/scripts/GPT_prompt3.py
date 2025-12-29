# coding: utf-8
import os
import base64
import requests
import json
import pandas as pd
import re
import time
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Optional


class EmotionAnalyzer:
    def __init__(self, api_key: str, coordinated_examples: List[str], non_coordinated_examples: List[str],
                 num_workers=4):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # 协调和不协调的示例图片路径
        self.coordinated_examples = coordinated_examples
        self.non_coordinated_examples = non_coordinated_examples

        # OpenAI API配置
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model_name = "gpt-4o"

        # 并行处理配置
        self.num_workers = num_workers

        # 情绪类型映射 (保留以防后续使用)
        self.emotion_type_map = {
            "angry": "1",
            "disgusted": "2",
            "fearful": "3",
            "happy": "4",
            "sad": "5",
            "surprised": "6"
        }
        self.emotion_options = list(self.emotion_type_map.keys())

        # 创建示例图片的base64编码
        self.encoded_coord_examples = [self.encode_image(img) for img in self.coordinated_examples]
        self.encoded_non_coord_examples = [self.encode_image(img) for img in self.non_coordinated_examples]

        # 合并后的提示词
        self.prompt = """
判断标准：以下为我们对于面部是否协调的判断标准示意图，请学习示意图中的判断标准并忽略上面部与下面部之间的空白分割线以及图片的灰度分布不均匀和拼图问题等情况，将上、下面部结合起来视为完整的人脸，按照以上判断标准对接下来任务中的问题：面部表情是否协调进行回答

首先，我会展示协调的示例图片：
[协调示例图片1]
[协调示例图片2]
[协调示例图片3]

然后，我会展示不协调的示例图片：
[不协调示例图片1]
[不协调示例图片2]
[不协调示例图片3]

现在请判断以下图片：
请严格按照以下格式逐一回答问题：

1.图片中所表现出来的表情是否协调
你认为所呈现的面孔图片中的表情是否可能同时出现在一个人的面孔上，请用是否来进行回答（如无法进行判断，请说出一个你认为可能性较大的答案）
回答：[是/否]

2.图片中表情的协调/不协调性等级
你认为图片中表情的协调性等级是多少，请用1—5之间的数字来进行回答（例如，在上一个选择中你的答案为协调，则1代表“些许协调”，2代表“较为协调”，3代表“协调”，4代表“非常协调”，5代表“十分协调”；不协调如上）
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

        # 设置请求重试参数
        self.max_retries = 10
        self.base_delay = 1
        self.max_delay = 30
        self.timeout = 120

        # 定义各类答案的有效选项，用于纠错
        self.valid_answers = {
            "coordination_bool": ["是", "否"],
            "coordination_level": ["1", "2", "3", "4", "5"],
            "emotion_polarity": ["积极情绪", "消极情绪"],
            "emotion_type": ["高兴", "惊讶", "伤心", "厌恶", "害怕", "愤怒"],
            "judgment_basis": ["上面部", "下面部"]
        }

    def encode_image(self, image_path: str) -> Optional[str]:
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"图像编码错误: {e}")
            return None

    def parse_response(self, text: str) -> Optional[Dict]:
        """解析模型回复，提取各个问题的答案并进行严格校验"""
        result = {
            "coordination_bool": None,
            "coordination_level": None,
            "emotion_polarity": None,
            "emotion_type": None,
            "judgment_basis": None
        }
        valid = True

        # 提取是否协调
        match = re.search(r"1[\.\s]*图片中所表现出来的表情是否协调.*?回答：\s*(?:\[)?(是|否)(?:\])?", text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer in self.valid_answers["coordination_bool"]:
                result["coordination_bool"] = answer
            else:
                valid = False
                print(f"  校验失败: coordination_bool 的值为 '{answer}', 必须是 {self.valid_answers['coordination_bool']}")
        else:
            valid = False
            print(f"  解析失败: 未能提取 coordination_bool 的回答")

        # 提取协调性等级
        match = re.search(r"2[\.\s]*图片中表情的协调/不协调性等级.*?回答：\s*(?:\[)?([1-5])(?:\])?", text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer in self.valid_answers["coordination_level"]:
                result["coordination_level"] = answer
            else:
                valid = False
                print(f"  校验失败: coordination_level 的值为 '{answer}', 必须是 {self.valid_answers['coordination_level']}")
        else:
            valid = False
            print(f"  解析失败: 未能提取 coordination_level 的回答")

        # 提取情绪极性
        match = re.search(r"3[\.\s]*图片中的整体情绪类型.*?回答：\s*(?:\[)?(积极情绪|消极情绪)(?:\])?", text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer in self.valid_answers["emotion_polarity"]:
                result["emotion_polarity"] = answer
            else:
                valid = False
                print(f"  校验失败: emotion_polarity 的值为 '{answer}', 必须是 {self.valid_answers['emotion_polarity']}")
        else:
            valid = False
            print(f"  解析失败: 未能提取 emotion_polarity 的回答")

        # 提取情绪类型
        match = re.search(r"4[\.\s]*图片所表现的情绪.*?回答：\s*(?:\[)?(高兴|惊讶|伤心|厌恶|害怕|愤怒)(?:\])?", text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer in self.valid_answers["emotion_type"]:
                result["emotion_type"] = answer
            else:
                valid = False
                print(f"  校验失败: emotion_type 的值为 '{answer}', 必须是 {self.valid_answers['emotion_type']}")
        else:
            valid = False
            print(f"  解析失败: 未能提取 emotion_type 的回答")

        # 提取判断依据
        match = re.search(r"5[\.\s]*判断依据.*?回答：\s*(?:\[)?(上面部|下面部)(?:\])?", text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer in self.valid_answers["judgment_basis"]:
                result["judgment_basis"] = answer
            else:
                valid = False
                print(f"  校验失败: judgment_basis 的值为 '{answer}', 必须是 {self.valid_answers['judgment_basis']}")
        else:
            valid = False
            print(f"  解析失败: 未能提取 judgment_basis 的回答")

        if not valid:
            return None

        return result

    def api_request(self, encoded_image: str) -> Optional[str]:
        """封装API请求逻辑，包含验证和重试机制"""

        # 构建包含示例图片的消息内容
        content = []

        # 分割提示词，在适当的位置插入图片
        prompt_parts = self.prompt.split("[协调示例图片1]")
        if len(prompt_parts) > 1:
            # 添加文本开头部分
            content.append({"type": "text", "text": prompt_parts[0]})

            # 添加协调示例图片
            for i, encoded_example in enumerate(self.encoded_coord_examples):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_example}"
                    }
                })
                # 在图片之间添加分隔文本
                if i < len(self.encoded_coord_examples) - 1:
                    content.append({"type": "text", "text": f"[协调示例图片{i + 2}]"})

            # 添加协调和不协调示例之间的文本
            next_parts = self.prompt.split("[协调示例图片3]")[1].split("[不协调示例图片1]")
            if len(next_parts) > 1:
                content.append({"type": "text", "text": next_parts[0]})

                # 添加不协调示例图片
                for i, encoded_example in enumerate(self.encoded_non_coord_examples):
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_example}"
                        }
                    })
                    # 在图片之间添加分隔文本
                    if i < len(self.encoded_non_coord_examples) - 1:
                        content.append({"type": "text", "text": f"[不协调示例图片{i + 2}]"})

                # 添加最后的说明文本
                final_parts = self.prompt.split("[不协调示例图片3]")[1].split("现在请判断以下图片：")
                if len(final_parts) > 1:
                    content.append({"type": "text", "text": final_parts[0] + "现在请判断以下图片："})

                    # 添加待分析的图片
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    })

                    # 添加问题部分
                    content.append({"type": "text", "text": final_parts[1]})
        else:
            # 如果没有找到占位符，就使用简化的格式
            content = [
                {"type": "text", "text": "以下是协调的示例图片："}
            ]

            # 添加协调示例图片
            for encoded_example in self.encoded_coord_examples:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_example}"
                    }
                })

            content.append({"type": "text", "text": "以下是不协调的示例图片："})

            # 添加不协调示例图片
            for encoded_example in self.encoded_non_coord_examples:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_example}"
                    }
                })

            content.append({"type": "text", "text": "现在请判断以下图片："})

            # 添加待分析的图片
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            })

            # 添加问题
            content.append({"type": "text", "text": self.prompt.split("现在请判断以下图片：")[
                1] if "现在请判断以下图片：" in self.prompt else self.prompt})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 500,
            "top_p": 1,
            "temperature": 1,
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.api_url,
                    json=data,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    response_json = response.json()
                    full_response = response_json['choices'][0]['message']['content'].strip()
                    return full_response

                elif response.status_code in [500, 502, 503, 504]:
                    print(f"服务器错误({response.status_code})，第{attempt + 1}次重试...")
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    time.sleep(delay)
                elif response.status_code == 429:
                    print("请求过载，等待5秒后重试...")
                    time.sleep(5)
                else:
                    print(f"请求错误({response.status_code}): {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.base_delay)
                    else:
                        print("达到最大重试次数，放弃处理")
                        return None

            except requests.exceptions.RequestException as e:
                print(f"请求异常 (第 {attempt + 1} 次尝试): {e}")
                if attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    time.sleep(delay)
                else:
                    print("达到最大重试次数，放弃处理")
                    return None

        return None

    def process_single_image(self, image_path: str) -> Optional[Dict]:
        """处理单张图像的分析任务，包含重试机制和严格的结果校验"""
        encoded_image = self.encode_image(image_path)
        if encoded_image is None:
            return None

        max_retries = 3
        parsed_result = None

        for retry_attempt in range(max_retries):
            response = self.api_request(encoded_image)

            if response is None:
                print(f"API request failed for image: {image_path} on retry {retry_attempt + 1}/{max_retries}")
                continue

            print(f"回复内容: {response[:150]}...")

            current_parsed_result = self.parse_response(response)
            if current_parsed_result is None:
                print(f"  回复内容解析或数据校验失败，正在重试 ({retry_attempt + 1}/{max_retries})")
                continue
            else:
                print(f"  回复内容解析和数据校验成功")
                parsed_result = current_parsed_result
                break

        if parsed_result:
            parsed_result.update({"pic": image_path})
            return parsed_result
        else:
            print(f"达到最大重试次数，处理图像失败: {image_path}")
            return None

    def process_images_parallel(self, image_paths: List[str]) -> List[Dict]:
        """并行处理多张图像"""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_path = {
                executor.submit(self.process_single_image, path): path
                for path in image_paths
            }

            # 使用tqdm显示进度
            for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(image_paths),
                               desc="处理图像"):
                image_path = future_to_path[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"处理图像 {image_path} 时出错: {e}")

        return results

    def find_image_paths(self, image_folder: str) -> List[str]:
        """查找所有符合条件的图像路径"""
        image_paths = []
        # 修改正则表达式以匹配新的文件名格式，例如 image_1.jpg, image_2.jpg 等
        image_pattern = re.compile(r"image_(\d+)\.jpg$")

        for filename in os.listdir(image_folder):
            if image_pattern.match(filename):
                image_path = os.path.join(image_folder, filename)
                image_paths.append(image_path)

        # 确保图像路径按序号排序
        def get_image_number(filepath):
            match = image_pattern.match(os.path.basename(filepath))
            return int(match.group(1)) if match else 0

        image_paths.sort(key=get_image_number)

        return image_paths


def main():
    api_key = "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"

    # 示例图片路径设置
    examples_folder = r"C:\Users\22688\Desktop\赵琳第二问\Image differentiation"

    # 协调图片示例 (3张)
    coordinated_examples = [
        os.path.join(examples_folder, r"C:\Users\22688\Desktop\赵琳第二问\Image differentiation\coord_example1.png"),
        os.path.join(examples_folder, r"C:\Users\22688\Desktop\赵琳第二问\Image differentiation\coord_example2.png"),
        os.path.join(examples_folder, r"C:\Users\22688\Desktop\赵琳第二问\Image differentiation\coord_example3.png")
    ]

    # 不协调图片示例 (3张)
    non_coordinated_examples = [
        os.path.join(examples_folder, r"C:\Users\22688\Desktop\赵琳第二问\Image differentiation\non_coord_example1.png"),
        os.path.join(examples_folder, r"C:\Users\22688\Desktop\赵琳第二问\Image differentiation\non_coord_example2.png"),
        os.path.join(examples_folder, r"C:\Users\22688\Desktop\赵琳第二问\Image differentiation\non_coord_example3.png")
    ]

    # 创建并行处理的分析器实例，根据CPU和GPU情况调整工作线程数
    analyzer = EmotionAnalyzer(api_key, coordinated_examples, non_coordinated_examples, num_workers=8)

    # 待分析图片文件夹
    image_folder = r"C:\Users\22688\Desktop\临时图片\赵琳\AU互斥+大模型\image_block"

    image_paths = analyzer.find_image_paths(image_folder)
    print(f"找到 {len(image_paths)} 个图像待处理")

    # 输出文件夹
    output_dir = r"C:\Users\22688\Desktop\临时图片\赵琳\新提示词咯\GPT4o-2"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(25, 31):  # 循环3次，从3到5
        # 并行处理所有图像
        results = analyzer.process_images_parallel(image_paths)

        # 创建DataFrame并保存结果
        if results:
            df = pd.DataFrame(results)

            # 处理文件名
            df['pic'] = df['pic'].apply(lambda x: os.path.basename(x))

            # 重新排列列的顺序
            df = df[
                ['pic', 'coordination_bool', 'coordination_level', 'emotion_polarity', 'emotion_type',
                 'judgment_basis']]

            excel_output_path = os.path.join(output_dir, f"combined_results{i:02d}.xlsx")
            df.to_excel(excel_output_path, index=False)
            print(f"表格已保存到: {excel_output_path}")
            print(f"共处理了 {len(results)} 个图像")
        else:
            print("没有成功处理任何图像")


if __name__ == "__main__":
    main()