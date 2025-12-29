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
    def __init__(self, api_key: str, num_workers=4):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model_name = "gpt-4o"

        self.num_workers = num_workers

        # 修改为一次性提问的提示词
        self.prompt = """
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

        self.max_retries = 10
        self.base_delay = 1
        self.max_delay = 30
        self.timeout = 120

        # 修改有效答案集合
        self.valid_answers = {
            "is_coordinated": ["是", "否"],
            "coordination_level": ["1", "2", "3", "4", "5", "6"],
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
            "is_coordinated": None,
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
            if answer in self.valid_answers["is_coordinated"]:
                result["is_coordinated"] = answer
            else:
                valid = False
                print(f"  校验失败: is_coordinated 的值为 '{answer}', 必须是 {self.valid_answers['is_coordinated']}")
        else:
            valid = False
            print(f"  解析失败: 未能提取 is_coordinated 的回答")

        # 提取协调性等级
        match = re.search(r"2[\.\s]*图片中表情的协调/不协调性等级.*?回答：\s*(?:\[)?([1-6])(?:\])?", text, re.DOTALL)
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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]
        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 500,
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

    def process_single_image(self, image_path: str) -> Dict:
        """处理单张图像的所有分析任务"""
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
        image_pairs = [{"experience": path} for path in image_paths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_pair = {
                executor.submit(self.process_single_image, pair["experience"]): pair
                for pair in image_pairs
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_pair), total=len(image_pairs), desc="处理图像"):
                pair = future_to_pair[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"处理图像 {pair['experience']} 时出错: {e}")

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
    api_key = "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"
    analyzer = EmotionAnalyzer(api_key, num_workers=8)

    image_folder = r"C:\Users\22688\Desktop\临时图片\赵琳\AU互斥+大模型\image_block"

    image_paths = analyzer.find_image_paths(image_folder)
    print(f"找到 {len(image_paths)} 个图像待处理")

    output_dir = r"C:\Users\22688\Desktop\临时图片\赵琳\gpt"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, 41):
        results = analyzer.process_images_parallel(image_paths)

        if results:
            df = pd.DataFrame(results)

            df['pic'] = df['pic'].apply(lambda x: os.path.basename(x))

            df = df[['pic', 'is_coordinated', 'coordination_level', 'emotion_polarity', 'emotion_type', 'judgment_basis']]

            excel_output_path = os.path.join(output_dir, f"combined_results{i:02d}.xlsx")
            df.to_excel(excel_output_path, index=False)
            print(f"表格已保存到: {excel_output_path}")
            print(f"共处理了 {len(results)} 个图像")
        else:
            print("没有成功处理任何图像")


if __name__ == "__main__":
    main()