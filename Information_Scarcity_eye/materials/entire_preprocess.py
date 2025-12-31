import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def put_chinese_text(img, text, position, text_color=(0, 255, 0), font_size=30):
    """在图片上添加中文文字"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("simhei.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    draw.text(position, text, text_color, font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def process_image(input_image, output_image):
    """处理图片，添加人脸网格并裁剪人脸"""
    # 读取图片
    frame = cv2.imread(input_image)
    if frame is None:
        print(f"无法打开图片: {input_image}")
        return

    # 初始化人脸网格检测器
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,  # 设置为静态图片模式
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    try:
        # 转换颜色空间并处理
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # 绘制人脸网格
                h, w, _ = frame.shape
                
                # 计算人脸边界框
                x_coords = [int(landmark.x * w) for landmark in landmarks.landmark]
                y_coords = [int(landmark.y * h) for landmark in landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # 扩大裁剪区域（添加边距）
                margin = 20
                x_min = max(0, x_min - margin)
                # 向上增加更多边距以包含额头
                y_min = max(0, y_min - margin * 2)  # 将向上的边距增加一倍
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
                
                # 获取151和195特征点的y坐标
                point_151_y = int(landmarks.landmark[151].y * h)
                point_195_y = int(landmarks.landmark[195].y * h)
                point_8_y = int(landmarks.landmark[8].y * h)  # 添加特征点8的y坐标
                
                # 首先绘制所有特征点
                for landmark in landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    # 绘制绿色圆点，大小为2像素
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # 然后绘制绿色网格线
                for connection in mp.solutions.face_mesh.FACEMESH_TESSELATION:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    # 获取连接点的坐标
                    start_point = landmarks.landmark[start_idx]
                    end_point = landmarks.landmark[end_idx]
                    
                    # 转换坐标到图像空间
                    start_x = int(start_point.x * w)
                    start_y = int(start_point.y * h)
                    end_x = int(end_point.x * w)
                    end_y = int(end_point.y * h)
                    
                    # 绘制绿色线条
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 1)
                
                # 最后绘制关键点，确保显示在最上层
                point_151_x = int(landmarks.landmark[151].x * w)
                point_151_y = int(landmarks.landmark[151].y * h)
                point_195_x = int(landmarks.landmark[195].x * w)
                point_195_y = int(landmarks.landmark[195].y * h)
                point_8_x = int(landmarks.landmark[8].x * w)
                point_8_y = int(landmarks.landmark[8].y * h)
                point_197_x = int(landmarks.landmark[197].x * w)
                point_197_y = int(landmarks.landmark[197].y * h)

                # 用不同颜色绘制这四个关键点，大小为5像素
                # 151和195为黄色，8和197为粉色
                cv2.circle(frame, (point_151_x, point_151_y), 5, (0, 255, 255), -1)  # 黄色
                cv2.circle(frame, (point_195_x, point_195_y), 5, (0, 255, 255), -1)  # 黄色
                cv2.circle(frame, (point_8_x, point_8_y), 5, (255, 0, 255), -1)      # 粉色
                cv2.circle(frame, (point_197_x, point_197_y), 5, (255, 0, 255), -1)  # 粉色
                
                # 裁剪人脸区域
                face_crop = frame[y_min:y_max, x_min:x_max]
                
                # 保存原始图片（带网格）
                cv2.imwrite(output_image, frame)
                
                # 保存裁剪后的人脸图片
                output_crop = output_image.rsplit('.', 1)[0] + '_crop.jpg'
                cv2.imwrite(output_crop, face_crop)
                
                # 使用151和195特征点进行裁剪
                feature_crop_151_195 = frame[point_151_y:point_195_y, x_min:x_max]
                output_feature_151_195 = output_image.rsplit('.', 1)[0] + '_feature_151_195.jpg'
                cv2.imwrite(output_feature_151_195, feature_crop_151_195)
                
                # 使用8和197特征点进行裁剪
                feature_crop_8_197 = frame[point_8_y:point_197_y, x_min:x_max]
                output_feature_8_197 = output_image.rsplit('.', 1)[0] + '_feature_8_197.jpg'
                cv2.imwrite(output_feature_8_197, feature_crop_8_197)
                
                print(f"特征点151-195裁剪已保存到: {output_feature_151_195}")
                print(f"特征点8-197裁剪已保存到: {output_feature_8_197}")

    finally:
        face_mesh.close()

if __name__ == "__main__":
    # 配置输入输出路径
    input_image = r"C:\Users\Administrator\Desktop\Rafd090_14_Caucasian_female_happy_frontal.jpg"  # 替换为您的输入图片路径
    output_image = r"C:\Users\Administrator\Desktop\\aa.jpg"  # 替换为您想要的输出路径

    # 处理图片
    process_image(input_image, output_image)
