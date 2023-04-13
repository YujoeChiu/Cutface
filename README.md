# Cutface
import os
import cv2
import mediapipe as mp
import numpy as np
import shutil

# 初始化人脸检测和绘制模块
mp_face_detection = mp.solutions.face_detection

# 定义输入图像文件夹和输出文件夹
input_folder = 'C:\\Users\\18661\\Desktop\\program\\noface\\sample'
output_folder = 'C:\\Users\\18661\\Desktop\\program\\noface\\sample_out'
unrecognized_folder = 'path/to/unrecognized/folder'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(unrecognized_folder, exist_ok=True)

# 读取文件夹中的所有图像文件
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 对每张图像进行处理
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print("存在部分无法读取图像")
        shutil.copy(image_path, os.path.join(unrecognized_folder, image_file))
        continue

    # 将图像从 BGR 转换为 RGB，因为 MediaPipe 使用的是 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用 MediaPipe 进行人脸检测
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        # 运行人脸检测
        results = face_detection.process(image)

        # 填充检测到的人脸区域
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                # 定义填充区域的顶点
                pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
                # 使用白色填充区域
                cv2.fillPoly(image, [pts], color=(255, 255, 255))

    # 将图像从 RGB 转换回 BGR，以便于使用 OpenCV 进行保存
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 保存处理后的图像
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, image)

print("批量处理完成！")
