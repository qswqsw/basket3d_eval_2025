from collections import defaultdict
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO

# Load the model
model1 = YOLO('./best_final.pt')
model2= YOLO('basket_best.pt')
# Open the video file
video_path = "./4.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print("Video frame rate:", fps)


# Store the track history
track_history = defaultdict(lambda: [])
count1=0
count2=0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model1.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()

        # if fail to detect any object, skip this frame
        """ if(boxes.size() == 0):
            continue """

        w_values = boxes[:, 2].numpy()  # 获取所有框的宽度
        w_avg = np.mean(w_values)  # 计算宽度的平均值
        r = w_avg / 2  # 计算半径

        # track_ids = results[0].boxes.id.int().cpu().tolist()
        #track_ids=[1]
        track_id=1
       
        # Visualize the results on the frame
        annotated_frame = results[0].plot()


        #-----------------绘制篮筐检测框，以及篮筐相关信息---------------------------------#
        result2 = model2.track(frame, persist=True)[0]
        frame = result2.plot()
         # 仅保留类别为 "ball" 的框
        ball_boxes2 = [box for box in (result2.boxes) if box.cls == torch.tensor([1.])]
        for box2 in ball_boxes2:
        #篮筐左上角坐标和右下角坐标
            x1, y1, x2, y2 = box2.xyxy[0].cpu().numpy().astype(int)

    # Draw bounding box on frame2
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        text = "Count: " + str(count1)+'/'+str(count2)  # 文本内容为 "Count: " 加上整型变量 count 的字符串形式
        position = (10, 30)  # 文本位置，以左上角为基准点
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
        fontScale = 1  # 字体大小
        color = (0, 255, 0)  # 文本颜色，使用 BGR 格式
        thickness = 2  # 文本粗细
        cv2.putText(annotated_frame, text, position, font, fontScale, color, thickness)

        #-----------------篮筐部分结束---------------------------------#


        #-----------------绘制真实轨迹---------------------------------#
        if len(boxes)!=0:
            x, y, w, h = boxes[0]
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 100:  # retain 90 tracks for 90 frames
                track.pop(0)
        else:
            track = track_history[track_id]

            # Draw the tracking lines
        #points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        points=np.array(track, dtype=np.int32) 
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)
        #-----------------绘制真实轨迹结束---------------------------------#


        #-----------------绘制预测轨迹---------------------------------#
        track2 = track_history[track_id].copy()
        if len(track2)>10:
        
        # 转换坐标系
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            for i in range(len(track2)):
                x, y = track2[i]  # 解包元组
                track2[i] = (x, height - y)  # 更新第二个元素

            # 将坐标点分开为 x 和 y 值
            x_subset = np.array([point[0] for point in track2])
            y_subset = np.array([point[1] for point in track2])

            # 使用 polyfit 拟合抛物线
            coefficients = np.polyfit(x_subset, y_subset, 2)
            p = np.poly1d(coefficients)
            # 生成更多的 x 值，并绘制对应的抛物线
            x_additional = np.linspace(max(x_subset)-1,width, int(width-max(x_subset))*2)  # 生成更多的 x 值
            # 在当前帧上绘制预测的轨迹
            predicted_points = [(x,  height-(p(x))) for x in x_additional]  # 生成预测轨迹点的坐标
            predicted_points = np.array(predicted_points, dtype=np.int32)  # 将预测轨迹点转换为 NumPy 数组
            annotated_frame = cv2.polylines(annotated_frame, [predicted_points], isClosed=False, color=(0, 255, 0), thickness=3)
        #-----------------绘制预测轨迹结束---------------------------------#



        #-----------------显示当前帧---------------------------------#
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

#########################################################################################
# 示例数据
# track = [(x1, y1), (x2, y2), ...]

# 转换坐标系
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

for i in range(len(track)):
    x, y = track[i]  # 解包元组
    track[i] = (x, height - y)  # 更新第二个元素

print("输出track:")

# 遍历并输出所有值
for point in track:
    print(point)

print("输出结束")

# 提取第10个到第25个坐标数据
subset_track = track[9:23]  # 注意：Python中的索引是从0开始的，所以第10个元素对应索引为9，第25个元素对应索引为24

# 将坐标点分开为 x 和 y 值
x_subset = np.array([point[0] for point in subset_track])
y_subset = np.array([point[1] for point in subset_track])

# 使用 polyfit 拟合抛物线
coefficients = np.polyfit(x_subset, y_subset, 2)
p = np.poly1d(coefficients)

# 输出拟合的抛物线方程
print("拟合的抛物线方程：")
print(p)

# 绘制原始数据点和拟合的抛物线
plt.scatter(x_subset, y_subset, label='Data')  # 绘制原始数据点
plt.plot(x_subset, p(x_subset), color='red', label='Fit')  # 绘制拟合的抛物线

# 生成更多的 x 值，并绘制对应的抛物线
x_additional = np.linspace(min(x_subset), max(x_subset) + 400, 100)  # 生成更多的 x 值
plt.plot(x_additional, p(x_additional), linestyle='dashed', color='blue', label='Continuation')  # 绘制抛物线的延续部分

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Parabolic Fit')
plt.legend()
plt.show()


