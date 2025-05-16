from collections import defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')

# Open the video file
video_path = "shoot1.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率和高度
fps = cap.get(cv2.CAP_PROP_FPS)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频的高度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Video frame rate:", fps)
print("图像高度：", height)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()

        # if fail to detect any object, skip this frame
        if boxes.shape[0] == 0:
            continue
        if results[0].boxes.id is None:
            continue

        w_values = boxes[:, 2].numpy()  # 获取所有框的宽度
        w_avg = np.mean(w_values)  # 计算宽度的平均值
        r = w_avg / 2  # 计算半径

        track_ids = [1]

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 100:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame
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

# 转换坐标系
for i in range(len(track)):
    x, y = track[i]  # 解包元组
    track[i] = (x, height - y)  # 更新第二个元素

# 提取第10个到第25个坐标数据
subset_track = track[9:88]  # 注意：Python中的索引是从0开始的

# 将坐标点分开为 x 和 y 值
x_data = np.array([point[0] for point in subset_track])
y_data = np.array([point[1] for point in subset_track])

# 使用样条插值进行拟合
tck = splrep(x_data, y_data, s=0)  # s=0表示确切地通过所有的数据点

# 生成插值点
x_interp = np.linspace(min(x_data), max(x_data), 500)
y_interp = splev(x_interp, tck)

# 计算最高点
max_height = np.max(y_interp)
max_height_index = np.argmax(y_interp)
max_height_coord = (x_interp[max_height_index], max_height)

# 计算斜率变化
derivatives = splev(x_interp, tck, der=1)

# 输出最高点和斜率变化
print(f"最高点的y值：{max_height:.2f}")
print(f"最高点的坐标：({max_height_coord[0]:.2f}, {max_height_coord[1]:.2f})")
print("斜率变化：", derivatives)

# 输出拟合结果
plt.figure(figsize=(12, 6))

# 原始数据和样条插值拟合曲线
plt.subplot(1, 2, 1)
plt.scatter(x_data, y_data, label='Data', color='blue')
plt.plot(x_interp, y_interp, color='red', label='Spline Fit')
plt.annotate(f"Max Height: ({max_height_coord[0]:.2f}, {max_height_coord[1]:.2f})", max_height_coord, textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Spline Interpolation Fit')
plt.legend()
plt.grid(True)

# 斜率变化曲线
plt.subplot(1, 2, 2)
plt.plot(x_interp, derivatives, color='green')
plt.xlabel('X')
plt.ylabel('Slope')
plt.title('Slope Change')
plt.grid(True)

plt.tight_layout()
plt.show()

# 打印插值点
print("插值点：")
for x, y in zip(x_interp, y_interp):
    print(f"({x:.2f}, {y:.2f})")
