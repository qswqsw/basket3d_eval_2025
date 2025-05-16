from collections import defaultdict
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from collections import defaultdict
from PIL import Image
import streamlit as st
import os
import tempfile

from ultralytics import YOLO

def Basket_Whole(video_file):
    model1 = YOLO('best.pt')
    model2 = YOLO('basket_best.pt')
    st.header("上传视频")

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        vid_cap = cv2.VideoCapture(tfile.name)
        height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        tfile.close()
        if width > height:
        # 横屏视频
        
            st.video(video_file)
        else:
            # 竖屏视频
            col1=st.columns(2)
            with col1[0]:
                st.video(video_file)
        video_filename = os.path.splitext(video_file.name)[0]  # 只保留文件名部分，不包括扩展名
        output_file = os.path.join('ball_trace_pred_result', f'result_of_{video_filename}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'X264')  # Codec for .mp4 format
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))


   
    if video_file:
        if st.button("Execution"):
            with st.spinner("Running..."):
                st.header("分析结果")
                try:
                    st.write("Video frame rate:", fps)
                    col_1, col_2 = st.columns(2)

                # 创建一个空占位符，专门用于更新左列的视频帧
                    with col_1:
                        st_frame = st.empty()
                    # Store the track history
                    track_history = defaultdict(lambda: [])
                    count1 = 0
                    count2 = 0

                    if not vid_cap.isOpened():
                        st.write("Error: Could not open video.")
                        return
                    
                    

                    while(vid_cap.isOpened()):
                        success, frame = vid_cap.read()
                        if success:
                            # Run YOLOv8 tracking on the frame, persisting tracks between frames
                            results = model1.track(frame, persist=True)
                            # Get the boxes and track IDs
                            boxes = results[0].boxes.xywh.cpu()

                            w_values = boxes[:, 2].numpy()  # 获取所有框的宽度
                            w_avg = np.mean(w_values)  # 计算宽度的平均值
                            r = w_avg / 2  # 计算半径

                            # track_ids = results[0].boxes.id.int().cpu().tolist()
                            # track_ids=[1]
                            track_id = 1

                            # Visualize the results on the frame
                            annotated_frame = results[0].plot()

                            # -----------------绘制篮筐检测框，以及篮筐相关信息---------------------------------#
                            result2 = model2.track(frame, persist=True)[0]
                            frame = result2.plot()
                            # 仅保留类别为 "ball" 的框
                            ball_boxes2 = [box for box in (result2.boxes) if box.cls == torch.tensor([1.])]
                            for box2 in ball_boxes2:
                                # 篮筐左上角坐标和右下角坐标
                                x1, y1, x2, y2 = box2.xyxy[0].cpu().numpy().astype(int)

                            # Draw bounding box on frame2
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            text = "Count: " + str(count1) + '/' + str(count2)  # 文本内容为 "Count: " 加上整型变量 count 的字符串形式
                            position = (10, 30)  # 文本位置，以左上角为基准点
                            font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
                            fontScale = 1  # 字体大小
                            color = (0, 255, 0)  # 文本颜色，使用 BGR 格式
                            thickness = 2  # 文本粗细
                            cv2.putText(annotated_frame, text, position, font, fontScale, color, thickness)

                            # -----------------篮筐部分结束---------------------------------#

                            # -----------------绘制真实轨迹---------------------------------#
                            if len(boxes) != 0:
                                x, y, w, h = boxes[0]
                                track = track_history[track_id]
                                track.append((float(x), float(y)))  # x, y center point
                                if len(track) > 100:  # retain 90 tracks for 90 frames
                                    track.pop(0)
                            else:
                                track = track_history[track_id]

                                # Draw the tracking lines
                            # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            points = np.array(track, dtype=np.int32)
                            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)
                            # -----------------绘制真实轨迹结束---------------------------------#

                            # -----------------绘制预测轨迹---------------------------------#
                            track2 = track_history[track_id].copy()
                            if len(track2) > 10:

                            # 转换坐标系
                                height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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
                                x_additional = np.linspace(max(x_subset) - 1, width,
                                                           int(width - max(x_subset)) * 2)  # 生成更多的 x 值
                                # 在当前帧上绘制预测的轨迹
                                predicted_points = [(x, height - (p(x))) for x in x_additional]  # 生成预测轨迹点的坐标
                                predicted_points = np.array(predicted_points, dtype=np.int32)  # 将预测轨迹点转换为 NumPy 数组
                                annotated_frame = cv2.polylines(annotated_frame, [predicted_points], isClosed=False,
                                                                color=(0, 255, 0), thickness=3)
                            # -----------------绘制预测轨迹结束---------------------------------#

                            # -----------------显示当前帧---------------------------------#
                            pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                            st_frame.image(pil_image, caption="Tracking Result")
                            out.write(annotated_frame)

                        else:
                            vid_cap.release()
                            break
                    vid_cap.release()  # 释放输入视频资源
                    out.release()  # 
                    st_frame.empty
                    st_frame.video(output_file)


                    
                    # 转换坐标系
                    for i in range(len(track)):
                        x, y = track[i]  # 解包元组
                        track[i] = (x, height - y)  # 更新第二个元素

                    # 提取第10个到第25个坐标数据
                    subset_track = track[9:50]  # 注意：Python中的索引是从0开始的

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
                    
                    

                    # 输出拟合结果
                    fig1=plt.figure(figsize=(12, 6))

                    # 原始数据和样条插值拟合曲线
                    plt.scatter(x_data, y_data, label='Data', color='blue')
                    plt.plot(x_interp, y_interp, color='red', label='Spline Fit')
                    plt.annotate(f"Max Height: ({max_height_coord[0]:.2f}, {max_height_coord[1]:.2f})",
                                         max_height_coord, textcoords="offset points", xytext=(0, 10), ha='center',
                                         fontsize=10)
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.title('Spline Interpolation Fit')
                    plt.legend()
                    plt.grid(True)

                    fig1_path ='ball_trace_pred_result/'+ video_filename+'_spline_fit.png'
                    fig1.savefig(fig1_path, format='png')
                    
                    

                    # 斜率变化曲线
                    fig2 = plt.figure(figsize=(12, 6))
                    plt.plot(x_interp, derivatives, color='green')
                    plt.xlabel('X')
                    plt.ylabel('Slope')
                    plt.title('Slope Change')
                    plt.grid(True)
                    fig2_path = 'ball_trace_pred_result/'+video_filename+'_slope_change.png'
                    fig2.savefig(fig2_path, format='png')
                  


                 
                    col_1, col_2 = st.columns(2)

                    # 在第一个列中显示标题和图像
                 
                    

                    # 在第二个列中显示标题和图像
                    
                    
                    with col_1:
                        st.header("样条拟合曲线")  # 为第一个图像添加标题
                        st.image(fig1_path)
                    with col_2:
                        st.header("斜率变化曲线")  # 为第二个图像添加标题
                        st.image(fig2_path)

                    
                    

                except Exception as e:
                    st.error(f"Error loading video: {e}")


