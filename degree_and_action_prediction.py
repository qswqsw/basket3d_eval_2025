import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

import tempfile

# 计算角度的函数
def angle_cal(ax, ay, bx, by, cx, cy):
    ax, ay, bx, by, cx, cy = map(int, [ax, ay, bx, by, cx, cy])
    radians = np.arctan2(cy - by, cx - bx) - np.arctan2(ay - by, ax - bx)
    angle = int(np.abs(radians * 180 / np.pi))
    if angle > 180.0:
        angle = 360 - angle
    return angle

from pathlib import Path
# 获取当前文件所在的目录
current_dir = Path(__file__).parent

# 构建模型路径（相对路径）
model_path = current_dir / 'best2.pt'
pose_model_path = current_dir / 'yolov8n-pose.pt'
# 实时角度及动作类型预测函数
def real_time_prediction(video_file):

     # 加载YOLO模型
    model = YOLO(str(model_path))
    model_2 = YOLO(str(pose_model_path))

    cap = cv2.VideoCapture(str(video_file))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
    cap.release()
   
    

# 根据宽高比计算显示的宽高
    st.header("上传视频")
    if width > height:
        # 横屏视频
        
        st.video(video_file)
    else:
        # 竖屏视频
        col1=st.columns(int(2))
        with col1[0]:
            st.video(video_file)

    


    
    # 在函数内部添加按钮
    if video_file and st.button("Execution", key="execution_button"):
            with st.spinner("Running..."):
                st.header("分析结果")

                # 创建两列布局
                col_1, col_2 = st.columns(2)

                # 创建一个空占位符，专门用于更新左列的视频帧
                with col_1:
                    frame_placeholder = st.empty()  # 创建一个空占位符，用于动态更新视频帧

                # 创建一个空占位符，专门用于更新右列的角度信息
                with col_2:
                    angle_placeholder = st.empty()  # 创建一个空占位符，用于动态更新右列角度数据
                try:
                    # 将上传的视频文件转为 OpenCV 可用的格式
                    video_bytes = video_file.read()

                    # 保存上传的字节流到临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                        tmp_video.write(video_bytes)  # 将上传的视频字节流写入临时文件
                        tmp_video_path = tmp_video.name  # 获取临时文件的路径
                    
                    video = cv2.VideoCapture(tmp_video_path)


                    frame_id = 0
                    detections = set()
                    pose_point = []

                    score = 0
                    miss = 0
                    check = 0
                    pending = 0
                    trigger = 0
                    countdown = 100
                    actual_traj = []
                    trajectory = []

                    # 获取视频的宽度、高度和fps
                    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = video.get(cv2.CAP_PROP_FPS)
                    import os
                    # 使用VideoWriter保存输出
                    video_filename = os.path.splitext(video_file.name)[0]  # 只保留文件名部分，不包括扩展名
                    output_file = os.path.join('action_pred_result', f'result_of_{video_filename}.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'X264')  # Codec for .mp4 format
                    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
                

                    # 创建一个空占位符，专门用于更新右列的角度信息
                

                    while True:
            #Read the video as frames
                        ret, frame = video.read()
                        if not ret:
                            break
                        if ret: 
                            # run the model of object detection on each frame
                            results = model.predict(frame)#,conf = 0.6)
                            # run the model of pose detection on each frame
                            results2 = model_2.predict(frame)
                            # plot the class boxes from object detection
                            annotated_frame = results[0].plot()
                            
                            #draw angles on the points
                            for result2 in results2:
                                # point indicates all the point locations to form a pose in tensor format
                                point = result2.keypoints.xy[0]
                                # this if statement is for a freeze bug occurs when no person detected in the frame. I solve it in the cheap way.(A running code is a good code, right?) 
                                if len(point) == 17:
                                    #this part is meant to input the point locations therefore I can use openCV's circle to draw it on the other model's frames.
                                    for a in range(0,16):
                                        pose_point.append([int(point[a][0]),int(point[a][1])])
                                    #point loactions necessary for angle calculation
                                    right_shoulder_x ,right_shoulder_y = point[6]
                                    right_elbow_x ,right_elbow_y = point[8]
                                    right_wrist_x ,right_wrist_y = point[10]
                                    left_shoulder_x ,left_shoulder_y = point[5]
                                    left_elbow_x ,left_elbow_y = point[7]
                                    left_wrist_x ,left_wrist_y = point[9]
                                    right_hip_x ,right_hip_y = point[12]
                                    right_knee_x ,right_knee_y = point[14]
                                    right_ankle_x ,right_ankle_y = point[16]
                                    left_hip_x ,left_hip_y = point[11]
                                    left_knee_x ,left_knee_y = point[13]
                                    left_ankle_x ,left_ankle_y = point[15]

                                    if all(num != 0 and not None for num in [int(right_shoulder_x),int(right_shoulder_y),int(left_shoulder_x),int(left_shoulder_y)]):
                                        cv2.line(annotated_frame,(int(right_shoulder_x),int(right_shoulder_y)),(int(left_shoulder_x),int(left_shoulder_y)) , (0, 255, 255), 5, -1)
                                    else:
                                        pass
                                    if all(num != 0 and not None for num in [int(right_shoulder_x),int(right_shoulder_y),int(right_elbow_x),int(right_elbow_y)]):
                                        cv2.line(annotated_frame,(int(right_shoulder_x),int(right_shoulder_y)),(int(right_elbow_x),int(right_elbow_y)) , (0, 255, 255), 5, -1)
                                    else:
                                        pass                
                                    if all(num != 0 and not None for num in [int(right_elbow_x),int(right_elbow_y),int(right_wrist_x),int(right_wrist_y)]):               
                                        cv2.line(annotated_frame,(int(right_elbow_x),int(right_elbow_y)),(int(right_wrist_x),int(right_wrist_y)) , (0, 255, 255), 5, -1)
                                    else:
                                        pass              
                                    if all(num != 0 and not None for num in [int(left_shoulder_x),int(left_shoulder_y),int(left_elbow_x),int(left_elbow_y)]):
                                        cv2.line(annotated_frame,(int(left_shoulder_x),int(left_shoulder_y)),(int(left_elbow_x),int(left_elbow_y)) , (0, 255, 255), 5, -1)
                                    else:
                                        pass              
                                    if all(num != 0 and not None for num in [int(left_elbow_x),int(left_elbow_y),int(left_wrist_x),int(left_wrist_y)]):
                                        cv2.line(annotated_frame,(int(left_elbow_x),int(left_elbow_y)),(int(left_wrist_x),int(left_wrist_y)) , (0, 255, 255), 5, -1)
                                    else:
                                        pass               
                                    if all(num != 0 and not None for num in [int(right_hip_x),int(right_hip_y),int(left_hip_x),int(left_hip_y)]):
                                        cv2.line(annotated_frame,(int(right_hip_x),int(right_hip_y)),(int(left_hip_x),int(left_hip_y)) , (0, 255, 255), 5, -1)
                                    else:
                                        pass                
                                    if all(num != 0 and not None for num in [int(right_hip_x),int(right_hip_y),int(right_knee_x),int(right_knee_y)]):
                                        cv2.line(annotated_frame,(int(right_hip_x),int(right_hip_y)),(int(right_knee_x),int(right_knee_y)) , (0, 255, 255), 5, -1)
                                    else:
                                        pass                
                                    if all(num != 0 and not None for num in [int(right_knee_x),int(right_knee_y),int(right_ankle_x),int(right_ankle_y)]):
                                        cv2.line(annotated_frame,(int(right_knee_x),int(right_knee_y)),(int(right_ankle_x),int(right_ankle_y)) , (0, 255, 255), 5, -1)
                                    else:
                                        pass                
                                    if all(num != 0 and not None for num in [int(left_hip_x),int(left_hip_y),int(left_knee_x),int(left_knee_y)]):
                                        cv2.line(annotated_frame,(int(left_hip_x),int(left_hip_y)),(int(left_knee_x),int(left_knee_y)) , (0, 255, 255), 5, -1)
                                    else:
                                        pass                
                                    if all(num != 0 and not None for num in [int(left_knee_x),int(left_knee_y),int(left_ankle_x),int(left_ankle_y)]):
                                        cv2.line(annotated_frame,(int(left_knee_x),int(left_knee_y)),(int(left_ankle_x),int(left_ankle_y)) , (0, 255, 255), 5, -1)
                                    else:
                                        pass                
                                    if all(num != 0 and not None for num in [int(right_hip_x),int(right_hip_y),int(right_shoulder_x),int(right_shoulder_y)]):
                                        cv2.line(annotated_frame,(int(right_hip_x),int(right_hip_y)),(int(right_shoulder_x),int(right_shoulder_y)) , (0, 255, 255), 5, -1)
                                    else:
                                        pass
                                    if all(num != 0 and not None for num in [int(left_hip_x),int(left_hip_y),int(left_shoulder_x),int(left_shoulder_y)]):
                                        cv2.line(annotated_frame,(int(left_hip_x),int(left_hip_y)),(int(left_shoulder_x),int(left_shoulder_y)) , (0, 255, 255), 5, -1)
                                    else:
                                        pass
                                #print('x_s:',right_shoulder_x,'y_s:',right_shoulder_y)

                            # angle calculation using predefined equation
                            r_elbow_angle = angle_cal(right_shoulder_x ,right_shoulder_y,right_elbow_x ,right_elbow_y,right_wrist_x ,right_wrist_y)
                            l_elbow_angle = angle_cal(left_shoulder_x ,left_shoulder_y,left_elbow_x ,left_elbow_y,left_wrist_x ,left_wrist_y)
                            r_knee_angle = angle_cal(right_hip_x ,right_hip_y,right_knee_x ,right_knee_y,right_ankle_x ,right_ankle_y)
                            l_knee_angle = angle_cal(left_hip_x ,left_hip_y,left_knee_x ,left_knee_y,left_ankle_x ,left_ankle_y)


                            # draw the points
                            for pointa in pose_point:
                                cv2.circle(annotated_frame, pointa, 5, (0, 255, 255), -1)
                            
                                


                            # this for loop is to draw the trajectory of the ball and record frames where I was in shooting motion
                            for result in results:
                                result_cls = result.boxes.cls
                                for i in result_cls:
                                    i = int(i)
                                # trajectory of the ball, 3 as in 'shooting' motion
                                if 3 in result_cls:
                                    for detection in result.boxes:
                                        x1,y1,x2,y2 = detection.xyxy[0]
                                        x_center = int((x1+x2)//2)
                                        y_center = int((y1+y2)//2)
                                        cls = int(detection.cls[0]) 
                                        if cls == 0:
                                            trajectory.append([x_center,y_center])
                                            actual_x = x_center-int(right_ankle_x)
                                            actual_y = int(right_ankle_y)-y_center
                                            actual_traj.append([actual_x,actual_y])
                                    frame_id += 1
                                # count number of frames where I was in shooting motion, 5 as in 'after shooting' phase where ball released from the ball    
                                if 5 in result_cls:
                                    check = 1
                                    time = round(frame_id/60,2)
                                    text = f'This shot takes {time} secs'
                                    cv2.putText(annotated_frame, text, (0,50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                                    #if time >2:
                                        #cv2.putText(annotated_frame, 'Too Slow!', (500,120), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                                    #elif 1<= time <=2:
                                        #cv2.putText(annotated_frame, 'Continue working on your shooting speed', (500,120), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                                    #else:
                                        #cv2.putText(annotated_frame, 'Fast Release!', (500,120), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                                # 8 as in 'pending' phase where the ball is near the rim, then I set a trigger to count down 100 frames (1.66 secs)
                                if 8 in result_cls and check == 1:
                                    pending = 1 
                                    trigger = 1
                                if trigger == 1:
                                    countdown -= 1
                                # 7 as in score, if a score box appears, then the score += 1, every other trigger reset to prevent score appearing in further frames to take into count.    
                                if 7 in result_cls and pending == 1 and countdown >0:
                                    score += 1
                                    check = 0
                                    pending = 0
                                    trigger = 0
                                    countdown = 100
                                    frame_id = 0
                                # if score did not appear, and countdown turns to 0, then miss += 1, reset every other triggers
                                if countdown == 0:
                                    miss += 1
                                    pending = 0 
                                    check = 0
                                    trigger = 0
                                    countdown = 100
                                    frame_id = 0
                            # show score, miss and accuracy in percentage
                            if score >0 or miss > 0:        
                                score_text = f'Current accuracy: {round(score/(score + miss),4)*100} %   shot made:{score}   shot miss:{miss}'
                                cv2.putText(annotated_frame, score_text, (0,100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 1, cv2.LINE_AA)        
                                
                                    


                            #Draw the points
                            #for pointb in trajectory:
                                #cv2.circle(annotated_frame, pointb, 5, (0, 255, 255), -1)
                            
                            # show the angles of elbow and knee
                            cv2.putText(annotated_frame,str(r_elbow_angle),(int(right_elbow_x),int(right_elbow_y)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
                            cv2.putText(annotated_frame,str(l_elbow_angle),(int(left_elbow_x),int(left_elbow_y)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
                            cv2.putText(annotated_frame,str(r_knee_angle),(int(right_knee_x),int(right_knee_y)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
                            cv2.putText(annotated_frame,str(l_knee_angle),(int(left_knee_x),int(left_knee_y)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

                    # 在右列显示一些信息或其他内容
                    

                        # 将每一帧保存到视频文件
                            
                            out.write(annotated_frame)
        
                        #reset the pose points before it goes into the next frame
                            pose_point = []
                        # show the video
                            #frame_placeholder.image(annotated_frame)
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                            # 在 Streamlit 中显示图像
                            frame_placeholder.image(annotated_frame_rgb)
                            angle_placeholder.markdown(f"""
                                <p style="font-size: 36px; color: black;">Right Elbow Angle: <strong>{r_elbow_angle:.2f}</strong></p>
                                <p style="font-size: 36px; color: black;">Left Elbow Angle: <strong>{l_elbow_angle:.2f}</strong></p>
                                <p style="font-size: 36px; color: black;">Right Knee Angle: <strong>{r_knee_angle:.2f}</strong></p>
                                <p style="font-size: 36px; color: black;">Left Knee Angle: <strong>{l_knee_angle:.2f}</strong></p>
                            """, unsafe_allow_html=True)

                    # 释放视频
                    video.release()  # 释放输入视频资源
                    out.release()  # 
                    frame_placeholder.empty
                    frame_placeholder.video(output_file)
                except Exception as e:
                    st.error(f"Error loading video: {e}")
