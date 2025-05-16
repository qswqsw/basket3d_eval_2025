#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam
from keypoint2 import process_whole
from keypoint_final2 import Basket_Whole
from degree_and_action_prediction import real_time_prediction
from basketball_pose_analysis import basketball_pose_analysis_page  # 引入篮球姿态分析页面

from constants import *
import AnalyzerModule as am
import os
import cv2
import tempfile
# setting page layout
st.set_page_config(
    page_title="投篮分析系统",
    page_icon="★",
    layout="wide",
    initial_sidebar_state="expanded" 
    )

# 主页面背景 https://images.pexels.com/photos/772803/pexels-photo-772803.jpeg?cs=srgb&dl=pexels-tyler-lastovich-772803.jpg&fm=jpg&_gl=1*11wrzp4*_ga*MTAzMjE1MjE0NC4xNzAyOTY3ODI3*_ga_8JE65Q40S6*MTcxMjA0MjIzNy4yLjEuMTcxMjA0MjQ4OS4wLjAuMA..
def main_bg():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url('https://images.pexels.com/photos/772803/pexels-photo-772803.jpeg?cs=srgb&dl=pexels-tyler-lastovich-772803.jpg&fm=jpg&_gl=1*11wrzp4*_ga*MTAzMjE1MjE0NC4xNzAyOTY3ODI3*_ga_8JE65Q40S6*MTcxMjA0MjIzNy4yLjEuMTcxMjA0MjQ4OS4wLjAuMA..');
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

def set_background_color(color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
            height: 100vh;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# 调用主页面背景




# main page heading
st.title("Welcome to our group presentation.")

# sidebar
st.sidebar.header("模型配置选择")

# model options
task_type = st.sidebar.selectbox(
    "任务类别选择",
    ["项目功能展示","检测", "分割", "关键点","篮球检测","投篮计数与轨迹预测","实时角度及动作类型预测","投篮姿态分析","肌肉激活分析"],
)

model_type = None
if task_type == "检测":
    main_bg()
    model_type = st.sidebar.selectbox(
        "选择模型",
        config.DETECTION_MODEL_LIST
    )
elif task_type == "分割":
    main_bg()
    model_type = st.sidebar.selectbox(
        "选择模型",
        config.SEGMENT_MODEL_LIST
    )
elif task_type == "关键点":
    main_bg()
    model_type = st.sidebar.selectbox(
        "选择模型",
        config.POSE_MODEL_LIST
    )
elif task_type =="篮球检测":
    main_bg()
    model_type = st.sidebar.selectbox(
        "选择模型",
        config.BASKERBALL_MODEL_LIST
    )
elif task_type == "投篮计数与轨迹预测":
    main_bg()
    st.sidebar.write("此模型为固定模型")
elif task_type == "实时角度及动作类型预测":
    main_bg()
    st.sidebar.write("此模型为固定模型")
elif task_type == "投篮姿态分析":
    set_background_color("lightgray")
    st.sidebar.write("此模型为固定模型")

elif task_type == "肌肉激活分析":
    main_bg()
    st.sidebar.write("此模型为固定模型")

elif task_type == "项目功能展示":
    main_bg()
    st.sidebar.header("此为本项目的基本样例展示")
else:
    st.error("Currently only 'Detection' function is implemented")


if task_type == "项目功能展示":

    st.subheader("处理前后对比：")
    col1,col2 = st.columns(2)
    with col1:
        st.video("hy_before.mp4")
    with col2:
        st.video("hy_after.mp4")
elif task_type == "投篮计数与轨迹预测":
    st.sidebar.write("选择您想要绘制的视频")
    video_file = st.sidebar.file_uploader("choose  video", type=['mp4', 'avi', 'mov', 'mkv'])
    if video_file is not None:
        Basket_Whole(video_file)
    else:
        st.sidebar.warning("请上传一个视频文件")

elif task_type == "实时角度及动作类型预测":
    st.sidebar.write("选择您想要绘制的视频")
    video_file = st.sidebar.file_uploader("choose  video", type=['mp4', 'avi', 'mov', 'mkv'])
    if video_file is not None:
       real_time_prediction(video_file)
    else:
        st.sidebar.warning("请上传一个视频文件")


elif task_type == "肌肉激活分析":
    # 视频上传
    st.title("肌肉激活分析")
    st.sidebar.title("上传视频")

    video_file = st.sidebar.file_uploader("选择视频文件", type=["mp4", "mov", "avi"])


    # 设置分析按钮
    if video_file:
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())  # 将文件内容写入临时文件
            temp_video_path = tmp_file.name  # 获取临时文件的路径
    
    # 使用 OpenCV 打开视频文件
        cap = cv2.VideoCapture(temp_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
        cap.release()
        os.remove(temp_video_path)
        
    
    # 根据宽高比计算显示的宽高
        if width > height:
            # 横屏视频
            
            st.video(video_file)
        else:
            # 竖屏视频
            col1=st.columns(int(2))
            with col1[0]:
                st.video(video_file)

    # 显示视频
        
        #st.video(video_file)  # 显示上传的视频
        analyze_button = st.button("开始分析")

    # 处理视频并分析
    if video_file and analyze_button:
        # 将上传的视频写入临时文件
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

       

        # 假设你传入的关节是右膝和右臀
        joints = [KNEE_RIGHT, HIP_RIGHT]
        joints = [am.SHOULDER_RIGHT,am.HIP_RIGHT,am.KNEE_RIGHT,am.ANKLE_RIGHT,am.ELBOW_RIGHT]
        limbs = [am.ARM_LOWER_RIGHT,am.ARM_UPPER_RIGHT,am.UPPER_BODY_RIGHT,am.LEG_UPPER_RIGHT,am.LEG_LOWER_RIGHT, am.FOOT_RIGHT]
        name=os.path.splitext(video_file.name)[0]
        
        #path = 'result_musle_activate/' + video_file.name + '.MOV'
        am.pipeline(path = temp_video_path, output_name = name, joints=joints,limbs=limbs, out_frame_rate=12)

        # 生成分析的输出
       

        # 显示分析图表
       
        col1, col2 = st.columns(2)

        # 在左列显示图表
        with col1:
            st.subheader("分析图表")
            chart_path = 'result_musle_activate/' + name + '.png'
            st.image(chart_path, caption="分析结果图", use_container_width =True)

        # 在右列显示建议
        with col2:
            st.subheader("运动建议")
            suggestions = "这里是运动建议的文本内容，可以根据分析结果进行动态展示。"
            st.write(suggestions)

        # 生成分析后的视频
        output_video_path = 'result_musle_activate/'+name+'.mp4'
    

        # 显示分析后的视频
        st.subheader("分析后的视频")
        if os.path.exists(output_video_path):
            cap = cv2.VideoCapture(output_video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
            if width > height:
            # 横屏视频
            
                st.video(output_video_path)
            else:
                # 竖屏视频
                col1=st.columns(int(2))
                with col1[0]:
                    st.video(output_video_path)
                       
        else:
            st.error("视频文件未生成或路径错误。")

       
        
        

elif task_type == "投篮姿态分析":
    
    basketball_pose_analysis_page()  # 调用篮球姿态分析页面函数
else:
    confidence = float(st.sidebar.slider(
        "置信度", 30, 100, 50)) / 100

    model_path = ""
    if model_type:

        if task_type == "检测":
            model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
        elif task_type == "分割":
            model_path = Path(config.SEGMENT_MODEL_DIR, str(model_type))
        elif task_type == "关键点":
            model_path = Path(config.POSE_MODEL_DIR, str(model_type))
        elif task_type == "篮球检测":
            model_path = Path(config.BACKERBALL_MODEL_DIR, str(model_type))
    else:
        st.error("Please Select Model in Sidebar")

    print(model_path)

    model = None
    # 加载预训练的 DL 模型
    try:
        model = load_model(model_path)
        if model:
            st.success("Model loaded successfully.")  #
        else:
            st.error("Model loaded, but it's None.")  #
    except Exception as e:
        st.error(f"Unable to load model. Please check the specified path: {model_path}")

    # image/video options
    st.sidebar.header("Image/Video Config")
    source_selectbox = st.sidebar.selectbox(
        "选择来源",
        config.SOURCES_LIST
    )

    source_img = None
    if source_selectbox == config.SOURCES_LIST[0]:  # Image
        infer_uploaded_image(confidence, model)
    elif source_selectbox == config.SOURCES_LIST[1]:  # Video
        infer_uploaded_video(confidence, model, task_type)
    elif source_selectbox == config.SOURCES_LIST[2]:  # Webcam
        infer_uploaded_webcam(confidence, model)
    else:
        st.error("Currently only 'Image' and 'Video' source are implemented")



