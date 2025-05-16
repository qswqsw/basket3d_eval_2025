#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
from ultralytics.utils import ops as op
import Point_Calculate as pc
import numpy as np
import torch

def _display_detected_frames(conf, model, st_frame, image,task_type):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # 将输入图像image调整为标准大小
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # 使用YOLOv8模型对调整大小后的图像进行对象预测
    res = model.predict(image, conf=conf,save=False)
    data = (res)
    # 在视频帧上绘制检测到的对象：调用res[0].plot()方法，将检测结果可视化为绘制的图像res_plotted
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    # 若是关键点检测，那么进行关键点的节点输出
    print(data)

@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(conf, model,task_type):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile() # 创建一个临时文件对象tfile，用于临时存储视频数据
                    tfile.write(source_video.read()) # 将source_video中的视频数据写入临时文件tfile中。
                    vid_cap = cv2.VideoCapture( # 使用OpenCV的cv2.VideoCapture函数打开临时文件中的视频。
                        tfile.name)
                    st_frame = st.empty() # 创建一个空的Streamlit元素st_frame，用于后续显示视频帧
                    while (vid_cap.isOpened()):  # 进入一个while循环，只要视频仍然打开，就执行以下代码块。
                        success, image = vid_cap.read() # 读取视频的下一帧，并将成功标志和帧图像存储在变量success和image中。
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image,
                                                     task_type
                                                     )
                        else:
                            vid_cap.release() # 释放视频捕获对象
                            break
                except Exception as e: # 捕获任何异常，并将异常对象存储在变量e中
                    st.error(f"Error loading video: {e}") # 在Streamlit应用程序中显示错误消息，指示视频加载过程中发生了错误，并显示具体的错误信息


def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
