#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

# 获取当前文件的绝对路径
file_path = Path(__file__).resolve()

# 获取当前文件的父目录
root_path = file_path.parent

# 将根路径添加到 sys.path 列表（如果尚不存在）
if root_path not in sys.path:
    sys.path.append(str(root_path))

# 获取根目录相对于当前工作目录的相对路径
ROOT = root_path.relative_to(Path.cwd())


# Source
SOURCES_LIST = ["Image", "Video", "Webcam"]


# DL model config
DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
SEGMENT_MODEL_DIR = ROOT / 'weights' / 'segment'
POSE_MODEL_DIR = ROOT / 'weights' / 'pose'
BACKERBALL_MODEL_DIR = ROOT / 'weights' / 'basketball'

YOLOv8n = DETECTION_MODEL_DIR / "yolov8n.pt"
YOLOv8s = DETECTION_MODEL_DIR / "yolov8s.pt"
YOLOv8m = DETECTION_MODEL_DIR / "yolov8m.pt"
YOLOv8l = DETECTION_MODEL_DIR / "yolov8l.pt"
YOLOv8x = DETECTION_MODEL_DIR / "yolov8x.pt"

DETECTION_MODEL_LIST = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt"]

SEGMENT_MODEL_LIST = [
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt"]

POSE_MODEL_LIST = [
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
    "yolov8m-pose.pt",
    "yolov8l-pose.pt",
    "yolov8x-pose.pt"]

BASKERBALL_MODEL_LIST = [
    "BasketBall.pt"]