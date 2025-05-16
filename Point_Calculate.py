import math
import streamlit as st

def calculate_angle(keypoint_data):
    data = keypoint_data[5:] # 忽略前五个数据

    # 将数据分组为17组，每组3个连续的数据
    grouped_data = [list(map(float, data[i:i+3])) for i in range(0, len(data), 3)][:17]
    # 提取第9、7、5号点的坐标
    point_5 = grouped_data[5]
    point_7 = grouped_data[7]
    point_9 = grouped_data[9]
    x1, y1, _ = point_7
    # 计算点构成的角的角度
    angle_right_elbow = calculate_angle_between_points(point_9, point_7, point_5)
    print(f"右肘x坐标值为:{x1}")
    print(f"右肘y坐标值为:{y1}")
    print(f"右肘角度（弧度）： {angle_right_elbow} 弧度")

def calculate_angle_between_points(point1, point2, point3):
    # 计算两点之间的角度，返回弧度值
    x1, y1, _ = point1
    x2, y2, _ = point2
    x3, y3, _ = point3

    vector1 = (x1 - x2, y1 - y2)
    vector2 = (x3 - x2, y3 - y2)

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude_product = math.sqrt(sum(v ** 2 for v in vector1)) * math.sqrt(sum(v ** 2 for v in vector2))

    cos_theta = dot_product / magnitude_product

    # 使用反余弦函数计算角度
    angle_rad = math.acos(cos_theta)

    # 将弧度转换为角度
    angle_deg = math.degrees(angle_rad)

    return angle_deg
