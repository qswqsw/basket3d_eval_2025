
# Limbs 肢体部分
LEG_UPPER_RIGHT = (24,26)#表示右大腿（右臀部和右膝之间的连线）。



LEG_LOWER_RIGHT = (26,28)#表示右小腿（右膝和右脚踝之间的连线）。
UPPER_BODY_RIGHT = (12,24)#表示右上身（右肩膀和右臀部之间的连线）。
ARM_UPPER_RIGHT = (12,14)#表示右上臂（右肩膀和右肘之间的连线）。
ARM_LOWER_RIGHT = (14,16)#表示右前臂（右肘和右腕之间的连线）。
FOOT_RIGHT = (28,32)#表示右脚（右脚踝和右脚之间的连线）。

LIMBS_ALL = [LEG_UPPER_RIGHT,UPPER_BODY_RIGHT,ARM_UPPER_RIGHT,ARM_LOWER_RIGHT,FOOT_RIGHT]

# Joints 关节部分
ANKLE_RIGHT = (26,28,32)#表示右踝关节（右脚踝和右脚之间的连线）。
ELBOW_RIGHT = (12,14,16)#表示右肘关节（右肩膀和右肘之间的连线）。
SHOULDER_RIGHT = (14,12,24)#表示右肩关节
HIP_RIGHT = (12,24,26)#表示右臀关节（右肩膀和右臀部之间的连线）。
KNEE_RIGHT = (24,26,28)#表示右膝关节（右臀部和右膝之间的连线）。

JOINTS_ALL = [ANKLE_RIGHT,ELBOW_RIGHT,SHOULDER_RIGHT,HIP_RIGHT,KNEE_RIGHT]

joint_to_text = {(26,28,32):'Ankle',(12,14,16):'Elbow',(14,12,24):'Shdlr', (12,24,26):'Hip', (24,26,28): 'Knee'}
joint_to_muscle = {(26,28,32):'Calf',(12,14,16):'Tricep',(14,12,24):'Shoulder', (12,24,26):'Glute', (24,26,28): 'Quad'}
