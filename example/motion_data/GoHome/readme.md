# GoHome动作数据添加脚本

## 功能说明

1. 读取input文件夹中的pkl文件
2. 在动作数据尾帧添加数据，使得机器人在3s内GoHome复位
3. 输出到output文件夹

## 使用方法

```bash
cd example/motion_data/GoHome
python add_gohome.py
```

## 说明

- 脚本会自动读取`input/`文件夹中的所有`.pkl`文件
- 对每个文件，会在末尾添加3秒的GoHome复位动作（使用平滑插值）
- 处理后的文件会保存到`output/`文件夹，文件名会添加`_gohome`后缀（例如：`motion.pkl` → `motion_gohome.pkl`）
- GoHome的关节角度定义在脚本中的`GOHOME_JOINT_ANGLES`变量

## GoHome关节角度

按照XML中actuator的顺序（共23个关节）：
- 左腿：hip_pitch=-0.1, hip_roll=0.0, hip_yaw=0.0, knee=0.3, ankle_pitch=-0.2, ankle_roll=0.0
- 右腿：hip_pitch=-0.1, hip_roll=0.0, hip_yaw=0.0, knee=0.3, ankle_pitch=-0.2, ankle_roll=0.0
- 腰部：yaw=0.0, roll=0.0, pitch=0.0
- 左臂：shoulder_pitch=0.2, shoulder_roll=0.2, shoulder_yaw=0.0, elbow=0.9
- 右臂：shoulder_pitch=0.2, shoulder_roll=-0.2, shoulder_yaw=0.0, elbow=0.9