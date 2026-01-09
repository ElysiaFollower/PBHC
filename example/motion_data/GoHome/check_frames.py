import joblib
import numpy as np
import os

os.chdir('D:/__Files/lessons/_normal/ComputerGraphic/Project/PBHC')

files = [
    'example/motion_data/GoHome/output/__1_punch-punch-punch_gohome.pkl',
    'example/motion_data/GoHome/input/__1_punch-punch-punch.pkl'
]

for f in files:
    print(f'\n=== {f} ===')
    try:
        data = joblib.load(f)
        key = list(data.keys())[0]
        motion = data[key]
        print(f'帧数: {motion["dof"].shape[0]}')
        print(f'fps: {motion.get("fps", "N/A")}')
        print(f'后3帧dof:')
        print(motion["dof"][-3:])
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()

