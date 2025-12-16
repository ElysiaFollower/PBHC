= Train robot using self video
== Extract motion from video, get npz file
Using GVHMR to extract motion from video to SMPL format. (according to motion_source/README.md)
- replace GVHMR's demo.py with demo.py in PBHC

== Retarget SMPL motion data to the robot

1. Git clone https://github.com/ZhengyiLuo/SMPLSim.git
  - 我安装在了PBHC/SMPLSim/
  - 运行时要将SMPLSim加到PYTHONPATH中
  - 最终执行指令在PBHC-main/smpl_retarget下执行```PYTHONPATH=~/PBHC-main/SMPLSim python mink_retarget/convert_fit_motion.py motion_data```
2. 
~/PBHC-main/smpl_retarget/
├── smpl_model/
│   └── smpl/
│       ├── SMPL_NEUTRAL.pkl   ← 必须存在
│       ├── SMPL_MALE.pkl
│       └── SMPL_FEMALE.pkl
├── motion_data/
└── mink_retarget/

3. 环境
python 3.9 (3.7试过不行) (必须<=3.9)
```bash
conda create -n ipman_p39 python=3.9
conda activate ipman_p39
conda install -c fvcore -c iopath -c conda-forge -c pytorch pytorch3d
cd ipman_r
pip install "setuptools<60.0"
pip install -r requirements.txt
```

安装mujoco到```$HOME/.mujoco/mujoco-2.3.6```

```bash
export MUJOCO_PATH=$HOME/.mujoco/mujoco-2.3.6
export MUJOCO_PLUGIN_PATH=$MUJOCO_PATH/bin
export LD_LIBRARY_PATH=$MUJOCO_PATH/lib:$LD_LIBRARY_PATH

pip install mujoco==3.3.6 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip uninstall open3d numpy -y
pip install open3d==0.18.0 -i https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall --no-cache-dir
pip install numpy==1.23.5 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install typer dm_control loop_rate_limiters mink qpsolvers[quadprog]
pip install smplx[all] -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "numpy>=1.20.0,<1.24.0" -i https://pypi.tuna.tsinghua.edu.cn/simple


```

pytorch3d 

=== Motion Filter



