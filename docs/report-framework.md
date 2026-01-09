
标题： G1机器人舞蹈实验


abstract

基于开源项目PBHC(https://github.com/TeleHuman/PBHC?tab=readme-ov-file)和相关论文进行复现。
而后尝试通过GVHMR项目进行视频动作提取、尝试通过自行重定向的方式将AMASS数据集中一些挑选出的动作进行提取。从而扩大训练动作集。
通过渐进阶段式的训练方式，不断在原模型的基础上进行强化，从而得到最终掌握较多动作的机器人策略模型。并在部分未经过训练的简单动作上进行了测试，观察到基本的模式遵循并且并未摔倒。

> PBHC is the official implementation of the paper KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills, supporting general motion tracking of the paper KungfuBot2: Learning Versatile Motion Skills for Humanoid Whole-Body Control.


Introduction: 

简单介绍一下基于的项目PBHC:

PBHC是一个离线模仿学习、进行运动跟踪的工作。

策略模型输入的核心就是未来数步内的参考动作轨迹，
通过强化学习训练出一个“有脑子”的控制器。即便有物理干扰或者地形微小变化或者参考动作本身不符合物理规律，它也能努力调整姿态去跟上那个动作文件的轨迹，保持不摔倒。

两篇论文分别提供了两种训练模式：
1. “专才”模式。
它为每一个动作文件训练一个单独的策略（Policy）。

比如你要跑“回旋踢”，你就加载“回旋踢”的动作文件和对应的模型；要跑“太极”，就加载“太极”的文件和模型。

2. “通才”模式。

它训练一个统一的策略来掌握所有动作 。

但在运行时（Inference/Simulation），你依然需要喂给它一个具体的动作文件（比如 walk.npy 或 dance.npy）。策略网络会看一眼这个文件说：“哦，你要我现在做这个动作”，然后开始模仿。如果没有这个文件作为引导（Goal），机器人就不知道该干什么。

----

Our work:

== 阶段1

我们首先摸索训练、使用、复现PBHC项目的基本模式。
首先从复现最简单的单动作跟踪开始打通我们的环境配置和基本工作流

该阶段的基本工作在于，确立工作流：
1. 通过虚拟屏幕或VNCServer解决GPU服务器无显示屏导致训练无法正常工作的问题
2. 使用`tmux new -s [terminal name] `创建持久化的终端，确保在断开与服务器的SSH连接之后训练仍能持续进行
3. 使用`conda activate tb_viewer
tensorboard --logdir logs/MotionTracking --port 6006` ，配合ssh端口转发，将训练日志转发到本地浏览器 检查训练情况，用于把握训练进度和进行结果模型的挑选
4. 发现Google 的MuJoCo原生支持Windows系统，故而在本地win机上对产出的策略模型进行仿真测试、录屏并整理结果。


该阶段的基本训练命令为：
```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=horse_stance_pose \
robot.motion.motion_file="example/motion_data/Horse-stance_pose.pkl" \
seed=1 \
+device=cuda:0 \
algo.config.num_learning_iterations=50000 \
algo.config.save_interval=2000
```
基本完全复现论文参数，不进行修改，以确保结果可靠性，从而正确验证环境配置无误。


发现效果很好。动作完美跟踪。
注：这里不再单独演示，因为在后续的工作下最终得到的模型同样能够完美复现这一部分的动作。

并在此过程联想到模型训练过程必备的进度管理和分层训练。
发现框架已经提供了`checkpoint=logs/MotionTracking/20251207_124854-charleston_dance_pose-branch1-motion_tracking-g1_23dof_lock_wrist/model_12000.pt \`
参数, 用于在前继模型的基础上进行训练。这为后续的训练模式奠定了基础——后续每个阶段基本上都是在前继模型的基础上进行的训练。


== 阶段2：进入通用策略模型训练阶段

阶段1我们复现Kungfubot1的，实现专才的训练

阶段2我们则开始尝试复现Kungfubot2的工作，仍然是复用PBHC已有的项目代码和训练参数，确保训练的可靠性。

在这一阶段需要的工作是：

在阶段1的基础上，通过
`python motion_source/motion_package.py "example/motion_data/singles"` 命令对多个动作数据进行打包。打包成merged_motion.pkl，用于后期同时对多动作进行训练。

在训练模式上，除了更改obs和reward配置以外，还需要分教师模型和学生模型进行训练，从而加速训练的进行。

基本思路是：拥有更多obs——更多特权信息(如xxx, 引用obs_ppo_teacher)的教师模型，快速学习。而后通过对教师蒸馏的方式，将知识快速灌注给学生。

教师训练的基本命令形如:
```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=general_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/obs_ppo_teacher \
+robot=g1/g1_23dof_general \
+domain_rand=main \
+rewards=motion_tracking/general_main \
experiment_name=teacher_training \
robot.motion.motion_file="example/motion_data/merged_motion.pkl" \
seed=666 \
+device=cuda:2 \
algo.config.num_learning_iterations=50000 \
algo.config.save_interval=1000
```

学生训练的基本命令形如
```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=general_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/obs_ppo_student \
+robot=g1/g1_23dof_general \
+domain_rand=main \
+rewards=motion_tracking/general_main \
experiment_name=student_training \
robot.motion.motion_file="example/motion_data/merged_motion.pkl" \
algo.config.dagger_only=True \
algo.config.teacher_model_path="logs/MotionTracking/20251208_072625-teacher_training-motion_tracking-g1_23dof_lock_wrist/model_49000.pt" \
seed=777 \
+device=cuda:1 \
algo.config.num_learning_iterations=50000 \
algo.config.save_interval=1000
```

其实num_envs有些类似于batch size (会同时有这么多机器人被模拟出来进行尝试进行学习), 在显存足够的时候可以通过调大该数值来增加每次迭代学习到的经验量。从而加速训练。
在训练动作数据总量很大的时候则要适当减小以免爆显存。


该阶段完成之后，我们也得到了一个很好的结果。事实上训练出来的模型已经能够同时对所有预置的训练动作(武术动作)做到非常好的动作跟踪。


注：在这一阶段为了更好地进行观察，我们稍微魔改了一下仿真代码加载动作的逻辑，增加了一个快捷键N。从而仿真时可以直接加载merged_motion.pkl而非单motion.pkl, 通过快捷键N——next，在不切换策略模型的基础上，对动作进行切换。
这一部分只是为了方便，不影响仿真结果。严谨起见提一嘴。
不过有趣的是，在这一阶段，进行动作切换的时候，我们已经观察到了机器人能够自行从上一个动作的当前状态尝试平滑的过渡到下一个动作的起始状态。
并且只有上一个状态不要太不稳定，基本都能成功过渡。我们注意到，这其实已经体现了一种“通用性”、“泛化性”的雏形。因为过渡行为本质上是训练集外的数据。

== 阶段3

为了更进一步完成实验。在这一阶段，我们尝试对数据集进行扩充。
首先我们研究了PBHC是如何获取动作数据的。
于是我们参考他们的方式，配置了GVHMR项目(https://github.com/zju3dv/GVHMR)
并使用这个项目，从视频中提取动作数据。

为了尽可能确保动作提取的准确性。我们尽量寻找全身无遮拦的舞蹈视频，以便减少因为信息丢失 ，导致模型胡乱预测，提取的动作数据不堪受用的问题。

为了确保实验的多样性。我们尝试了三类动作视频的提取。

1. 来自b站真人舞蹈视频。up主 @鱼肉肉pabo 跳的黑帮摇切片。

![gangster](./report-framework/stage3-gangster-source.png)

**图1：黑帮摇舞蹈视频截图（来源：B站@鱼肉肉pabo）**

2. 来自 高真实度渲染的 符华Addition 舞蹈动画视频(虚拟人物)(固定视角)

![addition](./report-framework/stage3-addition-source.png)

**图2：符华Addition舞蹈动画视频截图（虚拟人物渲染）**

3. 尝试对MMD的vmd动作进行 再利用，使用MikuMikuDance进行桥接，固定视角，录制MMD舞蹈视频，在此基础上进行动作提取

![idol](./report-framework/stage3-idol-source.png)

**图3：MMD idol舞蹈视频截图**

![dancehall-bug](./report-framework/stage3-dancehall-bug.png)

**图4：MMD dancehall舞蹈视频截图（出现跟踪丢失问题）**



起初考虑纯色背景无干扰可能利于提取，但如dancehall所示，有时可能出现人物跟踪丢失或者 提取的动作在世界坐标下的位置变换——即移动和面向——与实际不符的情况。

现在回看，猜测可能是因为CV方法大都依赖于梯度，在画面纹理特征不够丰富时难以保证匹配效果。





---

除此之外，我们还观察到动作存在其他问题：

- 双脚悬空
- 抖动严重
- 提出为SMPL格式，需要转换到G1格式



我们直接利用PBHC自带的mink_retarget来完成这件事。
并额外增加了巴特沃斯滤波器工具函数对动作进行滤波，并允许开启强制双脚贴地模式，来抛弃一定的动作数据质量，转而确保训练稳定性和动作的基本质量；

对于涉及跳跃的能力，将期望转交给后续整理的相对高质量的动捕数据进行训练。



我们在这一阶段的基础上进行了训练尝试。并产出了代号为 `student-v2-deeper-ckpt_112000`的模型。

在保持原有武术模型正确遵循的基础上，额外获得了对舞蹈动作的基本遵循能力。

![student-v2-deeper-idol](./report-framework/stage3-student-v2-deeper-idol.png)

**图5：student-v2-deeper模型执行idol舞蹈动作**





## 阶段4



尝试使用更高质量的确定性数据来构造训练集，尝试动捕数据和MMD vmd数据两个方向。

- MMD vmd: 由于MMD数据大量依赖IK，并且MMD人物骨骼中存在很多为了动画制作方便而虚构的骨骼。可能需要先实现高质量IK，再对结果进行重定向。时间成本较高，受限于时间原因，中途放弃。
- 动捕数据：过程中在Hugging face发现https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1，于是决定直接使用berkeley实验室重定向好的AMASS数据。在其中筛选高质量可用的片段，用于训练。





而后就是在student-v2-deeper-ckpt_112000的基础上，针对扩容后的动作不断进行训练。试图增强模型泛化性和稳定性。



针对训练中出现的问题，后而分别对动作进行了进一步的剔除、裁剪，并为了保证结束状态的稳定性给必要动作增加了goHome后缀使之复位。

最终采纳的训练动作集为:
```bash
    Directory: D:\__Files\lessons\_normal\ComputerGraphic\Project\PBHC\example\motion_data\singles


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         2025/12/6     11:46         478963 1_Charleston_dance.pkl
-a----         2025/12/6     11:46         259443 2_Bruce_Lee_pose.pkl
-a----         2025/12/6     11:46         165363 3_Horse-stance_pose.pkl
-a----         2025/12/6     11:46         157523 4_Horse-stance_punch.pkl
-a----         2025/12/6     11:46         137907 5_Hooks_punch.pkl
-a----         2025/12/6     11:46         124563 6_Roundhouse_kick.pkl
-a----         2025/12/6     11:46         141043 7_Side_kick.pkl
-a----        2025/12/29     22:54        1751959 _1_rin-idol_gohome.pkl
-a----        2025/12/29     22:54         789775 _2_gangster_0-_gohome.pkl
-a----        2025/12/29     22:54        2153527 _3_addition_7828-11600_gohome.pkl
-a----        2025/12/29     22:54        2648719 _4_rin-dancehall_330-4980_gohome.pkl
-a----        2025/12/29     22:54         917627 __1_punch-punch-punch_gohome.pkl
-a----        2025/12/29     22:54         113267 __2_walk-left_gohome.pkl
-a----        2025/12/29     22:54        1830191 __7_walking-around_gohome.pkl
-a----        2025/12/29     22:54        2409239 __9_volleyball-jump_gohome.pkl
```



并在训练的最终阶段增大滑步和终止(摔倒)惩罚，加大稳定性的考虑权重。为可能的真机部署减少风险。



取训练过程中mean_episode_length极大值点对应模型，最终得到模型`stability-ckpt_212000`

最终效果请见附件`sta-212000-导演剪辑版_plus.mp4`.

原武术动作和查尔斯舞能力依然保持。

![sta-212000-charleston](./report-framework/stage4-stability-charleston.png)

**图6：stability模型执行查尔斯顿舞蹈动作**

在新增的动捕数据上学会了跑步等能力

![sta-212000-walkaround](./report-framework/stage4-stability-walkaround.png)

**图7：stability模型执行walking-around动作**

学习到了一次成功的跳跃

![sta-212000-jump-success](./report-framework/stage4-stability-jump-success.png)

**图8：stability模型成功执行跳跃动作**

但第二次跳跃失败

![sta-212000-jump-fail](./report-framework/stage4-stability-jump-fail.png)

**图9：stability模型跳跃动作失败**

学习新舞蹈

![sta-212000-gangster](./report-framework/stage4-stability-gangster.png)

**图10：stability模型执行gangster舞蹈动作**

![sta-212000-addition](./report-framework/stage4-stability-addition.png)

**图11：stability模型执行addition舞蹈动作**

![sta-212000-dancehall](./report-framework/stage4-stability-dancehall.png)

**图12：stability模型执行dancehall舞蹈动作**

![sta-212000-idol-before-slip](./report-framework/stage4-stability-idol-before-slip.png)

**图13：stability模型执行idol舞蹈动作（打滑前）**

![sta-212000-idol-slipping](./report-framework/stage4-stability-idol-slipping.png)

**图14：stability模型执行idol舞蹈动作（打滑时）**

但是在idol这支舞蹈的尾声。将双脚岔开下伏的时候脚底打滑了，摔倒了。

一连观察了好几个后期训练的比较好的模型都出现这个问题。

考虑到，对于这一关, `student-v2-deeper-ckpt-11200`给出的解法是这样的。脚的朝向完全逆转了。

![student-v2-deeper-idol-foot-reverse](./report-framework/stage3-student-v2-deeper-idol-foot-reverse.png)

**图15：student-v2-deeper模型执行idol动作时的脚部朝向逆转解法**

分析：或许是因为，经过更多训练后，模型能够尽可能地去做更多符合人类工学的动作，但是常规解法并不是很能过这一关。



---



最后我们在几个完全没有训练过的动作上测试stability-ckpt_212000的能力

```bash
    Directory: D:\__Files\lessons\_normal\ComputerGraphic\Project\PBHC\example\motion_data\test


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----        2025/12/26     19:08          39954 __4_run-left.pkl
-a----        2025/12/29     22:54          62235 __5_rush-left_gohome.pkl
-a----        2025/12/29     22:54        1325931 __6_wave-hand_gohome.pkl
```

结论：全都没摔

![sta-212000-untrained-run](./report-framework/stage4-stability-untrained-run.png)

**图16：stability模型执行未训练动作run（正确）**

![sta-212000-untrained-rush](./report-framework/stage4-stability-untrained-rush.png)

**图17：stability模型执行未训练动作rush**

rush因为动作时间极短，仅1s左右，所以实际上并不能观察到冲刺的过程。但是可以观察到冲刺结束的急刹过程。并且注意到机器人在这样的背景下仍然保持鲁棒、保持不摔。

![sta-212000-untrained-wave-hand](./report-framework/stage4-stability-untrained-wave-hand.png)

**图18：stability模型执行未训练动作wave-hand**

长动作，有位移，有挥手，顺利完成无摔倒。

































