# 3D pose estimation

``` 
.
├── Eval.py # 单独验证方向
├── Eval_trans.py # 验证位置
├── Train.py # 训练方向
├── Train_combine.py # 训练方向+位置
├── Train_qu.py # 训练四元数
├── camera_cal
│   └── calib_cam_to_cam.txt # 相机内参
├── contrast_experiment # 对比实验模型结构
│   ├── model_qu.py # 四元数模型
│   ├── posenet_combine.py # 联合方向和位置
├── library
│   ├── File.py 
│   ├── Math.py # 旋转矩阵
│   ├── Reader.py # 读取标签文件
│   ├── drawBox.py # 画图代码
│   └── evaluate.py # 验证代码 方向+位置+add
├── torch_lib
│   ├── dataset_posenet.py # 读取数据
│   ├── mbv3_large.old.pth.tar # mobilenetv3权重文件
│   ├── mobilenetv3_old.py # mobilenetv3模型结构
│   └── posenet.py  # 方向网络结构
```
