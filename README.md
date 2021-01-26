# 3D pose estimation

``` 
.
├── Eval.py # 单独验证方向
├── Eval_trans.py # 验证位置
├── Eval_qu.py # 验证四元数
├── Eval_six.py # 验证六欧拉角
├── Eval_combine.py # 验证位置+方向
├── Eval_euler.py # 验证欧拉角
├── Train.py # 训练方向
├── Train_combine.py # 训练方向+位置
├── Train_qu.py # 训练四元数
├── Train_six.py # 训练六欧拉角
├── Train_erler.py # 训练欧拉角
├── camera_cal
│   └── calib_cam_to_cam.txt # 相机内参
├── contrast_experiment # 对比实验模型结构
│   ├── model_qu.py # 四元数模型
│   ├── model_six.py # 六欧拉角模型
│   ├── model_euler.py # 欧拉角模型
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

2021.01.26 记录：所有对比实验部分完成，画图部分只修改了Eval和Eval_combine的代码。combine部分是将方向的特征叠加到了位置特征上，还没有做进一步的改进，后面完善改进部分。
