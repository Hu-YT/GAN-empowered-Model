# GAN-empowered-Model
## 数据集说明
数据集是1224&times;10&times;5的形式，1224是观测角度数量(观测数量)，
10是scattering centers有10个点表示，每个点分别用x,y,z坐标，频率依赖性和功率五个量来表示。
## Version 1.0
### 对于损失函数分析（见loss.py）  
1. Critic()函数的输出结果是对于给定数据的评价值，这个评价值可以是任意值  
仍需解决：Critic的具体数值是怎么计算出来的？  
2. 梯度惩罚项（也可以用权重剪切）是用来满足K-Lipschitz约束的（k通常为1），可以使得Loss function的取值范围缩小  
3. 从理论的角度上说，Lg应该逐渐减小，Lc应该逐渐增大？
### 该版本的存在的问题：
1. Critic和Generator的结构是CNN，该结构理论上不适合训练本项目中需要训练的数据类型。
而且从epoch10000次的结果看，效果也没有很好。  
    目前预计将改进为Tranformer相关架构。
## Version 2.0
### 改进点
1. WGAN的Optimizer不能用基于动量法的如Adam，建议使用SGD或者RMSProp
2. 数据预处理，目前采用min-max线性Normalization的方法,xyz统一归一化
3. 改变batch_size大小比较效果

## Version 3.0
### 改进点
1. 使用Linear全连接层代替CNN，观察效果
2. 设置不同的batch_size大小比较效果



### 待研究改进的点
1. 权重weight初始化
2. 对各个critic数值的记录，用来比较验证效果
3. 改用Transformer

### 理想效果