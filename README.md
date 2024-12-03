# GAN-empowered-Model

## Version 1.0
### 对于损失函数分析（见loss.py）  
1. Critic()函数的输出结果是对于给定数据的评价值，这个评价值可以是任意值  
仍需解决：Critic的具体数值是怎么计算出来的？  
2. 梯度惩罚项（也可以用权重剪切）是用来满足K-Lipschitz约束的（k通常为1），可以使得Loss function的取值范围缩小  
3. 
### 该版本的存在的问题：
1. Critic和Generator的结构是CNN，该结构理论上不适合训练本项目中需要训练的数据类型。  
    目前预计将改进为Tranformer相关架构。
2. 