# Fall-Detection-System  
Research and Implementation of Fall Detection System based on FPGA and Deep Learning  
此仓库用于存放项目相关源代码  
  
软件开发计划：  
前置：完成对基本模型的构建。  
额外：调用电脑自带的摄像头，数据传输并分类，输出结果。  
1、处理：对数据集进行预处理（裁剪，旋转，拼接；降噪，采样率、、、），训练方式：交叉验证。测试模型效果（写进论文，所以重要）。  
进阶：使用如yolov8-pose/openpose等姿态识别模型做预处理/后处理，最终输出可识别人体并检测摔倒。  
可使用dy/dx做可解释化，也可输出识别特征。
2、对比：使用不同的模型对同一数据集做相同训练并评估其与Mobilenet的区别，例如数据集的需求量，准确率，泛用性，训练速度训练时间以及对性能资源的要求。（选做）  
3、 评估：对模型进行评估，需要多大量的数据即可达到一个良好的准确率，不足多少（学习率设置）会过拟合、泛用性变差等。  
方法：分离训练函数和测试函数，每训练一定周期（样本量）导出一次模型，对不同训练程度的模型做同背景测试集与不同背景测试集测试，画出曲线图。
4、数据集：找一些不同背景的测试集对训练好的模型测试，评估是否需要制作不同背景的数据集以提高泛用性，（如教室背景如何，实验室背景如何，室外背景如何）。  
5、改进：对mobilenet源码进行改进，修改层数，超参数的设置，评估改进效果，变化如何有何优势。（速率、准确率、所需数据量、泛用性等等）  
比较重要的就134，5算进阶，尽量在中期报告前完成（6.14）  
目前第九周，中期检查第十六周，尽快完成数据集的搭建4，第十第十一周完成13，第十二第十三周完成5，第十四周后做补缺与准备报告。  

硬件开发计划：  
完成网络每一步操作的verilog实现。  

教程链接：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing  

原始论文链接：https://arxiv.org/abs/1801.04381   
