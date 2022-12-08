# image-desensitization

本算法包可针对图片中的敏感区域，进行区域性脱敏及复原处理；

共提供3种图像脱敏方法：XOR、Shuffle、Blur+XOR。

### 数据准备
待处理图片及对应的xml标注文件放在/data文件夹下

### Demo运行
（1）XOR方法:

 \python xor.py
 
（2）Shuffle方法:

 \python shuffle.py
 
（3）Shuffle方法:

 \python blur_xor.py
 
 ### 结果展示
 最终结果保存在/results文件夹下

