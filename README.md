# 人脸识别 java 代码

共分为两个模块：提取人脸区域、将人脸图片映射为高维向量。

## 提取人脸区域

这里一共实现了两种方式，Opencv 和 mtcnn。

### Opencv Haar特征分类器

配置文件下载地址在[这里](https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php)。

### MTCNN

模型文件下载地址在[这里](https://raw.githubusercontent.com/cayden/UVCCamera/674dc567810ba70d39a7e533c0c39f06b4dbf68f/usbCameraTest/src/main/assets/mtcnn_freezed_model.pb)。

## 将人脸图片映射为高维向量。

这里使用 facenet 模型。(转换代码还未完成） 

模型文件下载地址在[这里](https://github.com/davidsandberg/facenet)。


# 参考

1. https://blog.csdn.net/shuzfan/article/details/52668935
2. https://zhuanlan.zhihu.com/p/25025596
3. https://github.com/vcvycy/MTCNN4Android/blob/master/app/src/main/java/com/example/vcvyc/mtcnn_new/MTCNN.java
