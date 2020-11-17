# 图像滤镜效果实现

## 一、人像美肤

### 1.1 运行环境

- OpenCV 4.5.0



### 1.2 运行步骤

1. 在控制台输入``cd SkinFilter``
2. 输入``./SkinFilter.exe``
3. 输入图片的路径，默认为``data``文件中的``lenna.png``
4. 默认在``SkinFilter``文件夹输出结果图像``newimage.png``
5. 在``SkinFilter``文件夹输出原图像与结果图像之间的差异``difference.png``



### 1.3 算法原理

首先程序调用``detectFaces``函数检测图像中人脸的位置。该函数使用OpenCV提供的预训练模型进行检测。模型存放在``SkinFilter/haarcascade_frontalface_default.xml``文件中。函数返回一个``Rect``数组，每个``Rect``记录某个包围人脸的矩形的x和y坐标以及其宽高。

```c++
void detectFaces(Mat &img, vector<Rect> &faces, String &classifierPath) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    CascadeClassifier classifier;
    classifier.detectMultiScale(gray, faces);
}
```

然后程序在每个人脸矩形区域进行双边滤波。

双边滤波同时考虑空域位置距离和灰度相似性，从而能够更好的保存图像的边缘信息。其公式如下：
$$
I^{filtered}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} f_r(||I(x_i) - I(x)||) g_s(||x_i - x||)
$$

$$
W_p = \sum_{x_i \in \Omega}f_r(||I(x_i) - I(x)||) g_s(||x_i - x||)
$$

其中：

- $$
  I^{filtered} 是滤波后的图像
  $$

- $$
  I 是原图像
  $$

- $$
  x 是当前要被滤波的像素坐标
  $$

- $$
  \Omega 是中心为 x 的窗口，即卷积核覆盖的区域
  $$

- $$
  f_r 是灰度核函数，用来平滑灰度的差异
  $$

- $$
  g_s 是空域核函数，用来平滑坐标位置的差异
  $$

在程序中，两个核函数都使用以下高斯函数：
$$
f(u) = \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{u^2}{2 \sigma^2}}
$$
同时，超参数设为：

1. 卷积核大小设为10 X 10大小
2. 灰度核函数的 σ 设为25，空域核函数的 σ 设为25



### 1.4 算法效果

左边为滤波前效果；中间为滤波后效果；右边为滤波前与滤波后的差异。

为了更好显示滤波前后的差异，右图显示的差异是实际差异的10倍。

![skinresult](Report/skinresult.png)

脸部细节对比：可以看到滤波后脸部皮肤更加光滑。

![skinresult](Report/skincompareresult.png)

## 二、LOMO滤镜

### 2.1 运行环境
- OpenCV 4.5.0



### 2.2 运行步骤

1. 在控制台输入``cd LomoFilter``
2. 输入``./LomoFilter.exe``
3. 输入图片的路径，默认为``data``文件中的``lenna.png``
4. 默认在``LomoFilter``文件夹输出结果图像``newimage.png``



### 2.3 算法原理

首先使用以下函数进行对比度拉伸，使亮的像素更亮，暗的像素更暗。

$$
y = \frac{1}{1 + e^{-\frac{x-0.5}{0.1}}}
$$

![lomofunc](Report/lomofunc.png)

在实现时，为了加快运算速度，程序使用查表法预先存储每个``x``对应的``y``值。当处理图片时，只需要将当前像素值作为表的下标便可以获得结果值。



然后在与图像大小相同的区域的中间绘制一个圆，半径为图像宽与高之间最小值的1/3。之后使用盒式滤波器对绘制的圆进行滤波，滤波器核的大小为图像宽与高之间最小值的1/2。

在实现时，为了加快运算速度，程序利用盒式滤波器核可分离的性质。这样就只需用计算两次向量与矩形的乘积便可获得卷积的结果。

$$
\frac{1}{9}
\left[
\begin{matrix}
1 & 1 & 1\\
1 & 1 & 1\\
1 & 1 & 1\\
\end{matrix}
\right]
= \frac{1}{9}
\left[
\begin{matrix}
1 \\
1 \\
1 \\
\end{matrix}
\right] 
\cdot
\left[
\begin{matrix}
1 & 1 & 1 \\
\end{matrix}
\right]
$$

最后将卷积结果和对比度拉伸结果进行乘积，获得最终结果。

### 2.4 算法效果

![lomoresult](Report/lomoresult.png)