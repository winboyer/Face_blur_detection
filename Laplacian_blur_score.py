# -*-coding=UTF-8-*-
"""
在无参考图下，检测图片质量的方法
"""
import os
import cv2
import numpy as np
import mtcnn.tools_matrix as tools
from skimage.transform import estimate_transform, warp
from skimage import filters
import math
import time
import matplotlib.pyplot as plt

class BlurScore:
    def __init__(self, strDir, saveDir=None):
        """
        创建BlurScore对象
        :param: strDir 存放测试图像的文件夹路径
        :return:  BlurScore对象
        """
        print("BlurScore object is created...")
        self.strDir = strDir # strDir：保存测试图像的文件夹路径
        self.saveDir = saveDir

    def _getAllImg(self):
        """
        根据目录读取所有的图片
        :return:  图片列表
        """
        names = []
        for root, dirs, files in os.walk(self.strDir):  
            for file in files:
                names.append(str(file))
        return names


    def _imageToMatrix(self, img):
        """
        将图片对象转化矩阵
        :param img: 图像对象
        :return imgMat: 返回矩阵
        """
        imgMat = np.matrix(img)
        return imgMat

    
    def getAllScore(self):
        """
        对整个数据集进行处理，把所有图像11个评价指标的结果记录在txt文件中
        :return: result.txt
        """
        names = self._getAllImg()
        # f1.write("Image name     Entropy \n")
        blur_scores_list = []
        total_times = 0.0
        if self.saveDir != None and not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)
        for index, name in enumerate(names):
            # print("Processing image: ", name)
            if not name.endswith(".jpg"):
                continue
            if (index+1)%1000 == 0:
                print("processed {} images".format(index+1))
            time_s = time.time()
            img, blur_value = self._Gaussian_Laplacian(name)
            # blur_value = self._Laplacian(name)
            # blur_value = self._Entropy(name)
            # blur_value = self._Brenner(name)
            # img, blur_value = self._Thenengrad(name)
            # img, blur_value = self._SMD(name)
            # img, blur_value = self._SMD2(name)
            # blur_value = self._Energy(name)
            # img, blur_value = self._JPEG(name)
            # blur_value = self._JPEG2(name)
            # img, blur_value = self._Variance(name)
            # blur_value = self._Vollath(name)

            blur_scores_list.append(blur_value)
            time_e = time.time()
            total_times += (time_e-time_s)
            # print('cost time ============ ', time_e-time_s)
            # print('blur_value=======', blur_value)
            # blur_label = f"{blur_value/1000:.3f}" # SMD /1000做归一化
            # blur_label = f"{blur_value/100:.3f}" # SMD2 /100做归一化
            # blur_label = f"{blur_value/10:.3f}" # JPEG /10做归一化
            # blur_label = f"{blur_value/5000:.3f}" # Thenengrad: /5000做归一化
            # blur_label = f"{blur_value/6000:.3f}"  # Variance: /6000做归一化
            blur_label = f"{blur_value/7000:.3f}"  # Gaussian_Laplacian: /7000做归一化

            save_filepath = os.path.join(self.saveDir, blur_label + '_' + name)
            cv2.imwrite(save_filepath, img)

        print('total time =========', total_times)
        print('average time ============', total_times/len(names))
        # plt.hist(blur_scores_list, rwidth=0.5)
        # plt.show()
        list_max = np.max(blur_scores_list)
        list_min = np.min(blur_scores_list)
        list_mean = np.mean(blur_scores_list)
        blur_scores_list.sort()
        list_mid = blur_scores_list[len(blur_scores_list)//2]
        print('list_min, list_mean, list_mid, list_max===========', list_min, list_mean, list_mid, list_max)

        return blur_scores_list

    def _Brenner(self, imgName, do_gray=True):
        """
        指标一：Brenner梯度函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        strPath = os.path.join(self.strDir, imgName)
        img = cv2.imread(strPath)
        img2gray = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)
        f = self._imageToMatrix(img2gray)
        x, y = f.shape
        score = 0
        for i in range(x-2):
            for j in range(y-2):
                score += (f[i+2, j] - f[i, j])**2
        return score


    def _Laplacian(self, imgName):
        """
        指标二：拉普拉斯方差算法
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        strPath = os.path.join(self.strDir, imgName)
        img = cv2.imread(strPath)
        score = cv2.Laplacian(img, cv2.CV_64F).var()
        return score

    def Thenengrad_v2(img):
        T = 50  # 边缘检测阈值
        Grad_value = 0
        Sx, Sy = 0
        x, y = img.shape
        for i in range(1, x - 1):
            current_ptr = img[i, :]  # 当前行
            pre_ptr = img[i - 1, :]  # 上一行
            next_ptr = img[i + 1, :]  # 下一行
            for j in range(1, y - 1):
                Sx = pre_ptr[j - 1] * (-1) + pre_ptr[j + 1] + current_ptr[j - 1] * (-2) + current_ptr[j + 1] * 2 + \
                     next_ptr[j - 1] * (-1) + next_ptr[j + 1];  # x方向梯度
                Sy = pre_ptr[j - 1] + 2 * pre_ptr[j] + pre_ptr[j + 1] - next_ptr[j - 1] - 2 * next_ptr[j] - next_ptr[
                    j + 1]  # y方向梯度
                # 求总和
                G = np.sqrt(Sx * Sx + Sy * Sy)
                if G > T:
                    Grad_value += G
        return Grad_value

    def _Thenengrad(self, imgName):
        """
        指标三：Thenengrad梯度函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        strPath = os.path.join(self.strDir, imgName)
        img = cv2.imread(strPath)
        img2gray = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)
        f = self._imageToMatrix(img2gray)

        tmp = filters.sobel(f)
        score = np.sum(tmp**2)
        score = np.sqrt(score)
        return img, score
    

    def _SMD(self, imgName):
        """
        指标四：SMD灰度方差函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        # 图像的预处理
        strPath = os.path.join(self.strDir, imgName)
        img = cv2.imread(strPath)
        img2gray = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)
        f = self._imageToMatrix(img2gray)/255.0
        x, y = f.shape
        score = 0
        for i in range(x - 1):
            for j in range(y - 1):
                score += np.abs(f[i+1,j]-f[i,j])+np.abs(f[i,j]-f[i+1,j])
        
        return img, score
        
        
    def _SMD2(self, imgName):
        """
        指标五：SMD2灰度方差函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        # 图像的预处理
        strPath = os.path.join(self.strDir, imgName)
        img = cv2.imread(strPath)
        img2gray = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)
        f = self._imageToMatrix(img2gray)/255.0
        x, y = f.shape
        score = 0
        for i in range(x - 1):
            for j in range(y - 1):
                score += np.abs(f[i+1,j]-f[i,j])*np.abs(f[i,j]-f[i,j+1])
        
        return img, score
        
    
    def _Variance(self, imgName):
        """
        指标六：方差函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        # 图像的预处理
        strPath = os.path.join(self.strDir, imgName)
        img = cv2.imread(strPath)
        img2gray = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)
        f = self._imageToMatrix(img2gray)
        score = np.var(f)
        
        return img, score
        
        
    def _Energy(self, imgName):
        """
        指标七：能量梯度函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        strPath = os.path.join(self.strDir, imgName)
        img = cv2.imread(strPath)
        img2gray = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)
        f = self._imageToMatrix(img2gray)
        x, y = f.shape
        score = 0
        for i in range(0, x-1):
            for j in range(0, y-1):
                score += (f[i+1, j] - f[i, j])**2 * (f[i,j + 1] - f[i,j])**2
                
        return score
    
    
    def _Vollath(self, imgName):
        """
        指标八：Vollath函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        strPath = os.path.join(self.strDir, imgName)
        img = cv2.imread(strPath)
        img2gray = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)
        f = self._imageToMatrix(img2gray)
        score = 0
        x, y = f.shape
        for i in range(x-1):
            for j in range(y):
                score += f[i,j]*f[i+1,j]
        score = score - x * y * np.mean(f)
       
        return score
        
    
    def _Entropy(self, imgName):
        """
        指标九：熵函数
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        strPath = os.path.join(self.strDir, imgName)
        img = cv2.imread(strPath)
        img2gray = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)
        f = np.array(img2gray,dtype = 'int64')
        x, y = f.shape
        count = x*y
        p = np.bincount(f.flatten())
        score = 0
        for i in range(0, len(p)):
            if p[i]!=0:
                score -= p[i]*math.log(p[i]/count)/count
        
        return score
        
    
    def _JPEG(self, imgName):
        """
        指标十：论文No-Reference Perceptual Quality Assessment of JPEG Compressed Images
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        strPath = os.path.join(self.strDir, imgName)
        img = cv2.imread(strPath)
        img2gray = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)
        f = np.array(img2gray)
        x,y = f.shape
        # 水平方向
        dh = np.zeros((x,y-1))
        for i in range(x):
            for j in range(y-1):
                dh[i,j] = f[i,j+1] - f[i,j]
        bh = 0
        for i in range(x):
            for j in range(int(y/8)-1):
                bh += abs(dh[i,8*j])
        bh /= x * (int(y/8)-1)
        
        ah = 0
        for i in range(x):
            for j in range(y-1):
                ah += abs(dh[i,j])
        ah = (1/7) * ((8/(x*(y-1))) * ah - bh)
        
        zh = np.zeros((x,y-2))
        for i in range(x):
            for j in range(y-2):
                if dh[i,j] * dh[i,j+1] < 0:
                    zh[i,j] = 1
                else:
                    zh[i,j] = 0
        
        Zh = 0
        for i in range(x):
            for j in range(y-2):
                Zh += zh[i,j]
        Zh /= x*(y-2)
        
        # 垂直方向
        dv = np.zeros((x-1,y))
        for i in range(x-1):
            for j in range(y):
                dv[i,j] = f[i+1,j] - f[i,j]
        bv = 0
        for i in range(int(x/8)-1):
            for j in range(y):
                bv += abs(dv[8*i,j])
        bv /= y * (int(x/8)-1)
        
        av = 0
        for i in range(x-1):
            for j in range(y):
                av += abs(dv[i,j])
        av = (1/7) * ((8/((x-1)*y)) * av - bv)
        
        zv = np.zeros((x-2,y))
        for i in range(x-2):
            for j in range(y):
                if dv[i,j] * dv[i+1,j] < 0:
                    zv[i,j] = 1
                else:
                    zv[i,j] = 0
        
        Zv = 0
        for i in range(x-2):
            for j in range(y):
                Zv += zv[i,j]
        Zv /= y*(x-2)
        
        # 汇总
        B = (bh + bv)/2
        A = (ah + av)/2
        Z = (Zh + Zv)/2
        S = -245.9 + 261.9 * pow(B,-0.024) * pow(A,0.016) * pow(Z, 0.0064)
        
        return img, S
        
        
    def _JPEG2(self, imgName):
        """
        指标十一：论文No-Reference Image Quality Assessment forJPEG/JPEG2000 Coding
        :param imgName: 图像的名称
        :return: 模糊度分数
        """
        S = self._JPEG(imgName)
        SS = 1 + 4 / (1 + math.exp((-1.0217) * (S-3)))
    
        return SS
    
    def _Gaussian_Laplacian(self, imgName):
        '''
        指标十二：
        对采集到的人脸图像进行如下处理：
        1.高斯模糊去噪，
        2.转换灰度图，
        3.在此图像上利用拉普拉斯算子滤波，
        4.直方图归一化映射到0-255，
        5.求均值方差，方差的阈值为300
        '''
        strPath = os.path.join(self.strDir, imgName)
        img = cv2.imread(strPath)
        #img = self._preprocess(imgName, do_gray = False)
        # 高斯滤波
        gauss_blur = cv2.GaussianBlur(img,(3,3),0)
        # 使用线性变换转换输入数组元素成8位无符号整型 归一化为0-255
        transform = cv2.convertScaleAbs(gauss_blur)
        # 灰度化
        grey = cv2.cvtColor(transform, cv2.COLOR_BGR2GRAY)
        # 使用3x3的Laplacian算子卷积滤波
        grey_laplace = cv2.Laplacian(grey, cv2.CV_16S, ksize=3)
        # 归一化为0-255
        resultImg = cv2.convertScaleAbs(grey_laplace)
        
        # 计算均值和方差
        mean, std = cv2.meanStdDev(resultImg)
        blurPer = std[0][0] ** 2
        
        return img, blurPer
        
if __name__ == "__main__":
    # BlurScore = BlurScore(strDir=r"/Users/jinyufeng/Downloads/20230801_faces_v2_2619id")
    BlurScore = BlurScore(strDir=r"/Users/jinyufeng/Downloads/20230801_v2_align_faces_v2",
                          # saveDir=r"/Users/jinyufeng/Downloads/20230801_v2_align_faces_v2_SMD"
                          # saveDir=r"/Users/jinyufeng/Downloads/20230801_v2_align_faces_v2_SMD2"
                          # saveDir=r"/Users/jinyufeng/Downloads/20230801_v2_align_faces_v2_JPEG"
                          # saveDir=r"/Users/jinyufeng/Downloads/20230801_v2_align_faces_v2_Thenengrad"
                          # saveDir=r"/Users/jinyufeng/Downloads/20230801_v2_align_faces_v2_Variance"
                          saveDir=r"/Users/jinyufeng/Downloads/20230801_v2_align_faces_v2_Gaussian_Laplacian"
                          )
    BlurScore.getAllScore()