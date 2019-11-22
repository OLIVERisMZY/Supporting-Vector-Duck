import math
import pandas as pd
import numpy as np
import random
from sklearn import svm
import matplotlib.pyplot as plt

df = pd.read_csv('D:\可爱臭鸭鸭\支持向量机数据.csv')#需要添加目标文件目录
train_data = np.array(df.iloc[:,0:3])
attributeMap={}
attributeMap['no']=-1
attributeMap['yes']=1
#============数据化=================
for i in range(len(train_data)):
        train_data[i,2]=attributeMap[train_data[i,2]]
train_label=train_data[:,2]
train_data=np.delete(train_data,2,1)#第三个数1表示列，0表示行
m,n=np.shape(train_data)


#核函数=================================================
def kernelTrans(x1, x2, kTup):
   #X1与X2都是一维数组
   len_x=len(x1)
   K=0
   if kTup=='lin': #线性核函数
        K =np.dot(x1.reshape(1,len_x),x2.reshape(len_x,1))
   elif kTup=='rbf':#高斯核函数
        x3=x1-x2
        K=0
        for i in range(len(x3)):
            x3[i]=x3[i]*x3[i]
        K=np.exp(np.sum(x3)/(-1*(1.3)**2)) #设置高斯核的带宽是2
   else:
        raise NameError('无法识别核函数名称')
   return K


#=====================================================
class optStruct:
    def __init__(self,data, label, C, toler, kTup):  # 存储各类参数
        self.data = data   #数据特征
        self.label = label #数据类别
        self.C = C         #软间隔参数C，参数越大，非线性拟合能力越强
        self.tol = toler   #停止阀值
        self.count =len(data) #数据行数
        self.alphas = np.zeros((self.count,1))
        self.E = np.zeros((self.count,1))
        self.b = 0
        self.K = np.zeros((self.count,self.count)) #核函数的计算结果
        for i in range(self.count):
            for j in range(self.count):
               self.K[i,j] = kernelTrans(self.data[i], self.data[j], kTup)
'''
#debug======核函数测试===========
K = np.zeros((m,m)) #核函数的计算结果
for i in range(m):
    for j in range(m):
       K[i,j] = kernelTrans(train_data[i], train_data[j], 'lin')
print(K)
'''
#计算Ek=========================================
def calcEk(oS, k):
    sum=0
    for j in range(len(oS.alphas)):
      sum+=oS.alphas[j]*oS.label[j]*oS.K[j,k]
    sum+=oS.b
    sum-=oS.label[k]
    Ek=sum
    return Ek
#计算支持向量的Ek================================
def cal_svEk(oS,k):
    sum=0
    all_Support_Vector=[]
    for i in range(oS.count):
        if oS.alphas[i] > 0 and oS.alphas[i] < oS.C:
            all_Support_Vector.append(i)
    if len(all_Support_Vector)==0:
        sum += oS.b
        sum -= oS.label[k]
        Ek = sum
        return Ek
    else:
      for sv in all_Support_Vector:
        sum+=oS.alphas[sv]*oS.label[sv]*oS.K[k,sv]
      sum+=oS.b
      sum-=oS.label[k]
      Ek=sum
      return Ek
#计算gk=========================================
def calcgk(oS, k): #i=1,2两条数据
    sum=0
    for j in range(len(oS.alphas)):
      sum+=oS.alphas[j]*oS.label[j]*oS.K[j,k]
    sum+=oS.b
    gk=sum
    return gk


#检验ai是否满足KT条件===============================
def innerL(i, oS):

    #如果不满足KT条件=================
    Fit_KT=True
    if (oS.alphas[i]==0 and oS.label[i]*calcgk(oS, i)<1):
        Fit_KT = False
    elif (oS.alphas[i]>0 and oS.alphas[i]<oS.C and oS.label[i]*calcgk(oS, i)!=1):
        Fit_KT = False
    elif (oS.alphas[i]==oS.C and oS.label[i]*calcgk(oS, i)>1):
        Fit_KT = False
    return Fit_KT

#====SMO算法================================
#输入参数：数据特征，数据类别，参数C，阀值toler，
#最大迭代次数，核函数（默认线性核）
def smoP(train_data, train_label, C, toler, maxIter,kTup='lin'):
    oS = optStruct(train_data, train_label,C,toler,kTup)
    #更新E==================
    for i in range(oS.count):
        oS.E[i]=calcEk(oS,i)
    # ==================
    iter = 0#迭代初始化
    # 迭代开始===============
    while iter<maxIter:
        all_Support_Vector=[]
        satisfy_Support_Vector=[]
        satisfy_index=0
        #=========找出所有的支持向量=========================
        for i in range(oS.count):
            if oS.alphas[i]>0 and oS.alphas[i]<oS.C:
                all_Support_Vector.append(i)
        # =========如果存在支持向量=========================
        if len(all_Support_Vector)!=0:
          for sv in all_Support_Vector:
            fit_or_not=innerL(sv,oS)
            if fit_or_not==False:
                satisfy_Support_Vector.append(sv)
          # =========如果支持向量中有不满足KT条件的=========================
          if(len(satisfy_Support_Vector)!=0):
              num = int(np.random.uniform(0, len(satisfy_Support_Vector)))
              satisfy_index=satisfy_Support_Vector[num]
              update = True
          # =========如果支持向量中没有不满足KT条件的
          elif(len(satisfy_Support_Vector)==0):
              satisfy_all_vector=[]
              for i in range(oS.count):
                fit_or_not = innerL(i, oS)
                if fit_or_not == False:
                    satisfy_all_vector.append(i)
              if(len(satisfy_all_vector)!=0):
                  num = int(np.random.uniform(0, len(satisfy_all_vector)))
                  satisfy_index = satisfy_all_vector[num]
                  update = True
        else:
            satisfy_all_vector = []
            for i in range(oS.count):
                fit_or_not = innerL(i, oS)
                if fit_or_not == False:
                    satisfy_all_vector.append(i)
            if (len(satisfy_all_vector) != 0):
                num = int(np.random.uniform(0, len(satisfy_all_vector)))
                satisfy_index = satisfy_all_vector[num]
                update=True
        #=========如果实在找不到那就随机一个吧
        if update==False:
            satisfy_index=int(np.random.uniform(0,oS.count))
        # 找到了第一个变量=====================
        first_index=satisfy_index
        second_index_list=[]
        E1=calcEk(oS,first_index)
        without=[]
        for j in range(oS.count):
            if first_index!=j:
                without.append(j)
        D_value=[]
        for j in without:
            value=np.abs(E1-oS.E[j])
            D_value.append(value)
        for i in range(len(D_value)):
            if D_value[i]==np.max(D_value):
                second_index_list.append(without[i])
        num=int(np.random.uniform(0, len(second_index_list)))
        second_index = second_index_list[num]
        i=first_index
        j=second_index
        print('SMO的两个向量是:'+str(i)+','+str(j))
        #求解两个变量的最优化问题============================================
        Ei = calcEk(oS, i)
        Ej = calcEk(oS, j)

        # 更新alpha值===================================
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.label[i] != oS.label[j]):
            L = max(0, alphaJold - alphaIold)
            H = min(oS.C, oS.C + alphaJold - alphaIold)
        else:
            L = max(0, alphaJold - alphaIold - oS.C)
            H = min(oS.C,alphaJold - alphaIold)
            '''
        if L == H and iter!=1:
            print("L==H")
            return 0
            '''
        eta = oS.K[i, i] + oS.K[j, j] - 2.0 * oS.K[i, j]
        aj_new_unc = alphaJold + oS.label[j] * (Ei - Ej) / eta
        # 选取ajnew的值=============================
        ajnew = 0
        if aj_new_unc > H:
            ajnew = H
        elif aj_new_unc <= H and aj_new_unc >= L:
            ajnew = aj_new_unc
        elif aj_new_unc < L:
            ajnew = L
        oS.alphas[j] = ajnew
        '''
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        '''
        ainew = alphaIold + oS.label[i] * oS.label[j] * (alphaJold - ajnew)
        oS.alphas[i] = ainew
        # 更新阈值==================================================
        b1 = -Ei - oS.label[i] * oS.K[i, i] * (ainew - alphaIold) - oS.label[j] * oS.K[j, i] * (
                    ajnew - alphaJold) + oS.b
        b2 = -Ej - oS.label[j] * oS.K[j, j] * (ajnew - alphaJold) - oS.label[j] * oS.K[j, j] * (
                    ajnew - alphaJold) + oS.b
        bnew = 0
        if (ainew > 0 and ainew < oS.C) and (ajnew > 0 and ajnew < oS.C):
            bnew = b1
        elif (ainew == 0 or ainew == oS.C or ajnew == 0 or ajnew == oS.C):
            bnew = (b1 + b2) / 2
        oS.b = bnew
        # 更新对应的Ei值===================
        oS.E[i] = cal_svEk(oS, i)
        oS.E[j] = cal_svEk(oS, j)
        #停机条件判断===================
        print('支持向量集为：' + str(all_Support_Vector))
        '''
        sum=0
        stop=False
        for i in range(oS.count):
            sum+=oS.alphas[i]*oS.label[i]
        if sum==0:
            ga=oS.label[i]*calcgk(oS,i)
            if ga>=1 and oS.alphas[i]==0:
                stop=True
            elif ga==1 and oS.alphas[i]>0 and oS.alphas[i]<oS.C:
                stop=True
            elif ga <= 1 and oS.alphas[i] == oS.C:
                stop = True
        if stop==True:
            print('达到停机条件')
            break
        '''
        iter+=1
    print('循环结束，总共循环了'+str(iter+1)+'次')
    return oS.alphas,oS.b,oS.K,all_Support_Vector


alpha,beta,K,sv=smoP(train_data, train_label,10, 0.0001, 2000,'lin')
result=[]
for i in range(len(train_data)):
    sum=0
    for j in range(len(train_data)):
        sum+=alpha[j]*train_label[j]*K[i,j]
    sum+=beta
    if sum>0:
        sum=1
    else:
        sum=-1
    result.append(sum)


print('预测值为：'+str(result))
print('真实值为：'+str(train_label))
D=train_label-result
sum=0
for i in range(len(train_data)):
    if(train_label[i]!=result[i]):
        sum+=1
e=sum/len(train_label)
print('正确率为：'+str((100-e*100))+'%')



