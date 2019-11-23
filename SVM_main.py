import math
import pandas as pd
import numpy as np
import random
from sklearn import svm
import matplotlib.pyplot as plt

df = pd.read_csv('D:\可爱臭鸭鸭\支持向量机数据.csv')#需要添加目标文件目录
all_data = np.array(df.iloc[:,0:3])
attributeMap={}
attributeMap['no']=-1
attributeMap['yes']=1
#============数据化=================
for i in range(len(all_data)):
        all_data[i,2]=attributeMap[all_data[i,2]]
train_data=all_data[0:299,:]
test_data= all_data[300:len(all_data),:]

train_label=train_data[:,2]
test_label=test_data[:,2]
train_data=np.delete(train_data,2,1)#第三个数1表示列，0表示行
test_data=np.delete(test_data,2,1)#第三个数1表示列，0表示行
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
        self.first_index_old=None
        self.second_index_old = None
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
    '''#理论上
    if (oS.alphas[i]==0 and oS.label[i]*calcgk(oS, i)<1):
        Fit_KT = False
    elif (oS.alphas[i]>0 and oS.alphas[i]<oS.C and oS.label[i]*calcgk(oS, i)!=1):
        Fit_KT = False
    elif (oS.alphas[i]==oS.C and oS.label[i]*calcgk(oS, i)>1):
        Fit_KT = False
    '''
    r=oS.E[i]*oS.label[i]
    if(r<oS.tol and oS.alphas[i]<oS.C)or(r>oS.tol and oS.alphas[i]>oS.C):
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
        #更新支持向量集合===========
        all_Support_Vector=[]
        for i in range(oS.count):
          if  oS.alphas[i] > 0 and oS.alphas[i] < oS.C:
              all_Support_Vector.append(i)
        defy_KT_Vector=[]

        #=========找出所有不满足KT的集合=========================
        first_index = None
        for i in range(oS.count):
            if innerL(i,oS)==False:
                defy_KT_Vector.append(i)
        # 如果有支持向量=======
        if len(all_Support_Vector)>0:
          for sv in all_Support_Vector:
              if innerL(sv,oS)==False:
                first_index=sv
                break
          # 如果支持向量中没有违反kt条件的=======
          #从整个数据集进行寻找=================
        if first_index==None and len(defy_KT_Vector)>0 :
           num=np.random.randint(0,len(defy_KT_Vector))
           first_index=defy_KT_Vector[num]
        else:
            num = np.random.randint(0, oS.count)
            first_index = num
        #得到了a1在寻找a2============================
        without=[]
        D_value=[]
        for i in range(oS.count):
            if i != first_index:
                without.append(i)
        #without为a1以外的集合============================
        for i in range(len(without)):
            value=np.abs(oS.E[i]-oS.E[first_index])
            D_value.append(value)
        index=np.argmax(D_value)
        second_index=without[index]
        '''
        # 如果和上一步相同，再找，从0<a<c的子集中找=======
        if second_index==oS.second_index_old:
            Support_Vector=[]
            for i in without:
                if oS.alphas[i] > 0 and oS.alphas[i] < oS.C:
                    Support_Vector.append(i)
            index = np.random.randint(0,len(Support_Vector))
            second_index=without[index]
        '''
        #=========如果和上一步相同,j从支持向量中随机========
        if second_index == oS.second_index_old:
            Support_Vector = []
            for i in without:
                if oS.alphas[i] > 0 and oS.alphas[i] < oS.C:
                    Support_Vector.append(i)
            index = np.random.randint(0, len(Support_Vector))
            second_index = without[index]

        oS.first_index_old  = first_index
        oS.second_index_old = second_index

        i=first_index
        j=second_index
        print('不满足KT条件的个数为：' + str(len(defy_KT_Vector)))
        print('选取优化的两个alpha为:'+str(i)+'+'+str(j))


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
        eta = oS.K[i, i] + oS.K[j, j] - 2.0 * oS.K[i, j]
        if eta>0:  #https://www.cnblogs.com/xxrxxr/p/7538430.html
           aj_new_unc = alphaJold + oS.label[j] * (Ei - Ej) / eta
           # 选取ajnew的值=============================
           ajnew = 0
           if aj_new_unc > H:
               ajnew = H
           elif aj_new_unc <= H and aj_new_unc >= L:
               ajnew = aj_new_unc
           elif aj_new_unc < L:
               ajnew = L
        else:#(eta<=0)
               ajnew = np.min(H,L)

        #更新a2====================================
        oS.alphas[j] = ajnew


        # 更新a1====================================
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
        iter+=1
    print('循环结束，总共循环了'+str(iter+1)+'次')
    return oS.alphas,oS.b,oS.K,all_Support_Vector


alpha,beta,K,sv=smoP(train_data, train_label, 100, 0.001, 800,'lin')

result=[]
for i in range(len(test_data)):
    sum=0
    for j in range(len(train_data)):
        sum+=alpha[j]*train_label[j]*kernelTrans(test_data[i],train_data[j],'lin')
    sum+=beta
    if sum>0:
        sum=1
    else:
        sum=-1
    result.append(sum)


D=test_label-result
sum=0
for i in range(len(test_data)):
    if(test_label[i]!=result[i]):
        sum+=1
e=sum/len(test_label)
print(sv)
print('正确率为：'+str((100-e*100))+'%')
#散点图绘制=====================================
plt.title('the result')

plt.xlabel('x1')
plt.ylabel('x2')
for i in range(len(test_data)):
    if result[i]==1:
       plt.scatter(test_data[i,0],test_data[i,1], s=20, c="#ff1212", marker='+')
    else:
       plt.scatter(test_data[i, 0],test_data[i,1], s=20, c="#cca110", marker='o')
# x: x轴坐标
# y：y轴坐标
# s：点的大小/粗细 标量或array_like 默认是 rcParams['lines.markersize'] ** 2
# c: 点的颜色
# marker: 标记的样式 默认是 'o'
plt.show()


