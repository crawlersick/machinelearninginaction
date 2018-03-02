from numpy import *
import matplotlib.pyplot as plt
def loadDataSet(fileName):
    dataMat=[];
    labelMat=[];
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj=H
    if L>aj:
        aj=L
    return aj

dataArr,labelArr=loadDataSet('../testSet.txt')
print(dataArr)
print(labelArr)

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=mat(dataMatIn);
    labelMat=mat(classLabels).transpose()
    b=0
    m,n=shape(dataMatrix)
    alphas=mat(zero((m,1)))
    it=0
    while(it<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            fXi=float(multiply(alphas,labelMat).T * \
                      (dataMatrix*dataMatrix[i,:].T))+b
            Ei=fXi-float(labelMat[i])

