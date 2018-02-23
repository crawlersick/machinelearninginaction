from numpy import *
def loaddata():
    dataMat=[]
    labelMat=[]
    fr=open('../testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(x):
    return 1.0/(1+exp(-x))

def gradAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    #convert to numpy matrix data type
    m,n=shape(dataMatrix)
    alpha=0.001
    maxCycle=500
    weights=ones((n,1))
    for k in range(maxCycle):
        h=sigmoid(dataMatrix*weights)
        error=labelMat-h
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights


def plotBestFit(wei,dm,lm):
    import matplotlib.pyplot as plt
    #weights=wei.getA()
    weights=array(wei)
    dataArr=array(dm)
    n=shape(dataArr)[0]
    xcord1=[]
    xcord2=[]
    ycord1=[]
    ycord2=[]
    for i in range(n):
        if int(lm[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

dm,lm=loaddata()
#wei=gradAscent(dm,lm)
#print(wei)
#plotBestFit(wei,dm,lm)


def stocGradAscent0(dataMatIn,classLabels):
    dataMatrix=array(dataMatIn)
    labelMat=classLabels
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=labelMat[i]-h
        weights=weights + alpha * error * dataMatrix[i]
    return weights

#wei=stocGradAscent0(dm,lm)
#print(wei)
#plotBestFit(wei,dm,lm)


def stocGradAscent1(dataMatIn,classLabels,numIter=150):
    dataMatrix=array(dataMatIn)
    labelMat=classLabels
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            rand=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[rand]*weights))
            error=labelMat[rand]-h
            weights=weights + alpha * error * dataMatrix[rand]
            del(dataIndex[rand])
    return weights        
wei=stocGradAscent1(dm,lm)
print(wei)
plotBestFit(wei,dm,lm)
