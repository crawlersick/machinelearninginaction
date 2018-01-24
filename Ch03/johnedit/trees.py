from math import log
def calcShannonEnt(dataset):
    numEntries=len(dataset)
    labelCounts={}
    for featVec in dataset:
        currentLabel=featVec[-1]
#        print(currentLabel)
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt
def createDataSet():
    dataset=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataset,labels
myDat,labels=createDataSet()
print(myDat)
x=calcShannonEnt(myDat)
print(x)
myDat[0][-1]='maybe'
x=calcShannonEnt(myDat)
print(x)
print('above is part1---------------------------------------------------------------')
def splitDataset(dataset,axis,value):
    retset=[]
    for featv in dataset:
        if featv[axis] == value:
            reducefeatv=featv[:axis]
            reducefeatv.extend(featv[axis+1:])
            retset.append(reducefeatv)
    return retset
myDat[0][-1]='yes'
myDat_st=splitDataset(myDat,0,1)
print('testing result of splitDataset(myDat,0,1)')
print(myDat_st)
myDat_st=splitDataset(myDat,0,0)
print('testing result of splitDataset(myDat,0,0)')
print(myDat_st)
    
def chooseBestFeatSplit(dataset):
    numFt=len(dataset[0])-1
    baseEntropy=calcShannonEnt(dataset)
    bestInfoGain=0.0
    bestFeat=-1
    for i in range(numFt):
        featList=[e[i] for e in dataset]
        uniqFeat=set(featList)
        newEntropy=0.0
        for value in uniqFeat:
            subDataset=splitDataset(dataset,i,value)
            prob=len(subDataset)/float(len(dataset))
            newEntropy+=prob*calcShannonEnt(subDataset)
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeat=i
    return bestFeat

print('testing result of choose best feature:')
bf=chooseBestFeatSplit(myDat)
print(bf)

import operator
def majorityCnt(classlist):
    classCount={}
    for vote in classlist:
        if vote not in classCount.keys(): classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


print('testing result of majority count:')
tt=majorityCnt([1,1,0,0,0]);
print(tt)

def createTree(dataset,labels):
    classList=[e[-1] for e in dataset]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataset[0])==1:
        return majorityCnt(classList)

