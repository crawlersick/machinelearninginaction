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
labels_o=labels[:]
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
    bestFeat=chooseBestFeatSplit(dataset)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featVals=[e[bestFeat] for e in dataset]
    uniqVals=set(featVals)
    for value in uniqVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataset(dataset,bestFeat,value),subLabels)
    return myTree
print('testing create tree:')
tree_t=createTree(myDat,labels)
print(tree_t)

print('=========================start the plot=======================')
import matplotlib.pyplot as plt
decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    tcreatePlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',va='center',ha="center",bbox=nodeType,arrowprops=arrow_args)

def tcreatePlot():
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    tcreatePlot.ax1=plt.subplot(111,frameon=False)
    plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

#tcreatePlot()

def getNumLeafs(myTree):
    numLeafs=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else: numLeafs+=1
    return numLeafs
def getTreeDepth(myTree):
    maxDepth=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__=='dict':
            thisDepth=getTreeDepth(secondDict[key])+1
        else:
            thisDepth=1
        if thisDepth > maxDepth:maxDepth=thisDepth
    return maxDepth

print('---testing leafs and depth---------------;')
print(type(tree_t).__name__)
print(tree_t)
leafs=getNumLeafs(tree_t)
dep=getTreeDepth(tree_t)
print(leafs)
print(dep)

def plotMidText(cntrPt,parentPt,txtString):
    xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    tcreatePlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #tcreatePlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

print('---above create the tree plot graph, focus on the object plt.. but actually i do not have much idea about this plot class......---------------;')
#createPlot(tree_t)

#use this tree to do the classfiy on specify item!
print('use this tree to do the classfiy on specify item! ')
def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featInd=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featInd]==key:
            if type(secondDict[key]).__name__=='dict':
                classlabel=classify(secondDict[key],featLabels,testVec)
            else:
                classlabel=secondDict[key]
    return classlabel
print(labels_o)
classres=classify(tree_t,labels_o,[1,0])
print(classres)

def storeTree(t,f):
    import pickle
    fw=open(f,'wb')
    pickle.dump(t,fw)
    fw.close

def grabTree(f):
    import pickle
    fr=open(f,'rb')
    return pickle.load(fr)

storeTree(tree_t,"tree_t_p")
tree_recover=grabTree("tree_t_p")
print(tree_recover)

print('now we have the last test on test data lenses.txt')
fr=open('../lenses.txt')
lenses=[line.strip().split('\t') for line in fr.readlines()]
lenlabels=['age','prescript','astigmatic','tearRate']
lentree=createTree(lenses,lenlabels)
print(lentree)
createPlot(lentree)
