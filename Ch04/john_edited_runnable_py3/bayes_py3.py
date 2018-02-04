

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataset):
    vocabSet=set([])
    for document in dataset:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word:%s is not my vocabulary!"%word)

    return returnVec

postmsgs,postclasses=loadDataSet()
vlist=createVocabList(postmsgs)
print(vlist)
wvlist=setOfWords2Vec(vlist,postmsgs[0])
print(wvlist)

from numpy import *
def trainNB0(trainMat,trainCate):
    numTrainDocs=len(trainMat)
    numWords=len(trainMat[0])
    pAbusive=sum(trainCate)/float(numTrainDocs)
    p0Num=ones(numWords)
    p1Num=ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCate[i]==1:
            p1Num+=trainMat[i]
            p1Denom+=sum(trainMat[i])
        else:
            p0Num+=trainMat[i]
            p0Denom+=sum(trainMat[i])
    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive
testmat=[]
for msg in postmsgs:
    testmat.append(setOfWords2Vec(vlist,msg))

print(testmat)
p0V,p1V,pAb=trainNB0(testmat,postclasses)
print(p0V)
print(p1V)
print(pAb)

def classfyNB(vectoverify,p0V,p1V,pAb):
    p1=sum(vectoverify*p1V)+log(pAb)
    p0=sum(vectoverify*p0V)+log(1.0-pAb)
    if p1>p0:
        return 1
    else:
        return 0

testEntry=['love','my','dalmation']
thisDoc=array(setOfWords2Vec(vlist,testEntry))
print (testEntry,'classified as: ',classfyNB(thisDoc,p0V,p1V,pAb))
testEntry=['stupid','garbage']
thisDoc=array(setOfWords2Vec(vlist,testEntry))
print (testEntry,'classified as: ',classfyNB(thisDoc,p0V,p1V,pAb))

def bagOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
        else:
            print("the word:%s is not my vocabulary!"%word)

