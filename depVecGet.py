# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:46:18 2017

@author: Administrator
import depVecGet。py 
调用：start（4,'D:\\lmqnlp\\HanLP\\tools_to_multi_files\\all_dataSet_split.txt.vec','D:\\lmqnlp\\LSTM\\FXGtemp1.txt',''D:\\lmqnlp\\LSTM\\result1w_window9.csv''）
import 
"""
#! usr/bin/python
# -*- coding:utf-8 -*-
from jpype import *
import os
import re
import chardet
import csv 
def start(windowSize,fvec,ftext,fout):
    '''
    【每个词】标签对应:
    自己词向量 +  [孩子index + 依存类型 + 孩子词向量] * windowSize
    维数 100   + [ 1        +   1      +    100   ] * 4  = 509
    若无依存结构信息 即无孩子 则
    
    验证集对应：FXGtemp1 and FXFnew 3000起
    '''
    
    ################
    #提取dep标签####
    ###############
    startJVM(getDefaultJVMPath(),"-Djava.class.path=D:\lmqnlp\hanlp1\hanlp-1.2.9.jar;D:\lmqnlp\hanlp1", "-Xms1g", "-Xmx1g") # gai!!!
    HanLP = JClass('com.hankcs.hanlp.HanLP')
    #windowSize = 4  #依存关系最多提取10个孩子
    dep = []
    depLabel=[]
    f0 = open('D:\\lmqnlp\\LSTM\\depLabel.txt','r') #生成依存标签-编号字典 gai!!!
    i0 = 0
    for line in f0.readlines():
        dep.append(line.strip('\n'))
        depLabel.append(i0)
        i0+=1
    dictDep = {}
    dictDep=dict(zip(dep,depLabel))
    print '依存关系类别词典建立完毕'
    
    f2 = open(fvec,'r') #词向量文件
    source=(f2.readlines())[1:] #可以读完
    word=[]
    vec=[]#6296个list,每个list100个str类型的数据
    for item in source:
        word.append((item.split(' ')[0]))
        vec0 = [float(ii) for ii in item.split(' ')[1:]]
        vec.append(vec0)
    
    mapping={}#必须有这个，不然出现NameError
    mapping=dict(zip(word,vec))
    
    print '词向量匹配中...'
    
    f1 = open(ftext,'r')   #抽取出片段的句子+编号数据
        
    #listVal = []#标点符号词向量准备
    #for i in range(100):
    #   listVal.append(float(0))
    
       
    print '提取依存序列'
    docIndex = [] #总
    for count in range(10000) :# gai..............................................................
        print count 
        sentIndex = [] #句存储list 包含每个词及他们的孩子编号序列
      
        l = f1.readline()
        if l.strip('\n') != '' :
            linewrite0 = HanLP.parseDependency(l.split('\t').decode('utf-8'))
            lineData = linewrite0.toString().encode('utf-8').split('\n')
            lineData.pop()
            for ll in lineData: #针对每个词提取依存孩子序列
                if ll.strip() == '':
                    continue

    
                wordIndex = [] #词存储lixt 每个孩子依次加入
                '''
                该词词向量
                '''
                if ll.split('\t')[7] == '标点符号' :
                    for count in range(100) :
                        wordIndex.append(float(0)) #标点词向量设置为0 list ？
    #                wordIndex.append(listVal) #标点词向量设置为0 list ？
    #               wordSeq.append(listVal) #标点词向量设置为0 list ？
                elif ll.split('\t')[1] not in mapping.keys():
                    for count in range(100) :
                        wordIndex.append(float(0)) #生僻或怪符号词向量设置为0 list ？
                else:
                    for x in mapping[ll.split('\t')[1]]:
                        wordIndex.append(x) #该词词向量
    #                wordIndex.append(mapping[ll.split('\t')[1]]) #该词词向量
   
                '''
                依存结构：该词孩子index+依存类型+孩子词向量
                '''
                flag = 0
                i = 0 
                for lll in lineData:
                    wordSeq = []                
                    if lll.split('\t')[6] == ll.split('\t')[0] and i< windowSize :
                        flag = 1
                        if lll.split('\t')[7] not in dictDep.keys() :
                            dictDep[lll.split('\t')[7]] = len(dictDep)
                        wordIndex.append(int(lll.split('\t')[0])) #孩子编号
                        wordIndex.append(dictDep.get(lll.split('\t')[7])) #孩子与当前词的依存关系类别
    #                    wordSeq.append(int(lll.split('\t')[0])) #孩子编号 
    #                    wordSeq.append(dictDep.get(lll.split('\t')[7])) #孩子与当前词的依存关系类别
                        if lll.split('\t')[7] == '标点符号' :
                            for count in range(100) :
                                wordIndex.append(float(0)) #标点词向量设置为0 list ？
    #                        wordIndex.append(listVal) #标点词向量设置为0 list ？
    #                        wordSeq.append(listVal) #标点词向量设置为0 list ？
                        elif lll.split('\t')[1] not in mapping.keys():
                            for count in range(100) :
                                wordIndex.append(float(0)) #生僻或怪符号词向量设置为0 list ？
                        else:
                            for x in mapping[lll.split('\t')[1]] :
                                wordIndex.append(x) #该词词向量
    #                        wordIndex.append(mapping[lll.split('\t')[1]]) #孩子词向量
    #                        wordSeq.append(mapping[lll.split('\t')[1]]) #孩子词向量                    
    #                    wordIndex.append(wordSeq)
                        i+=1
                for count in range(102*(windowSize-i)):
                    wordIndex.append(float(0))
    #            if flag == 0 : #如果该词无孩子 则孩子序列则为0
    #                wordIndex.append(0)
                sentIndex.append(wordIndex) #注意！！依存句法结构提取出来第一词编号为1，而此处第一词编号 为 0 但index同 且提取出的孩子序号同原来编号      
        docIndex.append(sentIndex)
    
    createListCSV(fout,docIndex)

'''
save
'''

def createListCSV(fileName, dataList):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for data in dataList:
            for item in data:
                csvWriter.writerow(item)
            csvWriter.writerow('1')
        csvFile.close

