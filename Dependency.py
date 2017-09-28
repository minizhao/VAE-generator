# -*- coding: utf-8 -*-
import gensim
import pickle as pkl
import jpype
"""
Created on Tue Sep 26 15:39:40 2017
"""
def add_dependency_to_vec(words_vec,sentences_data,idx_2_w):
    '''
    参数说明
    words_vec：词向量数据，有如下属性:( words_vec.vocab :表示包含的所以词的词典,\
          如果 有个词 w 在words_vec.vocab中 ,可用 words_vec[w] 获得其词向量（100维）)
    sentences_data：文本数据包含8w条左右的数据，如下所示：
              [ [1,3,6,7,9,0,0],
                [5,6,7,9,1,3,0],
                ....
                ....
                ....
              ]
             每句话由一个list表示，其中每一个数字表示词的编号
    idx_2_w:编号转词的词典，如词w编号5，可以根据idx_2_w[5]得到词w
    '''

    windowSize = 3
    jpype.startJVM(jpype.getDefaultJVMPath(),"-Djava.class.path=/home/gpu2/zhaozd/hanlp1/hanlp-1.2.9.jar:/home/gpu2/zhaozd/hanlp1", "-Xms1g", "-Xmx1g") # gai!!!
    HanLP = jpype.JClass('com.hankcs.hanlp.HanLP')
    #windowSize = 4  #依存关系最多提取10个孩子
    dep = []
    depLabel=[]
    f0 = open('../vae/depLabel.txt','r') #生成依存标签-编号字典 gai!!!
    i0 = 0
    for line in f0.readlines():
        dep.append(line.strip('\n'))
        depLabel.append(i0)
        i0+=1
    dictDep = {}
    dictDep=dict(zip(dep,depLabel))
    print ('依存关系类别词典建立完毕')
    mapping=words_vec   
    print ('词向量匹配中...')
    print ('提取依存序列')
    docIndex = [] #总
    allWord=[]
    for count in range(len(sentences_data)) :
        if count % 1000 ==0:
            print (count)
        sentIndex = [] #句存储list 包含每个词及他们的孩子编号序列
        sentWord = []        
        l = ' '.join([idx_2_w[int(x)] for x in sentences_data[count]])
        if l.strip('\n') != '' :
            linewrite0 = HanLP.parseDependency(l)
            lineData = linewrite0.toString().split('\n')
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
                elif ll.split('\t')[1] not in mapping.vocab:
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
                        elif lll.split('\t')[1] not in mapping.vocab:
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
                sentWord.append(ll.split('\t')[1])
        docIndex.append(sentIndex)
        allWord.append(sentWord)
    return docIndex,allWord

  

if __name__ == "__main__":
    #===load data ===#
    
    model = gensim.models.Word2Vec.load('gensim_model')
    words_vec=model.wv
    
    pkl_file = open('word2vec_file/data_10w.pkl', 'rb')
    sentences_data=pkl.load(pkl_file)
    
    pkl_file = open('word2vec_file/w_2_idx.pkl', 'rb')
    w_2_idx=pkl.load(pkl_file)
    
    
    idx_2_w = dict(zip(w_2_idx.values(), w_2_idx.keys()))
 
    
    print ("load data done")
    docIndex,allWord=add_dependency_to_vec(words_vec,sentences_data,idx_2_w)
    
    with open('word2vec_file/added_dependency_vec.pkl','wb') as output:
        pkl.dump(docIndex, output)
    with open('word2vec_file/added_dependency_word.pkl','wb') as output:
        pkl.dump(allWord, output)
    
    print (len(docIndex))
    print (len(allWord))
   


"""
def createListCSV(fileName, dataList):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for data in dataList:
            for item in data:
                csvWriter.writerow(item)
            csvWriter.writerow('1')
        csvFile.close
"""
