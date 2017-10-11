# -*- coding: utf-8 -*-
import gensim
import pickle as pkl
import jpype
import sys
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
    jpype.startJVM(jpype.getDefaultJVMPath(),"-Djava.class.path\
            =/home/gpu2/zhaozd/hanlp1/hanlp-1.2.9.jar:/home/gpu2/zhaozd/hanlp1", "-Xms1g", "-Xmx1g") 
    HanLP = jpype.JClass('com.hankcs.hanlp.HanLP')
    #windowSize = 4  #依存关系最多提取10个孩子
    
    dep = []
    depLabel=[]

    with open('../vae/depLabel.txt','r') as f:
    '''生成依存标签-编号字典 '''
        
        for line in f.readlines():
            dep.append(line.strip('\n'))
            depLabel.append(len(depLabel))

    dictDep=dict(zip(dep,depLabel))
    print ('依存关系类别词典建立完毕')


    print ('词向量匹配中...')
    print ('提取依存序列')

    docIndex = [] #总
    allWord=[]

    for count in range(len(sentences_data)) :

        #print progress
        sys.stdout.write('generated:{0}/total:{1}\r'.format(count,len(sentences_data))) 
        sys.stdout.flush()
       
        sentIndex = [] #句存储list 包含每个词及他们的孩子编号序列
        sentWord = []

        sent = ' '.join([idx_2_w[int(x)] for x in sentences_data[count]])

        if sent.strip('\n') != '' :
            #最大熵依存句法分析器

            parse_result = HanLP.parseDependency(sent).toString().split('\n')
           
            #delte laste ''  and ' '
            parse_result.remove('')
            parse_result.remove(' ')

            assert len(parse_result)>0

            for word in parse_result: #针对每个词提取依存孩子序列
              
                wordIndex = [] #词存储lixt 每个孩子依次加入
                '''
                该词词向量
                '''
                if word.split('\t')[7] == '标点符号' :
                    wordIndex.extend(zeros(100)) #标点词向量设置为0 list ？

                else:
                    wordIndex.extend(words_vec[word.split('\t')[1]]) #该词词向量
   
                '''
                依存结构：依存类型+孩子词向量
                '''
                flag = 0
                child_num = 0 
                for candidate_word in parse_result:

                    if candidate_word.split('\t')[6] == word.split('\t')[0] and child_num< windowSize :
                        flag = 1
                        if candidate_word.split('\t')[7] not in dictDep.keys() :
                            dictDep[candidate_word.split('\t')[7]] = len(dictDep)

                        wordIndex.append(dictDep.get(candidate_word.split('\t')[7])) #孩子与当前词的依存关系类别
  

                        if candidate_word.split('\t')[7] == '标点符号' :
                           
                            wordIndex.extend(zeros(100)) #标点词向量设置为0 list ？
                        #split word
                        else:
                            wordIndex.extend(words_vec[candidate_word.split('\t')[1]]) #该词词向量
                    
                        child_num+=1

                wordIndex.append(zeros(102*(windowSize-child_num)))
                sentIndex.append(wordIndex) 
                sentWord.append(word.split('\t')[1])
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
   

