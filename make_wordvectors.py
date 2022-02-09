from gensim.models import word2vec
import pickle
import pandas as pd
import numpy as np
import MeCab 

#NeologDを利用した各文書の平均ベクトルを求める
model = word2vec.Word2Vec.load('./data/models/neolog.model')

with open('./data/pickles/train_sent.pkl','rb') as f:
        sentences = pickle.loads(f.read())

wakati = []
mt = mt_neolog = MeCab.Tagger('-Owakati -d /home/koichi/src/neologd/dic/')
    
for i in range(len(sentences)):
    result = mt.parse(sentences[i])
    wakati.append(result)

print(len(wakati))

vectors = np.empty((0,100))
feature_vec = np.zeros(100)
count = 0

wakati_term = []
words = []
for line in wakati:    
    wakati_term = line.split()
    #words.append(wakati_term)


    for i in range(len(wakati_term)):
        if wakati_term[i] in model.wv:
            fe = np.reshape(model.wv[wakati_term[i]],(1,100))
            feature_vec = np.add(feature_vec, fe)
            #print("feature vec=",feature_vec.shape) #shape (1,100)
            count = count + 1
        else:
            print("無かったよ:",wakati_term[i])
            pass

    if count > 0:
        average_vec = feature_vec/count
    else:
        average_vec = feature_vec
    
    vectors = np.append(vectors,average_vec,axis=0)


print(len(vectors))
print(vectors) #1913x100

with open('./data/pickles/train_vec.pkl','wb') as f:
    f.write(pickle.dumps(vectors))
    #sentences.pkl に sentences を保存
