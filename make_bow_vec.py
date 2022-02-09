from gensim import corpora, matutils
import pickle
import numpy as np


with open('./data/pickles/wakati_sentences_neolog.pkl','rb') as f:
    sentences = pickle.load(f)
dictionary = corpora.Dictionary.load_from_text('dictionary.txt')
#辞書の読み出し
word_vectors =[]
for s in sentences:
    ss = s.split()
    tmp = dictionary.doc2bow(ss)
    dense = list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
    word_vectors.append(dense)
print(len(word_vectors))
print(word_vectors)
#del_num = [30,64,699,701,708,713,716,719,721] # 欠損リスト 既にデータから抜いてるので良い
#for i in del_num:
#    del word_vectors[i]
word_vectors = np.array(word_vectors)
print(len(word_vectors)) #1913 ほんとに [0,0,0,1,...] のベクトル
with open('./data/pickles/train_vec_bow.pkl','wb') as f:
    f.write(pickle.dumps(word_vectors))
    #sentences.pkl に sentences を保存 （編
