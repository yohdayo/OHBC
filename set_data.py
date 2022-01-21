#excelデータを整形して配列に格納するプログラム
#ラベルも保存
import glob
import pandas as pd
from itertools import chain
import pickle
import numpy as np


def make_traindata():
    sentence = []
    sentences = []
    labels = []
    print("excel => array")
    #ディレクトリ内の *.xlsx をリストの形で取得
    file_list = glob.glob('./data/chronologies_with_label/saito/*.xlsx')
    ###
    #(2018.7.9.13時時点クロノロ)県南西部医療圏活動拠点本部(川崎医科大学附属病院)[165x5]
    #(2018.7.20時点クロノロ)岡山県南西部医療圏活動拠点本部（倉敷市保健所内）    [786x5]
    #活動記録                                                          [793x5]
    #県南西部災害保健医療調整本部（7.24-31）                               [182x5]
    #を取得．全てシートは1枚．5列で構成される
    ###
    for filename in file_list:
        df = pd.read_excel(filename)
        print(filename)
        sentence = df["内容"]
        #excelシートの内容部分を抽出
        sentences.append(sentence)
        #4 * x の配列
    
    for filename in file_list:
        df = pd.read_excel(filename)
        #print(filename)
        label = df["緊急度"]
        #excelシートの内容部分を抽出
        labels.append(label)
        #4 * x の配列
    print("抜き出した")

    sentences_one = list(chain.from_iterable(sentences))
    labels_one = list(chain.from_iterable(labels))

    for s in sentences_one:
        if isinstance(s,float):
            sentences_one.remove(s)

    for s in labels_one:
        if isinstance(s,float):
            labels_one.remove(s)

    #リストを1列にする．nanを除いた1923の配列

    print(len(sentences_one))
    print(len(labels_one))

    return sentences_one, labels_one

def labelWOnumber(sentences, array):
    labels_num = []
    for i,kanji in enumerate(array):
        if kanji == "低":
            labels_num.append([1,0])
        elif kanji == "中":
            labels_num.append([1,0])
        elif kanji == "高":
            labels_num.append([0,1])
        else:
            print(i)
            del sentences[i]
#重要度の低中高を数字に．それ以外のデータは削除
#30,64,699,701,708,713,716,719,721が欠損地になる
        
    print(len(labels_num))
    print(len(sentences))
            
    labels_num = np.array(labels_num)
    return sentences, labels_num 



if __name__ == '__main__': 
    labels = []
    sentences = []
    labels_num = []
    sentences, labels = make_traindata()
    train_sent, train_label = labelWOnumber(sentences, labels)



    with open('./data/pickles/train_sent.pkl','wb') as f:
        f.write(pickle.dumps(train_sent))
        #sentences.pkl に new_sentence_array を保存

    with open('./data/pickles/train_label_two.pkl','wb') as f:
        f.write(pickle.dumps(train_label))
        #sentences.pkl に new_sentence_array を保存
