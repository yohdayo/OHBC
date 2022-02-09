#excelデータを整形して配列に格納するプログラム
import glob
import pandas as pd
from itertools import chain
import pickle
import MeCab

def excel2array():
    print("excel => array")
    #ディレクトリ内の *.xlsx をリストの形で取得
    file_list = glob.glob('./data/chronologies/*.xlsx')
    ###
    #(2018.7.9.13時時点クロノロ)県南西部医療圏活動拠点本部(川崎医科大学附属病院)[165x5]
    #(2018.7.20時点クロノロ)岡山県南西部医療圏活動拠点本部（倉敷市保健所内）    [786x5]
    #活動記録                                                          [793x5]
    #県南西部災害保健医療調整本部（7.24-31）                               [182x5]
    #を取得．全てシートは1枚．5列で構成される
    ###
    for filename in file_list:
        df = pd.read_excel(filename)
        sentence = df["内容"]
        #excelシートの内容部分を抽出
        sentences.append(sentence)
        #4 * x の配列

    sentences_one = list(chain.from_iterable(sentences))
    for s in sentences_one:
        if isinstance(s,float):
            sentences_one.remove(s)
    #リストを1列にする．nanを除いた1923の配列
    return sentences_one

def excel2array2022():
    # 竹内
    # set_data.pyで作った文を使う
    # (理由) 文書かタグが不適切な場合は文書ごと排除した結果整理した文
    # 　　　データ構造として，文(x)とタグ(y)が別のpickleにまとめられてるため
    #      不整合を起こさないように同じ文で処理する
    with open('./data/pickles/train_sent.pkl','rb') as sent:
        train_sent = pickle.load(sent)
        print(train_sent)
        
        return train_sent


def preprocessing(array):
    print("preprocessing")
    #文字列整理の関数
    
    #sentencesを全部全角に直す
    #zen_sentences = []
    #for i in range(len(array)):
     #   zen_sentence = array[i].translate(str.maketrans({chr(0x0021 + i): chr(0xFF01 + i) for i in range(94)})) 
     #   zen_sentences.append(zen_sentence)
    #print(zen_sentences)
    #####これは不要かもしれん
    
    #余計な文字消去
    new_sentences = []       
    for sentence in array:
        new_sentence = [x.replace('\n','') for x in sentence] #改行削除
        new_sentence = [x.replace('\r','') for x in new_sentence] #改行削除
        new_sentence = [x.replace('\r\n','') for x in new_sentence] #こちらも改行削除
        new_sentence = [x.replace('\u3000','') for x in new_sentence] #全角空白削除
        new_sentences.append(new_sentence)    

    #バラバラの文字結合
    final_sentence = []
    for strs in new_sentences:
        sentence = ""
        for s in strs:
            sentence += s
        final_sentence.append(sentence)
    
    #print(final_sentence[0:10])
    return final_sentence

def wakatigaki(array):
    print("wakatigaki")
    mt_default = MeCab.Tagger("-Owakati")
    mt_come = MeCab.Tagger("-Owakati -u /home/koichi/src/comejisho/ComeJisyoUtf8-2r1/ComeJisyoUtf8-2r1.dic")
    mt_neolog = MeCab.Tagger('-Owakati -d /home/koichi/src/neologd/dic/')
    fo_default = open('./data/wakati.txt', 'w')
    fo_come = open('./data/wakati_ComeJisyo.txt', 'w')
    fo_neolog = open('./data/wakati_neolog.txt', 'w')
    
    for i in range(len(array)):
        result = mt_default.parse(array[i])
        wakati_default.append(result)
        fo_default.write(result)
    fo_default.close()

    for i in range(len(array)):
        result = mt_come.parse(array[i])
        wakati_come.append(result)
        fo_come.write(result)
    fo_come.close()

    for i in range(len(array)):
        result = mt_neolog.parse(array[i])
        wakati_neolog.append(result)
        fo_neolog.write(result)
    fo_come.close()

    return wakati_default, wakati_come, wakati_neolog

if __name__ == '__main__': 
    sentence = []
    sentences = []
    wakati_default = []
    wakati_come = []
    wakati_neolog = []
    sentence_array = excel2array2022()
    new_sentence_array = preprocessing(sentence_array)

    wakati_default, wakati_come, wakati_neolog = wakatigaki(new_sentence_array)
    
    with open('./data/pickles/sentences.pkl','wb') as f:
        f.write(pickle.dumps(new_sentence_array))
        #sentences.pkl に new_sentence_array を保存
        #これはさらの文書
    
    with open('./data/pickles/wakati_sentences.pkl','wb') as f:
        f.write(pickle.dumps(wakati_default))
        #ipadicで分かち書きされたテキスト
    
    with open('./data/pickles/wakati_sentences_ComeJisyo.pkl','wb') as f:   
        f.write(pickle.dumps(wakati_come))
        #ComeJisyoで分かち書きされたテキスト

    with open('./data/pickles/wakati_sentences_neolog.pkl','wb') as f:
        f.write(pickle.dumps(wakati_neolog))
        #neologで分かち書きされたテキスト

