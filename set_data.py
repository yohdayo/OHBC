#excelデータを整形して配列に格納するプログラム
#ラベルも保存
import glob
import pandas as pd
from itertools import chain
import pickle
import numpy as np
import re

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
            print("sentence is deleted due to its wrong tag format =",kanji)
            del sentences[i]
#重要度の低中高を数字に．それ以外のデータは削除
#30,64,699,701,708,713,716,719,721が欠損地になる
        
    print("number of lables (final)",len(labels_num))
    print("number of sentences (final)",len(sentences))
            
    labels_num = np.array(labels_num)
    return sentences, labels_num 

def labelWOnumber3(sentences, array):    
    # 3クラス分類
    labels_num = []
    for i,kanji in enumerate(array):
        if kanji == "低":
            labels_num.append([1,0,0])
        elif kanji == "中":
            labels_num.append([0,1,0])
        elif kanji == "高":
            labels_num.append([0,0,1])
        else:
            print("sentence is deleted due to its wrong tag format =",kanji)
            del sentences[i]
        
    print("number of lables (final)",len(labels_num)) # 1913
    print("number of sentences (final)",len(sentences))
            
    labels_num = np.array(labels_num)
    return sentences, labels_num 

def make_train_data2022():
    # 竹内
    # 2人のアノテータの緊急度を読み込んで高い方(合算して平均四捨五入)をとる
    # 緊急度をマージする (平均，四捨五入)
    eme_lev_st2id = {'高':3,'中':2,'低':1}
    eme_lev_id2st = {}
    for k,v in eme_lev_st2id.items():
        eme_lev_id2st[v]=k
    print(eme_lev_id2st)
    print("excel => array")
    #ディレクトリ内の *.xlsx 
    bpath = '2021Feb4'
    mypath = ['県保健医療調整本部','県南西部災害保健医療調整本部','倉敷市保健所内']
    
    file_list = []
    for path in mypath:
        pp = "./data/chronologies/{0}/{1}/*.xlsx".format(bpath,path)
        #print("pp====",pp)
        local_file_list = glob.glob(pp)
        file_list.extend(local_file_list)
    print ("file_list==",file_list)
    
    ###
    # 県保健医療調整本部 (活動記録（石澤）.xlsx,活動記録（齋藤）.xlsx)        [792x5]
    # 県南西部災害保健医療調整本部 
    #   (県南西部災害保健医療調整本部（石澤）.xlsx,県南西部災害保健医療調整本部（齋藤）.xlsx) [182x5]
    # 倉敷市保健所内 
    #   (川崎医科大学附属病院（石澤）.xlsx, 川崎医科大学附属病院（齋藤）.xlsx)[164x5]
    #   (倉敷市保健所内（石澤）.xlsx, 倉敷市保健所内（齋藤）.xlsx)          [785x5]

    # アノテータ毎に緊急度をとりだして平均して最終結果を作る
    
    def get_conts_and_levels(file_list,annotator): 
        # 各ファイルから内容と緊急度取り出す
        # 作業者2名のそれぞれの緊急度を取り出す
        sent_levels_tmp=[]
        for filename in file_list:
          if annotator in filename:
            print("annotator={0}, corpusname=={1}".format(annotator,filename))
            df = pd.read_excel(filename,engine='openpyxl')
            sentence_emergency_level = df[["内容","緊急度"]]
            #excelシートの内容部分を抽出
            sent_levels_tmp.append(sentence_emergency_level)

        sent_level = []
        for df_sent_level in sent_levels_tmp: 
            #各コーパス毎. 中身はdataframe型
            sent_level.append(df_sent_level[["内容","緊急度"]])
        return sent_level #List[DataFrame]
    
    #各アノテータごとの 緊急度の合算を行う
    i_sent_level = get_conts_and_levels(file_list,"石澤") # typeはlist[df]
    s_sent_level = get_conts_and_levels(file_list,"齋藤") # typeはlist[df]
    # 緊急度マージのためのファイル数のcheck
    if len(i_sent_level) != len(s_sent_level):
        print("error, Ishizawa corpus does not match to Saitoh, {0} vs {1}" \
              .format(len(i_sent_level),len(s_sent_level)))
        exit(0)
    
    def get_average_level(annot_a,annot_b):
        # 高，中，低の2名のアノテータの出力の平均をとる
        def get_score_int(tag): #アノテータのタグからスコアを取り出す
            if len(tag) != 1:
                # 低¥n中など2つ入っている
                # 高い方をとる
                out_score = 0
                if '高' in tag: out_score = 3
                if '中' in tag: out_score =  2
                if '低' in tag: out_score =  1
                print("complex tag=",tag, " score=", out_score)
                return out_score
            else:
                return eme_lev_st2id[tag]
        a_score = get_score_int(annot_a)
        b_score = get_score_int(annot_b)
        total_score = int((a_score + b_score)/2.0+0.5)
        # print(a_score," ", b_score," ", total_score)
        average_tag = eme_lev_id2st[total_score]
        # print(average_tag, " "," a=",annot_a, " b=",annot_b)
        return average_tag

    def value_check(annot_a,annot_b): 
        # タグが入ってるかcheckする "中低" のような2つ入ってるのは許す
        # まずtypeのチェック
        if type(annot_a) != str or type(annot_b) != str:
            print("not string", annot_a, annot_b)
            exit(0)
        aa = annot_a.strip()
        bb = annot_b.strip()
        # print("aa=",aa, " bb=",bb)
        pattern = '[高|中|低]'
        if not re.search(pattern,annot_a) or not re.search(pattern,annot_b):
            return False
        return True

    contents_list = []
    levels_list = []
    for idata, sdata in zip (i_sent_level,s_sent_level):
        # 2人の作業ファイルデータを1つにする
        sent_level = pd.concat([idata,sdata],axis=1)
        # nanを排除 (行を削除)
        sent_level = sent_level.dropna(how='any')
        #print(sent_level)
        slist = sent_level.values.tolist()
        for tmpl in slist:
            #print(tmpl)
            if tmpl[0] != tmpl[2]: #2人の作業者でクロノロの文字が一致しているか? 
                print("error! different between two annotators")
                print("is=",tmpl[0])
                print("sa=",tmpl[2])
                exit(0)
        # 2人の作業が一致しているので，高，中，底をマージして1つにする
        for sl in slist:
            contents = sl[0]
            annot_a = sl[1] # 石澤さん
            annot_b = sl[3] # 齋藤さん
            #print("st--",contents,",",annot_a,",",annot_b,"--end")
            if not value_check(annot_a,annot_b):
                print("skipped: warning annot value is not correct in",contents,",", annot_a, ",", annot_b)
                continue
            annot_av = get_average_level(annot_a,annot_b)#平均取る
            contents_list.append(contents)
            levels_list.append(annot_av)

    #print(levels_list,len(levels_list))
    return contents_list, levels_list


if __name__ == '__main__': 
    labels = []
    sentences = []
    labels_num = []
    sentences, labels = make_train_data2022()
    train_sent, train_label = labelWOnumber(sentences, labels) # 2class 分類
    _, train_label3 = labelWOnumber3(sentences, labels) # 3 class 分類


    with open('./data/pickles/train_sent.pkl','wb') as f:
        f.write(pickle.dumps(train_sent))
        #sentences.pkl に new_sentence_array を保存

    with open('./data/pickles/train_label_two.pkl','wb') as f:
        f.write(pickle.dumps(train_label))
        #sentences.pkl に new_sentence_array を保存

    with open('./data/pickles/train_label.pkl','wb') as f:
        f.write(pickle.dumps(train_label3))