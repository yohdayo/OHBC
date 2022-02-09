## extract_sentences.py
- excelからデータを抜き出す
- サラの文書，ipadicでの分かち書き，ComeJisyoでの分かち書き，neologでの分かち書きをpickle
- コーパスの抜き出し方として set_dataのsentenceを利用した by kocihi 2022/2

## word2vec.py
- wakati_xxxxx.txtからモデルを作成

## set_data.py
- ラベル付きのexcelからmy_tf.pyで使うためのラベルを抜き出す
- 緊急度の数字ラベルを漢字に変え，欠損値を省いた配列にする

## make_wordvectors.py
- set_data.pyで作成したtest_sent.pklから各文書に対応する文書ベクトルを作成

## my_tf.py
- 識別する

## cnn_my_tf.py
- cnnにしてみました

# 竹内改変 2022/2
コーパスを2名から読み込んで，緊急度の平均をとる
コーパスからデータをとるときに pandasで nan削除で取り出す
set_data.pyから更新

## データ整理
###  ./data/pickles/train_sent.pkl
     set_data.pyでラベルと同時に保存した文のリスト
     extract_sentences.py これでデータを作る
     word2vec.py これで ./data/models/neolog.model を作成する
     make_wordvectors.py これで ./data/pickles/train_vec.pkl を作成する
     python make_dictionary.py で dictionary.txtを作る
       ここで 
       {'あっ': 0, 'あり': 1, 'から': 2, 'が': 3, 'た': 4,...
       'センター': 595, '仮設': 596, '連島': 597, 'KuraDRO': 598, 'D': 599, '_': 600, 'x': 601, 'Hp': 602, '当': 603, '号車': 604, '）_': 605, '_→': 606, 'びじ': 607, '帰投': 608, '救急搬送': 609, '為': 610, '_⇒': 611, 'スポットクーラー': 612}
       と613単語できているようだ. dictionary.txtも 613種類
    python make_bow_vec.py で ./data/pickles/train_vec_bow.pklを作る

     ./data/pickles/train_label_two.pkl (2クラス分類ラベル 高or中低) 1913データ