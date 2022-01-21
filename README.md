## extract_sentences.py
- excelからデータを抜き出す
- サラの文書，ipadicでの分かち書き，ComeJisyoでの分かち書き，neologでの分かち書きをpickle

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
