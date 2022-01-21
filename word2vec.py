from gensim.models import word2vec


wakati_sentences = word2vec.LineSentence("./data/wakati_neolog.txt")
model = word2vec.Word2Vec(wakati_sentences,
                          sg=1,#skip-gramを使用
                          size=100,#単語をベクトル表現する際の次元数
                          min_count=1,#出現回数がこの数以下の単語を学習に使うデータから省く
                          window=10,#前後いくつまでの単語を周辺語として扱うか
                          hs=1#学習に階層化ソフトマックス関数を使用
                          )
model.save("./data/models/neolog.model")

#print(len(model.wv.vocab))
for i in model.most_similar("診断"):
    print(i)

