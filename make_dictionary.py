# -*- coding: utf-8 -*-
import MeCab
from gensim import corpora, matutils
import pickle
mecab = MeCab.Tagger('mecabrc')


def tokenize(text):
    '''
    とりあえず形態素解析して名詞だけ取り出す感じにしてる
    '''
    node = mecab.parseToNode(text)
    while node:
        if node.feature.split(',')[0] == '名詞':
            yield node.surface.lower()
        node = node.next


def get_words(contents):
    '''
    形態素解析してリストにして返す
    '''
    ret = []
    for k, content in contents.items():
        ret.append(get_words_main(content))
    return ret


def get_words_main(content):
    '''
    一つの記事を形態素解析して返す
    '''
    return [token for token in tokenize(content)]



if __name__ == '__main__':
    with open('./data/pickles/wakati_sentences_neolog.pkl','rb') as f:
        sentences = pickle.load(f)
    test = sentences
    words =[]
    for s in test:
        words.append(s.split())
    print(words)

    all_words = []
    for sen in words:#wordsのなかの各文に対して
        for term in sen:
            if term in all_words:
                pass
            else:
                 all_words.append(term)
    print(all_words)
    dictionary = corpora.Dictionary(words)
    dictionary.filter_extremes(no_below=10, no_above=0.3)
    # no_berow: 使われてる文章がno_berow個以下の単語無視
    # no_above: 使われてる文章の割合がno_above以上の場合無視

    print(dictionary.token2id)
    #すべての文で使われてる単語の重複無しセットになりました。単語になんかIDふってくれた。

    dictionary.save_as_text('dictionary.txt')
    #辞書のセーブ
    #dictionary = corpora.Dictionary.load_from_text('livedoordic.txt')
    #辞書の読み出し
    #vec = dictionary.doc2bow(words)
    #print(vec)
    """
    tmp = dictionary.doc2bow(all_words)
    dense = list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
    print(dense)
    """
    
