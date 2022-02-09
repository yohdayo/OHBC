# 山崎さんの識別モデル
# modefied by koichi 2022/2

from wsgiref.util import setup_testing_defaults
import numpy as np
import pickle
import collections
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

def learning(data, label):
    EPOCHS = 40
    BATCH_SIZE = 32
    
    input_dim = data.shape[1]
    model = tf.keras.Sequential([tf.keras.Input((input_dim,)),
                                 tf.keras.layers.Dense(300, activation='relu'),
                                 tf.keras.layers.Dense(100, activation='relu'),
                                 tf.keras.layers.Dense(3, activation='softmax')])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    c_w = {0:1, 1:5, 2:100}

    model.fit(data, label, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[tensorboard_callback],class_weight=c_w)

    return model

def predict(data, label, model, s_test):
    test_loss, test_acc = model.evaluate(data,  label, verbose=2)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    
    f = open("logs.txt", mode='w') 
    f.write("pred, ans, sent\n")

    predictions = model.predict(data)
    y = []
    for loop in predictions:
        y.append(np.argmax(loop))
    #print(predictions)
    #print(y)
    count = 0
    count_tei = 0
    count_tyu = 0
    count_ko = 0
    
    print("pred, ans, sent")
    foutput = []
    tag_labels = []
    for i in range(len(label)):
        tag_label = np.argmax(label[i])
        print( y[i],"  ,",tag_label, " ,",s_test[i])
        outstr = "{0}, {1}, {2} ".format(y[i],tag_label,s_test[i])
        #foutput.append([str(y[i]), ",  ", str(tag_label), ",  ", str(S_test[i])])
        foutput.append(outstr)
        tag_labels.append(tag_label)
        
        if int(y[i]) == int(np.argmax(label[i])):
            count +=1
        if int(y[i]) == 0 and int(np.argmax(label[i])) == 0:
            count_tei += 1
        elif int(y[i]) == 1 and int(np.argmax(label[i])) == 1:
            count_tyu += 1
        elif int(y[i]) == 2 and int(np.argmax(label[i])) == 2:
            count_ko += 1
    
    f.write('\n'.join(foutput))

    print("accuracy = ",count/len(predictions))
    print("tei = ",count_tei)
    print("tyu = ",count_tyu)
    print("ko = ",count_ko)
    print("estimated labels=",collections.Counter(y))
    print("correct labels  =",collections.Counter(tag_labels))

    f.close()

def save_train_test_data(X_train, X_test, Y_train, Y_test, S_train, S_test):
    #全てのデータを保存する
    with open('./data/train_test/x_train.pkl','wb') as f:
        pickle.dump(X_train,f) # [0,0,0,1,0,...] bow
    with open('./data/train_test/x_test.pkl','wb') as f:
        pickle.dump(X_test,f)
    with open('./data/train_test/y_train.pkl','wb') as f:
        pickle.dump(Y_train,f) # [1,0,0],[0,1,0].. one-hot vector
    with open('./data/train_test/y_test.pkl','wb') as f:
        pickle.dump(Y_test,f)
    with open('./data/train_test/s_train.pkl','wb') as f:
        pickle.dump(S_train,f) # ['クラドロ本部資機材．．']
    with open('./data/train_test/s_test.pkl','wb') as f:
        pickle.dump(S_test,f)
    
    print("X_train",X_train[0:4]) #どらちも bowベクトル
    print("X_test",X_test[0:4])
    print("Y_train",Y_train[0:4]) #どちらも正解ベクトル列 one-hot
    print("Y_test",Y_test)
    print("S_train",S_train[0:4]) #どちらも文字列
    print("S_test",S_test[0:4])

def load_fixed_train_test():
    # 読み込み
    with open('./data/train_test/x_train.pkl','rb') as f:
        X_train = pickle.load(f) # [0,0,0,1,0,...] bow
    with open('./data/train_test/x_test.pkl','rb') as f:
        X_test = pickle.load(f)
    with open('./data/train_test/y_train.pkl','rb') as f:
        Y_train = pickle.load(f) # [1,0,0],[0,1,0].. one-hot vector
    with open('./data/train_test/y_test.pkl','rb') as f:
        Y_test = pickle.load(f)
    with open('./data/train_test/s_train.pkl','rb') as f:
        S_train = pickle.load(f) # ['クラドロ本部資機材．．']
    with open('./data/train_test/s_test.pkl','rb') as f:
        S_test = pickle.load(f)
    return X_train,X_test,Y_train,Y_test,S_train,S_test

if __name__ == '__main__': 
    tei = 0
    tyu = 0 
    ko = 0
    with open('./data/pickles/train_label.pkl','rb') as label:
        train_label = pickle.load(label)
        ###個数数えました
        for lab in train_label:
            if np.array_equal(lab, np.array([1, 0, 0])):# 低
                tei += 1
            elif np.array_equal(lab, np.array([0, 1, 0])):# 中
                tyu += 1
            elif np.array_equal(lab, np.array([0, 0, 1])):# 高
                ko += 1
        
        print(tei, tyu, ko, str(tei+tyu+ko))
        # 山崎さん 1535 337 42 1914
        # 竹内のデータ 1332 506 75 1913
    
    with open('./data/pickles/train_sent.pkl','rb') as label:
        train_sent = pickle.load(label)

    with open('./data/pickles/train_vec_bow.pkl','rb') as data:
        train_data = pickle.load(data)   

    # 保存のために1度だけする
    # X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(train_data, train_label, train_sent, train_size=0.8)
    # save_train_test_data(X_train, X_test, Y_train, Y_test, S_train, S_test)
    X_train, X_test, Y_train, Y_test, S_train, S_test = load_fixed_train_test() #保存したものを読み込む

    for yy in Y_test:
        print(yy)
    
    model = learning(X_train, Y_train)
    predict(X_test, Y_test, model, S_test)
    print("test_len = ",len(X_test))

