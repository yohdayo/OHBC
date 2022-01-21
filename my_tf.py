

import numpy as np
import pickle
import collections
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

def learnning(data, label):
    EPOCHS = 40
    BATCH_SIZE = 32

    model = tf.keras.Sequential([tf.keras.Input((606,)),
                                 tf.keras.layers.Dense(128, activation='relu'),
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

def predict(data, label, model):
    test_loss, test_acc = model.evaluate(data,  label, verbose=2)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    
    with open("logs.txt", mode='w') as f:
        f.write("pred, ans, sent")

    predictions = model.predict(data)
    y = []
    for loop in predictions:
        y.append(np.argmax(loop))
    print(predictions)
    print(y)
    count = 0
    count_tei = 0
    count_tyu = 0
    count_ko = 0
    #print("pred, ans, sent")
    
    for i in range(len(label)):
        print( y[i],"  ,",np.argmax(label[i]), " ,",S_test[i])
        #f.writelines(y[i],"  ,",np.argmax(label[i]), " ,",S_test[i])
        if int(y[i]) == int(np.argmax(label[i])):
            count +=1
        if int(y[i]) == 0 and int(np.argmax(label[i])) == 0:
            count_tei += 1
        elif int(y[i]) == 1 and int(np.argmax(label[i])) == 1:
            count_tyu += 1
        elif int(y[i]) == 2 and int(np.argmax(label[i])) == 2:
            count_ko += 1
    
    

    print("accuracy = ",count/len(predictions))
    print("tei = ",count_tei)
    print("tyu = ",count_tyu)
    print("ko = ",count_ko)
    print(collections.Counter(y))
    
    f.close()


if __name__ == '__main__': 
    tei = 0
    tyu = 0 
    ko = 0
    with open('./data/pickles/train_label.pkl','rb') as label:
        train_label = pickle.load(label)
        """個数数えました
        for lab in train_label:
            if np.array_equal(lab, np.array([1, 0, 0])):
                tei += 1
            elif np.array_equal(lab, np.array([0, 1, 0])):
                tyu += 1
            elif np.array_equal(lab, np.array([0, 0, 1])):
                ko += 1
        
        print(tei, tyu, ko, str(tei+tyu+ko))
        1535 337 42 1914
        """
    with open('./data/pickles/train_sent.pkl','rb') as label:
        train_sent = pickle.load(label)

    with open('./data/pickles/train_vec_bow.pkl','rb') as data:
        train_data = pickle.load(data)   

    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(train_data, train_label, train_sent, train_size=0.8)
    #全てのデータから
    
    model = learnning(X_train, Y_train)
    predict(X_test, Y_test, model)
    print("test_len = ",len(X_test))

