import numpy as np
import pickle
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def learnning(data, label):
    EPOCHS = 40
    BATCH_SIZE = 32
    """
    model = tf.keras.Sequential([tf.keras.Input((606,)),
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dense(3, activation='softmax')])
    """
    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(606, (2, 2), activation='relu',padding = 'same', input_shape=(3,1)),
                                        tf.keras.layers.MaxPooling2D((2, 2)),
                                        tf.keras.layers.Conv2D(606, (2, 2), activation='relu',padding = 'same'),
                                        tf.keras.layers.MaxPooling2D((2, 2)),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(3, activation='softmax')])

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    model.fit(data, label, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[tensorboard_callback])

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
    print("len = ",len(label))
    
    f.close()


if __name__ == '__main__': 
    with open('./data/pickles/train_label.pkl','rb') as label:
        train_label = pickle.load(label)

    with open('./data/pickles/train_sent.pkl','rb') as label:
        train_sent = pickle.load(label)
    
    with open('./data/pickles/train_vec_bow.pkl','rb') as data:
        train_data = pickle.load(data)   

    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(train_data, train_label, train_sent, train_size=0.9)
    #全てのデータから
    
    model = learnning(X_train, Y_train)
    predict(X_test, Y_test, model)
    print("test_len = ",len(X_test))

