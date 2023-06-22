from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from Recipe import *
import os
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import ast
import keras
from keras import layers
import tensorflow as tf

main = tkinter.Tk()
main.title("Inverse Cooking: Recipe Generation from Food Images")
main.geometry("1300x1200")

global filename
global classifier
recipe_list = []
global dataset

def uploadDataset():
    textarea.delete('1.0', END)
    global filename
    global dataset
    recipe_list.clear()
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)    
    textarea.insert(END,'Dataset loaded\n\n')

    dataset = pd.read_csv(filename,nrows=1000)
    for i in range(len(dataset)):
        r_id = dataset.get_value(i, 'recipe_id')
        r_name = dataset.get_value(i, 'recipe_name')
        ingredients = dataset.get_value(i, 'ingredients')
        nutritions = dataset.get_value(i, 'nutritions')
        cooking = ast.literal_eval(dataset.get_value(i, 'cooking_directions')).get('directions')
        r_name = r_name.strip().lower()
        obj = Recipe()
        obj.setRecipeID(r_id)
        obj.setName(r_name)
        obj.setIngredients(ingredients)
        obj.setNutritions(nutritions)
        obj.setCooking(cooking)
        recipe_list.append(obj)
    indian = np.load('index.txt.npy',allow_pickle=True)
    for i in range(len(indian)):
        recipe_list.append(indian[i])
    obj = recipe_list[len(recipe_list)-1]
    print(obj.getName())
    textarea.insert(END,"Recipes data loaded\n")            
        
def buildCNNModel():
    textarea.delete('1.0', END)
    global classifier
    if os.path.exists('model/1model.json'):
        with open('model/1model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/1model_weights.h5")
        classifier._make_predict_function()   
        print(classifier.summary())
        f = open('model/1history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        textarea.insert(END,"CNN training process completed with Accuracy = "+str(accuracy))
    else:
        encoding_dim = 32
        X_train = np.load('model/X.txt.npy')
        Y_train = np.load('model/Y.txt.npy')
        X = X_train.reshape(X_train.shape[0],(64 * 64 * 3))
        print(X.shape)
        input_img = keras.Input(shape=(X.shape[1],))
        encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
        decoded = layers.Dense(Y_train.shape[1], activation='softmax')(encoded)
        autoencoder = keras.Model(input_img, decoded)
        encoder = keras.Model(input_img, encoded)
        encoded_input = keras.Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
        autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
        hist = autoencoder.fit(X, Y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)

def buildvgg():
    textarea.delete('1.0', END)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        model.load_weights("model/model_weights.h5")
        model._make_predict_function()   
        print(model.summary())
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        textarea.insert(END,"\nVGG Trasfer Learning Model Prediction Accuracy = "+str(accuracy)+"\n\n")
    else:
        #creating google net inceptionv3 model by ignoring its top model details and using imagenet weight
        #the object name is base_model
        base_model = tf.keras.applications.vgg16.VGG16(input_shape = (64, 64, 3), include_top = False, weights = 'imagenet')
        #last google net model layer will be ignore to concatenate banana custom model
        base_model.trainable = False
        #getting 3 class from banana dataset as ripe, over ripe and green
       
        #creating own model object
        add_model = Sequential()
        #adding google net base_model object to our custome model
        add_model.add(base_model)
        add_model.add(GlobalAveragePooling2D())
        add_model.add(Dropout(0.5))
        add_model.add(Dense(encoding_dim, activation='softmax'))

        model = add_model
        #compiling model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        #start training model
        hist = model.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 10)
        model.save_weights('model/model_weights.h5')
        model_json = model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        textarea.insert(END,"VGG Trasfer Learning Model Prediction Accuracy = "+str(accuracy)+"\n\n")
        textarea.insert(END,"See Black Console to view VGG layers\n") 

def alexnet():    
    textarea.delete('1.0', END)
    if os.path.exists('model/model1.json'):
        with open('model/model1.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        model.load_weights("model/model_weights1.h5")
        model._make_predict_function()   
        print(model.summary())
        f = open('model/history1.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']  
        accuracy = acc[9] * 100
        textarea.insert(END,"Alexnet Trasfer Learning Model Prediction Accuracy = "+str(accuracy)+"\n\n")
    else:
        model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(224,224,3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(encoding_dim, activation='softmax')])
        hist = model.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 10)
        model.save_weights('model/model_weights.h5')
        model_json = model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        textarea.insert(END,"Alexnet Trasfer Learning Model Prediction Accuracy = "+str(accuracy)+"\n\n")
        textarea.insert(END,"See Black Console to view Alexnet layers\n") 




def predict():
    textarea.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = classifier.predict(img)
    predict = np.argmax(preds)
    if predict > 0:
        predict = predict - 1
    print(predict)
    obj = recipe_list[predict]
    textarea.insert(END,"Recipe Name\n")
    textarea.insert(END,obj.getName()+"\n\n")
    textarea.insert(END,"Ingredients Details\n")
    textarea.insert(END,obj.getIngredients()+"\n\n")
    textarea.insert(END,"Cooking Details\n")
    textarea.insert(END,obj.getCooking()+"\n\n")
    textarea.insert(END,"Nutritions Details\n")
    textarea.insert(END,obj.getNutritions()+"\n\n")

    img = cv2.imread(filename)
    img = cv2.resize(img, (800,400))
    cv2.putText(img, 'Receipe Name : '+obj.getName(), (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow('Receipe Name : '+obj.getName(), img)
    cv2.waitKey(0)


    
def close():
    main.destroy()
    
font = ('times', 14, 'bold')
title = Label(main, text='Inverse Cooking: Recipe Generation from Food Images')
title.config(bg='mint cream', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Recipe Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='mint cream', fg='olive drab')  
pathlabel.config(font=font1)           
pathlabel.place(x=320,y=100)

cnnButton = Button(main, text="Build CNN Model", command=buildCNNModel)
cnnButton.place(x=50,y=150)
cnnButton.config(font=font1) 

cnnButton1 = Button(main, text="Build VGG Model", command=buildvgg)
cnnButton1.place(x=320,y=150)
cnnButton1.config(font=font1) 

cnnButton11 = Button(main, text="Build Alexnet Model", command=alexnet)
cnnButton11.place(x=650,y=150)
cnnButton11.config(font=font1) 


predictButton = Button(main, text="Upload Image & Predict Recipes", command=predict)
predictButton.place(x=50,y=200)
predictButton.config(font=font1) 

closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=320,y=200)
closeButton.config(font=font1)

font1 = ('times', 12, 'bold')
textarea=Text(main,height=20,width=150)
scroll=Scrollbar(textarea)
textarea.configure(yscrollcommand=scroll.set)
textarea.place(x=10,y=250)
textarea.config(font=font1)

main.config(bg='gainsboro')
main.mainloop()
