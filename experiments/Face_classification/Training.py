from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
# from keras.preprocessing.image import ImageDataGenerator


import numpy as np
import glob
from PIL import Image
import pickle
import os


class Classify_Gender(object):

    def __init__(self, model_name, input_shape):
        self.nClasses = 2
        if model_name != None:
            self.model = load_model(location, model_name)
        else:
            self.model = self.createModel(input_shape)
            
    def createModel(self, input_shape):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nClasses, activation='softmax'))
        return model

    def train(self,train_data, train_labels, test_data, test_labels, batch_size = 10, epochs = 10):
        try:    
            self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

            history = self.model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, verbose=1,
                               validation_data=(test_data, test_labels))
            print(history)
            return True
        except Exception as e:
            print(e)
            return False

    def predict(self, test_data, test_labels, retrain = True):
        temp_data = []
        if retrain and type(test_data) != "numpy.array.array":
            for data in test_data:
                temp_data.append(np.asarray(data))
                print("Testing for data\n", data)
                return self.model.evaluate(data, test_labels)
        else:
            print("Not retraining data to the model")
            return self.model.evaluate(test_data, test_labels)


    def save_model(self, model_name, save_location):
        if os.path.exists(save_location):
            try:
                with open(save_location + model_name) as f:
                    pickle.dump(self.model, f)
                    print("Model Saved as", model_name, "in", "save_location")
                    return True
            except Exception as e:
                print(e)
                return False
        else:
            print("Failed to save Model")
            return False


    def load_model(self, location, model_name):
        with open(location + "/" + model_name) as f:
            model = pickle.load(f)
            print("Model Loaded Successfully")
        return model

    def load_training_data(location):
        train_data = []
        if location != None:
            for image in glob.glob(location):
                temp_image = Image.load(image)
                train_data = np.asarray(train_data)
        print(len(train_data))


    def plot_out(self):
        plt.figure(figsize=[8,6])
        plt.plot(history.history['loss'],'r',linewidth=3.0)
        plt.plot(history.history['val_loss'],'b',linewidth=3.0)
        plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.title('Loss Curves',fontsize=16)

        # Accuracy Curves
        plt.figure(figsize=[8,6])
        plt.plot(history.history['acc'],'r',linewidth=3.0)
        plt.plot(history.history['val_acc'],'b',linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Accuracy',fontsize=16)
        plt.title('Accuracy Curves',fontsize=16)


train_X = []
train_Y = []


def load_data(location, labelfile1, labelfile2):
    train_data = []
    train_labels = []
    males = open(labelfile1).readlines()
    females = open(labelfile2).readlines()
#     print(type(males))
    if location != None:
        for image in males:
            image = image.replace("\n", "")
            print("Loading", image)
            imagename = location + image
            temp_image = np.array(Image.open(imagename))
            
            train_data.append(temp_image)
            
            train_labels.append([1, 0])
        for image in females:
            image = image.replace("\n", "")
            print("Loading", image)
            imagename = location + image
            temp_image = np.array(Image.open(imagename))
            train_data.append(temp_image)
            train_labels.append([0, 1])
        print('Shape is ', temp_image.shape)
    else:
        print("Invalid File or Location")
#     print(len(train_data))
    return np.array(train_data), np.array(train_labels)


train_X, train_Y = load_data("C:\\Users\Goutham\\Documents\\Claire\\Suhas\\Face_classification\\original_images\\",
                                              labelfile1 = "C:\\Users\\Goutham\\Documents\\Claire\\Suhas\\Face_classification\\5-folds\\male_namesOriginal.txt",
                                              labelfile2 = "C:\\Users\\Goutham\\Documents\\Claire\\Suhas\\Face_classification\\5-folds\\female_namesOriginal.txt")

print(len(train_X), len(train_Y))

c = Classify_Gender(None, train_X[0].shape)

c.train(train_X, train_Y, train_X, train_Y)

test_images = []
for image in glob.glob("C:\\Users\Goutham\\Documents\\Claire\\Suhas\\Face_classification\\whole_images\\*.jpg"):
	test_images.append(np.array(Image.open(image)))

c.predict(np.array(test_images), np.array([1, 0], [0, 1]))

c.save_model(model_name = "testmodel.tm", save_location = "C:\\Users\\Goutham\\Documents\\Claire\\Suhas\\Face_classification\\model_data\\")

c.plot_out()