from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import glob
from PIL import Image
import pickle
import os


class Classify_Gender(object):

    def __init__(self, model_name, input_shape):
        nClasses = {
            0 : 55,
            1 : 45
        }
        if model_name != None:
            self.model = load_model(location, model_name)
        else:
            self.model = Sequential()
            self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape, return_sequences=False))
            self.model.add(Conv2D(32, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))

            self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
            self.model.add(Conv2D(64, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))

            self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
            self.model.add(Conv2D(64, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))

            self.model.add(Flatten())
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(nClasses, activation='softmax'))

        return self.model


    def train(self,train_data, train_labels, test_data, test_labels, batch_size = 250, epochs = 100):
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
                with open(save_location + '/' + model_name) as f:
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


def load_training_data(location):
    train_data = []
    if location != None:
        for image in glob.glob(location + "*.jpg"):
            print(image)
            temp_image = Image.open(image)
            train_data = np.asarray(train_data)
    else:
        print("Invalid File or Location")
    print(len(train_data))
    return train_data


def train_labels_(filename):
    train_label = np.array()
    return


train_data = load_training_data("C:\\Users\Goutham\\Documents\\Claire\\Suhas\\Face_classification\\whole_images\\")
# class_gender = Classify_Gender(None, train_data[0].shape)
# class_gender.train_data(train_data, train_labels_(), )