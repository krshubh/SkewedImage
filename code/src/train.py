import sys
import os
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Flatten
from keras.layers import GaussianNoise
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from tqdm import tqdm
import random
import cv2
from keras.utils.np_utils import to_categorical
from scipy.ndimage import rotate

def createModel(input_shape):
	model = keras.Sequential([
	keras.layers.Conv2D(32, kernel_size = 3, padding='valid', input_shape=input_shape, activation='relu'),
	keras.layers.Dropout(0.3),
	keras.layers.MaxPooling2D(pool_size = 2),
	keras.layers.Conv2D(32, kernel_size = 3, padding='valid', activation='relu'),
	keras.layers.Dropout(0.3),
	keras.layers.MaxPooling2D(pool_size=2),
	keras.layers.Conv2D(64, kernel_size = 3, padding='valid', activation='relu'),
	keras.layers.Dropout(0.3),
	keras.layers.MaxPooling2D(pool_size=2),
	keras.layers.Conv2D(64, kernel_size = 3, padding='valid', activation='relu'),
	keras.layers.Dropout(0.3),
	keras.layers.MaxPooling2D(pool_size=2),
	keras.layers.Conv2D(64, kernel_size = 3, padding='valid', activation='relu'),
	keras.layers.Dropout(0.3),
	keras.layers.MaxPooling2D(pool_size=2),
	keras.layers.Conv2D(128, kernel_size = 3, padding='valid', activation='relu'),
	keras.layers.BatchNormalization(),
	keras.layers.Dropout(0.3),
	keras.layers.MaxPooling2D(pool_size=2),
	keras.layers.Flatten()
	# keras.layers.BatchNormalization(momentum=0.5, epsilon=0.001, center=True, scale=True)
	])
	model.add(Dense(360, activation='softmax'))
	model.compile(optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy'])
	return model

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save("models/model_{}.h5".format(epoch))
        
def plot_accuracy_loss(history):
	#	"Accuracy"
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig("plots/accuracy.png")
	plt.show()
	# "Loss"
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig("plots/loss.png")
	plt.show()


if __name__ == "__main__":
	# Just disables the warning, doesn't enable AVX/FMA
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	tf.config.set_visible_devices([], 'GPU')

	input_dim = 400
	# change the value of k to get more training data.
	# number of angle per image
	k = 2
	fol = os.path.join('data', "train")
	X = list()
	y = list()
	for i in tqdm(sorted(os.listdir(fol))):
		if i.endswith(".png"):
			img = cv2.imread(os.path.join(fol, i))
			w,h,d = img.shape
			overlay = np.zeros((input_dim, input_dim, 3))
			if w > input_dim and h > input_dim:
				overlay[:,:,:] = img[(w-input_dim)//2:(w+input_dim)//2,(h-input_dim)//2:(h+input_dim)//2,:]
			elif w > input_dim :
				overlay[:,(input_dim - h)//2:(input_dim + h)//2,:] = img[(w-input_dim)//2:(w+input_dim)//2,:,:]
			elif h > input_dim :
				overlay[(input_dim - w)//2:(input_dim + w)//2,:,:] = img[:,(h-input_dim)//2:(h+input_dim)//2,:]
			else :
				overlay[(input_dim - w)//2:(input_dim + w)//2,(input_dim - h)//2:(input_dim + h)//2,:] = img
			
			for angle in [random.randint(1,359) for i in range(k)] :
				new_img = rotate(overlay, angle , reshape=False, order=1)
				X.append(new_img)
				y.append(angle)

	X = np.array(X)
	y = np.array(y)
	y = to_categorical(y, 360)

	model = createModel((input_dim,input_dim,3))
	print(model.summary())
	save_model = CustomSaver()
	es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
	history = model.fit(X, y,
                    	callbacks=[save_model, es_callback],
                    	validation_split=0.1,
                    	epochs=50,
                    	batch_size=16)
	plot_accuracy_loss(history)


