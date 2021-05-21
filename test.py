import sys
import os
import numpy as np
import tensorflow as tf
import cv2
from scipy.ndimage import interpolation as inter
from pathlib import Path

if __name__ == "__main__":
	# Just disables the warning, doesn't enable AVX/FMA
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	model = tf.keras.models.load_model("model.h5")

	input_dim = 400

	fol = os.path.join('word_rotation_dataset', "test")
	Xtest = list()
	file_name = list()
	count = 1
	for i in sorted(os.listdir(fol)):
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
		print(i)
		file_name.append(i)
		Xtest.append(overlay)
		count += 1
		if count > 100 :
			break


	Xtest = np.array(Xtest)
	ypred = model.predict(Xtest)

	for i in range(len(ypred)) :
		best_angle = np.argmax(ypred[i])
		print(best_angle)
		img = inter.rotate(cv2.imread(os.path.join(fol, file_name[i])), -best_angle , reshape=True, order=1)
		cv2.imwrite(os.path.join('output', file_name[i]),img)