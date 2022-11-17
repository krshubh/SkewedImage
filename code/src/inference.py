import sys
import os
import numpy as np
import tensorflow as tf
import cv2
from scipy.ndimage import rotate
from pathlib import Path

if __name__ == "__main__":
	# Just disables the warning, doesn't enable AVX/FMA
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	input_file = sys.argv[1]
	file_name = Path(input_file).name

	model = tf.keras.models.load_model("model.h5")

	input_dim = 400

	test = list()

	img = cv2.imread(input_file)


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

	test.append(overlay)
	test = np.array(test)

	vec = model.predict(test)
	best_angle = np.argmax(model.predict(test))

	print(best_angle)

	img = rotate(img, -best_angle , reshape=True, order=1)
	cv2.imwrite(os.path.join(os.path.join('data','output'), file_name),img)
