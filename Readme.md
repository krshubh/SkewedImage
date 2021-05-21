1) To run :
	-> python main.py 'image_fol/image_name.jpg' 
	-> python main.py 'image_name.jpg' 


2) Approach :
	a) Training 
	For training 
		-> run python train.py
		-> change the k value to increase training dataset.
		   currently, I have used only one random rotated data per image.
		-> I have created 360 output class for particular degree. 

	b) Prediction 
	There is model.h5 file which I have generated after training. I am currect predicted rotated angle using this model.
	Then rotating the given image.


