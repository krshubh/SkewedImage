## Introduction
We are given skewed images with some degree. We are predicting the angle of skewed image using CNN based model then rotating 
it in opposite direction to get unskewed image.

## How to run
	-> python main.py 'image_fol/image_name.jpg' 
	-> python main.py 'image_name.jpg' 

## Training
        -> run python train.py
	
## Test
        -> run python test.py

## Important points
	-> change the k value to increase training dataset.Currently, we have used 
	   only two random rotated data per image.
	-> We are doing multi-class classification and there are 360 classes for each degree.
	-> First, predicting rotated angle using this model then rotating the given image.
	
	
![Skewed Image](https://github.com/krshubh/SkewedImage/blob/master/word_rotation_dataset/test/000005100.png?raw=true)

![Output Image](https://github.com/krshubh/SkewedImage/blob/master/output/000005100.png?raw=true)


