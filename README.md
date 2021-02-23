# Mask detection using Tensorflow and CV2

The purpose of this project was to detect if a person was wearing a mask or not using our laptop camera.

We used Tensorflow for the model, cv2 for the camera and the image processing and haarcascade for the faces detection.

The model was trained with 96x96 sized images in RGB.

An alarm song is played if the person is not wearing a mask and a happy emoji is displayed if the person is wearing one.

## Neural network

The Convolutional Neural Network was made with Tensorflow and trained with 658 images of each classes.
We obtained 96% of accuracy on the image test set. 

## Mask detection

Face detection is done with haarcascade frontalface pre-trained model. We then use our classifier to predict the class on the detected zone.

