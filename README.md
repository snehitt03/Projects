This is Machine Learning project which is aimed at recognizing 11 distinct persolnalities.
The dataset of this project i.e. images for every distinct has been downloaded from web using chrome extensions for downloading large quantity of images.
The downloaded images are then cropped and resized using Haar Cascades algorithm to crop if two eyes and face region is visible in that photo/image.
I applied Daubechies wavelet transform to extract the features of images,and then vertically stacked this transformed and orignal image by converting the resultant image into a 1D Matrix and then fed it to the SVM(Support Vector Machine) model to categorize and train the machine learning model.
Thus the trained model was able to detect each image with an accuracy of 87%.
