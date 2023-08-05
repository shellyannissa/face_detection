## Overview
This is a machine learning model built over the VGG16 architecture to pinpoint the 
location of a human face in an image. Data taken was labelled using labelme, and was augmented 
with the aid of albumentations to expand the dataset to contain over 6000 images. Use case would 
include webcam proctoring for tests, photo editing (camera focusing) ,automotive safely monitoring etc..

![faces_with rectangle](https://github.com/shellyannissa/face_detection/assets/118563935/123e2c84-ce04-44fc-9290-ea6e227f94b5)


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Techstack](#techstack)
- [Disclaimer](#usage)


## Installation<a name="installation"/>
The sequence of events are clearly catalogued in the colab_notebook.ipynb file
For running python scripts :-

Installing dependencies and cloning the repository

```
pip install tensorflow numpy 
pip install opencv-python albumentations matplotlib
git clone https://github.com/shellyannissa/face_detection.git
cd face_detection

```
Create Dataset by capturing images of yourself
```
python3 capture_image.py
```

Annotating the images

in macos:-
```
brew install pyqt5
brew install wkentaro/labelme/labelme
labelme
```
linux distributiosns
```
sudo apt-get install python3-pyqt5
sudo apt-get install python3-pyqt5.qtwebkit python3-pyqt5.qtsvg python3-pyqt5.qtserialport python3-pyqt5.qtopengl
sudo apt-get install python3-pyqt5.qtsvg
pip install labelme
labelme
```
open the folder face_detection/face in the labelme gui draw rectangular boxes enclosing 
faces within each image. Store the json label files within a new folder face_detection/labels




