# comvis
A collection of Computer Vision exercises using OpenCV.  

## Setting Up OpenCV  

### 1. Install Python  
Download & install Python from [python.org](https://www.python.org/downloads/).  
Make sure to check **"Add Python to PATH"** during installation.  

### 2. Install OpenCV  
Open **Command Prompt (cmd)** and run:  
```sh
pip install opencv-python
pip install opencv-python-headless  # If you don't need GUI features
```

### 3.Verify Installation
Run the following in python:
```sh
import cv2
print(cv2.__version__)
```

### 4.Running the code
To run a example:
```sh
python ex1/example.py #this is only for example so change the file name to run that file
```

### 5.Test OpenCV with an Image
```sh
import cv2

# Load an image
image = cv2.imread("test.jpg")  # Replace with your image file

# Show the image
cv2.imshow("Test Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# Contributing
Feel free to fork and contribute by adding more exercises or improving the code.

# License
This project/collection of exercise is open-source and free to use.
```sh

Just save this as **`README.md`** in your GitHub repository. Let me know if you need modifications! 🚀
```
