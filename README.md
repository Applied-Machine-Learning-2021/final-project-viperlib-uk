# 1. VIPER Summer Project 2021

[V.I.P.E.R](https://github.com/jasdehart/viper_summer_2021) stands for Visual Inspection of Personal Exposed Records

Online privacy has become immensely important with the growth of technology and the expansion of communication. Social Media Networks have risen to the forefront of current communication trends. With the current trends in social media, the question now becomes how can we actively protect ourselves on these platforms? Users of social media networks share billions of images a day. Whether intentional or unintentional, users tend to share private information within these images. In this library we implement a proof-of-concept with the COCO detection library and multiple blocking techniques that could be implemented into any social media.

For this summer project, the VIPERLIB AMLI team has made a script that users could use to to block specified classes from the COCO dataset. This is used as a building block for bigger projects, for example a social media network that blocks images for followers based on the privacy score of that image.

# 2. Team
- [Faith Makumbi](https://github.com/faith098)
- [To'Nia Ray](https://github.com/tabe228)
- [Blair Hall](https://github.com/B-Hall1)
- [Oscar Feliz](https://github.com/Cobra-irl)
- [Jasmine DeHart - Guide](https://github.com/jasdehart)    
- [Christan Grant - Advisor](https://github.com/cegme)

# 3. Table of Contents
- [1. VIPER Summer Project 2021](#1-viper-summer-project-2021)
- [2. Team](#2-team)
- [3. Table of Contents](#3-table-of-contents)
- [4. Requirements](#4-requirements)
- [5. How to use](#5-how-to-use)
  - [5.1. Server instructions](#51-server-instructions)
  - [5.2. Local instructions](#52-local-instructions)
- [6. In-depth Explanation](#6-in-depth-explanation)
  - [6.1. Helper Functions](#61-helper-functions)
  - [6.2. VIPER-specific helper functions](#62-viper-specific-helper-functions)
  - [6.3. Image class](#63-image-class)
  - [6.4. Image Modifiers](#64-image-modifiers)
  - [6.5. Image Handling](#65-image-handling)


# 4. Requirements 

to install the required libraries, run:
```
pip install -r /path/to/requirements.txt
```


# 5. How to use
ViperLib has been broken into 2 separate folders to optimize for multiple purposes: Speed and Information. The folders are "Server" and "Local" (Testing). Both methods are ran through command line, so make sure to change directory (CD) depending on  use. 
## 5.1. Server instructions
To run the Flask server, run:
```
python3 -m flask run  
```

to make a docker image and run it as a server, run:
```
docker build --tag viper .
```
```
docker run -d -p 5000:5000 viper
```


The server will then accept POST Request with keys:
| KEY | VALUE |
| ------------- | ------------- |
| image  | an image url |
| modifier  |  [blur / br, block / bk, noise / ne, saliency / sy] |
| relationship  | integer [0 (full blocking), 1 (semi blocking), 2 (no blocking)] |

After recieved a POST request, the server will return an image. This has been tested with [postman](https://www.postman.com/downloads/).

## 5.2. Local instructions
```
usage: VIPER.py [-h] image_file mod

positional arguments:
  image_file  the image file name. Image should be in the folder.
  mod         Possible argument: [blur / br, block / bk, noise / ne, saliency / sy]
```

run:
```
python3 VIPER.py IMAGE_FILE MOD
```



# 6. In-depth Explanation

## 6.1. Helper Functions
These functions are general functions that don't do anything significantly on their own but help other functions along the way. 

Used for downloading files from online
```python
Download(base_url, file_name)
```
Uses Download() to download the model files.
```python
download_model()
```
Runs the model for an image
```python
run_model(input_image)
```

## 6.2. VIPER-specific helper functions
VIPER-specific helper functions are helper functions that don't do anything significant on their own as well. Unlike the functions in 6.1, these functions were made specifically to help with the class Image and modifier functions.


Coco returns detecion boxes in decimals. This function returns them as their pixel values.
```python
VIPER_COCO_box_to_pixels(box_bounds, cv2_image)
```
Determines if a class number is private or not
```python
VIPER_COCO_priv_or_not(class_num)
```

Second iteration of private or not, which uses relationship to determine what gets blocked or not. **NOTE: only exists on server**
```python
VIPER_COCO_priv_or_not_v2(class_num, relationship)
```

Determines if a list of class numbers are private or not
```python
VIPER_COCO_priv_and_not(detected_classes_list)
```

Does the blurring for individual objects
```python
VIPER_object_blur(cv2_image, x1, x2, y1, y2, blurAmount)
```

Does the blocking for individual objects
```python
VIPER_object_block(cv2_image, x1, x2, y1, y2, color)
```

Does the calculation for the privacy score.
```python
privacy_score(image, detections, num_of_detections, detection_classes)
```

## 6.3. Image class

This is the class that is used to hold all the information. There is the image (that will be modified), a backup of the image, detections from the model, a list of private class numbers, number of detections, the detected classes, and finally the privacy score. There is also a rerun model function that runs after the image class is modified, giving us the opportunity to see the change in privacy score.

```python
class Image():

  def __init__(self, image_path):
    self.image
    self.original

    self.detections

    self.num_of_detections
    self.detection_classes
    self.private_classes
    self.privacy_score

  def rerun_model(self):
    self.privacy_score_ad
```

**NOTE: privacy score doesn't exist on the server script for performance purposes.**

## 6.4. Image Modifiers
Image modifiers are the obscuring methods that VIPER has focused on. Each function takes the image object and modifies it's self.image variable.


```python
VIPER_blur(ImageType)
```

```python
VIPER_block(ImageType)
```
**NOTE: VIPER_blur and VIPER_block take relationship as a second parameter for the server.**

```python
VIPER_advarserial_modifier(ImageType)
```

```python
VIPER_saliency_modifier(ImageType)
```

## 6.5. Image Handling
The image handlers are functions that simply the code and streamlines everything in small commands. These take the multiple functions that run the program and simplifies them for ease of use.


Runs the modifications and replaces the image in the ImageType with its result. **NOTE: image_modifier and command_line take relationship as a 3rd parameter in server.**
```python
image_modifier(ImageType, modifier)
```

Saves Image into a jpg file.
```python
image_saver(ImageType)
```

Fuction for the command line usage.
```python
command_line(image_path, modifier)
```
