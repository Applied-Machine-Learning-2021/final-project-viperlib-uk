#######################################################################
# Helper
#######################################################################
import logging

logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s"
    )

logging.info("Loading libraries, takes aproximately 4 seconds.")
# Libraries that will be needed for everything to run
import cv2 as cv # used for handling the image aspect
# import tarfile # used to untar the model downloaded
# import shutil
# import urllib.request # used to download
import os # file handling
import tensorflow as tf # used for detections model
import numpy as np # V1.2, added to fix adv noise saving.
import urllib # add
import pandas as pd # add

logging.info('Done.')

if "VIPER_DF.pkl" in os.listdir():
  df = pd.read_pickle("VIPER_DF.pkl")
else:
  df = pd.DataFrame(columns = ['filename', 'detections'])

# Helper function which downloads from url

def save_df():
  df.to_pickle("VIPER_DF.pkl")

def Download(base_url, file_name):
  if file_name not in os.listdir():
    logging.info('Downloading ' + file_name)
    import urllib.request

    url = base_url + file_name
    urllib.request.urlretrieve(url, file_name)
    logging.info('Download Complete')

    # untar downloaded file if it is a tar file
    if file_name.find('.tar.gz') != -1:
      logging.info("Extracting " + file_name)
      import tarfile
      import shutil
      
      dir_name = file_name[0:-len('.tar.gz')]

      if os.path.exists(dir_name):
        shutil.rmtree(dir_name) 

      tarfile.open(file_name, 'r:gz').extractall('./')
      logging.info("Extraction Complete")
  else:
    logging.info(file_name + ' already exists.')


# Helper function which downloads the model, makes the object_classes, and returns it for later use.
def download_model():
  Download('https://raw.githubusercontent.com/nightrome/cocostuff/master/','labels.txt')
  Download('http://download.tensorflow.org/models/object_detection/', 'ssd_mobilenet_v1_coco_2018_01_28.tar.gz')
  


# Helper function which loads the model and later runs the model for every image
class Model():
  def __init__(self):

    # Code from here downward (but still init) are a list of lines that must be ran before running the model
    self.outputs = (
      'num_detections:0',
      'detection_classes:0',
      'detection_scores:0',
      'detection_boxes:0',
    )

    self.frozen_graph = os.path.join('ssd_mobilenet_v1_coco_2018_01_28', 'frozen_inference_graph.pb')
    
    with tf.io.gfile.GFile(self.frozen_graph, "rb") as f:
      self.graph_def = tf.compat.v1.GraphDef()
      self.loaded = self.graph_def.ParseFromString(f.read())
  
  # Runs Model
  def run_model(self,input_image):
    
    def wrap_graph(graph_def, inputs, outputs, print_graph=False):
      wrapped = tf.compat.v1.wrap_function(
        lambda: tf.compat.v1.import_graph_def(self.graph_def, name=""), [])

      return wrapped.prune(
        tf.nest.map_structure(wrapped.graph.as_graph_element, inputs),
        tf.nest.map_structure(wrapped.graph.as_graph_element, outputs))
    
    logging.info('Running model.')  
    model = wrap_graph(graph_def=self.graph_def, inputs=["image_tensor:0"], outputs=self.outputs)
    tensor = tf.convert_to_tensor([input_image], dtype=tf.uint8)
    detections = model(tensor)
    logging.info('Done.')

    return detections

#######################################################################
# VIPER Helpers
#######################################################################

# Gets a box bound, an image, and returns the actual pixel coordinates on the image
def VIPER_COCO_box_to_pixels(box_bounds, cv2_image):
    y1 = int((box_bounds[0]*cv2_image.shape[0]).numpy())
    x1 = int((box_bounds[1]*cv2_image.shape[1]).numpy())
    y2 = int((box_bounds[2]*cv2_image.shape[0]).numpy())
    x2 = int((box_bounds[3]*cv2_image.shape[1]).numpy())
    
    return [x1,x2,y1,y2]

# Returns if a number is private or not
def VIPER_COCO_priv_or_not(class_num):
  if class_num in [12, 14, 30, 33, 46, 63, 65, 66, 70, 71, 72, 73, 74, 75, 76, 77, 85, 110, 133,31,69,3]:
    return True
  else:
    return False

# Return if a number is private or not depending on relationship
def VIPER_COCO_priv_or_not_v2(relationship,class_num):
  SP = [31, 33, 46, 71, 72, 73, 74, 75, 76, 110]
  MP = [3, 12, 14, 63, 65, 66, 69, 70, 77, 85, 133]

  # if 2 .....	don't block 
  if relationship == 2:
    return False
  
  # if 1 .....	block semi private
  if relationship == 1:
    if class_num in SP:
      return True
  
  # if 0 (or anything else).....	block semi private and most private
  else:
    if class_num in SP or class_num in MP:
      return True

  return False

# Returns a tuple of (num_of_privates, num_of_non_privates)
def VIPER_COCO_priv_and_not(detected_classes_list):
  num_private = 0
  num_non_private = 0 

  for i in detected_classes_list:
    if VIPER_COCO_priv_or_not(int(i)):
      num_private += 1
    else:
      num_non_private += 1
  return num_private, num_non_private 

# IMPORTANT! MAY NEED CHANGING
def VIPER_object_blur(cv2_image, x1, x2, y1, y2, blurAmount):
    blurred_object = cv2_image[y1:y2,x1:x2]
    ksize = (blurAmount,blurAmount)
      
    blurred_object = cv.blur(blurred_object, ksize, cv.BORDER_DEFAULT)
    cv2_image[y1:y2,x1:x2] = blurred_object

    return cv2_image

# IMPORTANT! MAY NEED CHANGING
def VIPER_object_block(cv2_image, x1, x2, y1, y2, color):
    thickness = -1
    cv.rectangle(cv2_image,(x1,y1),(x2,y2), color, thickness)

    return cv2_image

#######################################################################
# Image Blueprint
#
# Usage: variable_name = Image(Image_Path)
#
# Variables:
# - image = cv2 image that will be manipulated with manipulators
# - original = backup of cv2 image just in case
# - detections = model results
# - num_of_detections = number of detections found in the image
# - detection_classes = list of classes from detected objects. ie: [17,18,18]
# - private_classes  = list of list. Inner list holds object class number and if they're private or not. ie. [[17, False],[18, False],[18, False]]
#######################################################################

class Image():
  privacy_score = None
  

  def __init__(self, image_path):
    download_model()
    global df
    
    if image_path in os.listdir():
      self.image = cv.imread(image_path)
      self.original = cv.imread(image_path)

    else:
      req = urllib.request.urlopen(image_path)
      arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
      self.image = cv.imdecode(arr, -1)
      
      self.original = cv.imdecode(arr, -1)

    if (df["filename"]==image_path).any():
      self.detections_index = (df[df["filename"]=='https://i.redd.it/48lqzsoobs861.jpg'].index.values).flat[0]
      self.detections = df.iloc[self.detections_index,1]
    else:
      self.detections = Model().run_model(self.image)
      df = df.append({'filename' : image_path, 'detections' : self.detections}, ignore_index = True)
      df.to_pickle("VIPER_DF.pkl")
    
    self.private_classes = []

    self.num_of_detections = len([x for x in self.detections[3][0] if not ((x == self.detections[3][0][-1])[0])])
    self.detection_classes = [self.detections[1][0][x].numpy() for x in range(self.num_of_detections)]

    self.object_classes = {}

    for line in open("labels.txt"):
      line = line[:-1]
      key, value = line.split(': ')
      self.object_classes[int(key)] = value

    for i in self.detection_classes:
      object_class = int(i)
      object_bool = VIPER_COCO_priv_or_not(object_class)

      self.private_classes.append([int(object_class), object_bool])

#######################################################################
# Image Modifiers
#######################################################################

# blur
def VIPER_blur(ImageType, relationship):
  num_of_detections = ImageType.num_of_detections
  detection_classes = ImageType.detection_classes

  for i in range(num_of_detections):
    box = ImageType.detections[3][0][i]
    box_class = ImageType.detections[1][0][i]

    if VIPER_COCO_priv_or_not_v2(relationship, box_class): 
      left, right, top, bottom = VIPER_COCO_box_to_pixels(box, ImageType.image)
      VIPER_object_blur(ImageType.image,left, right, top, bottom, 300)

# block
def VIPER_block(ImageType, relationship):
  num_of_detections = ImageType.num_of_detections
  detection_classes = ImageType.detection_classes

  for i in range(num_of_detections):
    box = ImageType.detections[3][0][i]
    box_class = ImageType.detections[1][0][i]

    color = (0, 0, 0)
    if VIPER_COCO_priv_or_not_v2(relationship, box_class):
      left, right, top, bottom = VIPER_COCO_box_to_pixels(box, ImageType.image)
      VIPER_object_block(ImageType.image, left, right, top, bottom, color)


# saliency
def VIPER_saliency_modifier(ImageType):
  saliency = cv.saliency.StaticSaliencySpectralResidual_create() #This intializes the saliency, and the below code creates a finer image which possibly could be used for instance segmentation

  (success, saliencyMap) = saliency.computeSaliency(ImageType.image) # this computes the map
  ImageType.image = (saliencyMap * 255).astype("uint8") # type casts and standardizes the image pixels 
  # threshMap = cv.threshold(saliencyMap, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1] #applies thereshold to get binary image from map 

# adverserial noise - FIX FIX FIX FIX
def VIPER_advarserial_modifier(ImageType):
  # Helper function to preprocess the image so that it can be inputted in MobileNetV2
  def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image

  def create_adversarial_pattern(input_image, input_label, pretrained_model):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
      tape.watch(input_image)
      prediction = pretrained_model(input_image)
      loss = loss_object(input_label, prediction)
      
    gradient = tape.gradient(loss, input_image) # Get the gradients of the loss w.r.t to the input image.
    signed_grad = tf.sign(gradient) # Get the sign of the gradients to create the perturbation

    return signed_grad

  pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,weights='imagenet')
  pretrained_model.trainable = False

  adv_image = preprocess(ImageType.image)
  image_probs = pretrained_model.predict(adv_image)

  # Get the input label of the image.
  labrador_retriever_index = 1
  label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
  label = tf.reshape(label, (1, image_probs.shape[-1]))

  perturbations = create_adversarial_pattern(adv_image, label, pretrained_model)

  eps = 0.05
  adv_x = adv_image + eps*perturbations
  adv_x = tf.clip_by_value(adv_x, -1, 1)
  ImageType.image = adv_x[0]
  ImageType.image = np.clip(ImageType.image, 0, 1)
  ImageType.image = (ImageType.image * 255).astype(np.uint8)

#######################################################################
# Image Handlers
#######################################################################

# image changer
def image_modifier(ImageType, modifier, relationship):
  if modifier.lower() == "blur" or modifier.lower()  == 'br':
    VIPER_blur(ImageType, relationship)
  elif modifier.lower()  == "block" or modifier.lower()  == 'bk':
    VIPER_block(ImageType, relationship)
  elif modifier.lower()  == "noise" or modifier.lower()  == 'ne':
    VIPER_advarserial_modifier(ImageType)
  elif modifier.lower()  == "saliency" or modifier.lower()  == 'sy':
    VIPER_saliency_modifier(ImageType)
  else:
    logging.info("No modifier ran.")

#image saver
def image_saver(ImageType):
  image=ImageType.image
  cv.imwrite('image.jpg',image)

# command line
def command_line(image_path, modifier, relationship):
  ImageType = Image(image_path)
  image_modifier(ImageType, modifier, relationship)
  
  image_saver(ImageType)
  return ImageType.image