from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import path
train = pd.read_csv('human_labels.csv')
#train = pd.read_csv('human_labels.csv')

if not os.path.exists('flipped_images'):
    os.mkdir('flipped_images')

if not os.path.exists('scaled_images'):
    os.mkdir('scaled_images')
if not os.path.exists('rotated_images'):
    os.mkdir('rotated_images')
if not os.path.exists('sheared_images'):
    os.mkdir('sheared_images')
if not os.path.exists('flip_scaled_images'):
    os.mkdir('flip_scaled_images')
if not os.path.exists('flip_translated_images'):
    os.mkdir('flip_translated_images')
a = train.iloc[:, :].values

h = 6   # number of unique images in train
Matrix = [[] for y in range(h)] 

# segregrating entries per image
k = 0
img_name = []
img_name.append(a[0][0])
Matrix[k].append(a[0][1:6])
for i in range(1, len(a)):
    if a[i][0] == a[i-1][0]:
        Matrix[k].append(a[i][1:6])
    else:
        k = k+1
        Matrix[k].append(a[i][1:6])
        img_name.append(a[i][0])

#Matrix[0], Matrix[1] etc are my bounding boxes


########### flipping
k = 401

for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)
    
    transforms = Sequence([RandomHorizontalFlip(1)])
    img, bboxes = transforms(img, bboxes)

    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'flipped_images/img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'flipped_images/img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('flipped_images/img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('FLIP.csv', index = False)
    

############ scaling
k = k+1

for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    scale = RandomScale(0.2, diff = True)
    img,bboxes = scale(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'scaled_images/img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'scaled_images/img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('scaled_images/img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('SCALED.csv', index = False)
    
    
    
    
######rotation
   
k = k+1

for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    rotate = RandomRotate(10)  ## rotating by 10 degrees
    img, bboxes = rotate(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'rotated_images/img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'rotated_images/img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('rotated_images/img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('ROTATION.csv', index = False)
    
    
    
########################Shearing
k = k+1

for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    shear = RandomShear(0.7)  
    img, bboxes = shear(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'sheared_images/img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'sheared_images/img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('sheared_images/img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('SHEAR.csv', index = False)
    
    
    
   
#########flip and scale
k = k+1

for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    transforms = Sequence([RandomHorizontalFlip(1), RandomScale(0.3, diff = True)])
    img, bboxes = transforms(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'flip_scaled_images/img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'flip_scaled_images/img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('flip_scaled_images/img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('FLIP_SCALE.csv', index = False)
    
    

##### flip and translation
k=k+1
for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    transforms = Sequence([RandomHorizontalFlip(1), RandomTranslate(0.3, diff = True)])
    img, bboxes = transforms(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'flip_translated_images/img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'flip_translated_images/img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('flip_translated_images/img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('FLIP_TRANSLATE.csv', index = False)

