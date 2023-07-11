# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:02:08 2023

@author: talha
"""

import csv
import sys
import os
import os.path as osp
from fmutils import fmutils as fmu
from PIL import Image

def extract_crop(img_path, bb_left, bb_top, bb_width, bb_height):
    # Open the image file
    img = Image.open(img_path)

    # Calculate the right and bottom coordinates of the bounding box
    bb_right = bb_left + bb_width
    bb_bottom = bb_top + bb_height

    # Extract the crop
    crop = img.crop((bb_left, bb_top, bb_right, bb_bottom))

    return crop

'''
<frame_number>,<identity_number>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<confidence>,<class>,<visibility>
'''
camera = 'cannon2'
cam_id = 2 # use different for each camera or video sequence

op_dir = f'D:/D/Anacondas_python/Data_Processors/khubaib_joint2points/tracking data/processed_v2/'
data_path = f'D:/D/Anacondas_python/Data_Processors/khubaib_joint2points/tracking data/{camera}/img/'
lbls = f'D:/D/Anacondas_python/Data_Processors/khubaib_joint2points/tracking data/{camera}/gt/gt.txt'


# Parse the gt.txt file
with open(lbls, 'r') as f:
    reader = csv.reader(f)
    gt_data = [row for row in reader]

max_pid = 0
for data in gt_data:
    frame_number = int(data[0])
    '''WARNING carefully shift the IDs by adding'''
    identity_number = int(data[1]) + 0 # pid  add 0 for first video sequence
    bb_left = float(data[2])
    bb_top = float(data[3])
    bb_width = float(data[4])
    bb_height = float(data[5])
    # confidence = int(data[6])
    # class_id = int(data[7])
    # visibility = float(data[8])
    
    # get the max pid in the video frame
    if max_pid < identity_number:
        max_pid = identity_number

    # Now you have the bounding box coordinates and can use them to extract the crop
    img_path = f'{data_path}/{frame_number:06}.PNG'
    crop = extract_crop(img_path, bb_left, bb_top, bb_width, bb_height)
    crop = crop.save(f"{op_dir}/{frame_number}_{identity_number}_{cam_id}_{frame_number:06}.png")

print(f'Now start PIDs from {max_pid}')
#%%
x = os.listdir('D:/D/Anacondas_python/Data_Processors/khubaib_joint2points/tracking data/processed_v2/')
y = []
for i in range(len(x)):
    y.append(x[i].split('_')[1:3])

a,b,c = [],[],[]
for i in range(len(y)):
    if y[i][1] == '0':
        a.append(y[i][0])
    if y[i][1] == '1':
        b.append(y[i][0])
    if y[i][1] == '2':
        c.append(y[i][0])
    
print(set(a), set(b), set(c))  
print('None of the sets should have any overlapping values')