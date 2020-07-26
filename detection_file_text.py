# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:23:07 2020

@author: Chandraprakash Sharm
"""


import cv2
import os
import glob
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet
import numpy as np
import pprint as pp

def Overlap(l1, r1, l2, r2): 
    # If one rectangle is on left side of other 
    if(l1['x'] > r2['x'] or l2['x'] > r1['x']): 
        return False
    if(l1['y'] > r2['y'] or l2['y'] > r1['y']): 
        return False
    return True

def detection(inputImg, file_name):
    global size
    global predictions
    global tfnet2
   
    predictions = []
    predictions = tfnet2.return_predict(inputImg)
#    print(type(predictions))
    
#    filter the results obtained from model
    newImage = np.copy(inputImg)
    size = newImage.shape
    sorted(predictions, key=lambda i: (i['topleft']['x'], i['topleft']['y']))
    
    for result in predictions:
        for r in predictions:
            if result!=r:
                print('disticnt values'+ str(r['label'] + " " + str(result['label'])))
                if result["label"]==r["label"] and Overlap(result['topleft'], result['bottomright'], r['topleft'], r['bottomright']):
                    print("match")
                    if result['confidence']>r['confidence']:
                        predictions.remove(r)
                    else:
                        predictions.remove(result)
    print(predictions)
    f = open("result\\test\\output_txt\\"+file_name+".txt", "w")
    for p in predictions:
        f.write(str(p['label'])+" "+str(p['confidence'])+" "+str(p['topleft']['x'])+" "+str(p['topleft']['y'])+" "+str(p['bottomright']['x'])+" "+str(p['bottomright']['y'])+"\n")   
    f.close()
#    initialize variables for detection 
options = {"model": "cfg/tiny-yolo-voc-3c.cfg",
           "load": 8625,
           "gpu": 1.0,
           "threshold":0.5}
tfnet2 = TFNet(options)

img_dir = "result\\test\\input\\" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)

for f1 in files:
#    print(str(f1))
    img = cv2.imread(f1)
#    img = cv2.resize(img, (800, 400), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    file_name = os.path.basename(str(f1))[0:os.path.basename(str(f1)).find('.')]
    print(file_name)
    detection(img, file_name)
#    f.write(str(darr[1]) + "\n\n")
#    cv2.imwrite("result\\output\\ckpt_300_test_output\\"+str(i)+".jpg",darr[0])