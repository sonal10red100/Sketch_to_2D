# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:53:56 2020

@author: Chandraprakash Sharm
"""

import cv2
import os
import glob
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet
import numpy as np
import pprint as pp


from IPython import get_ipython
#from lines_old import wire_detection
from new import wire_detection
from new_reconst import reconstruct
#from new_reconst import not_gate
#from new_reconst import and_gate
#from new_reconst import or_gate
#from new_reconst import wires
#from new_reconst import construct_graph
#from new_reconst import connected
#from new_reconst import yrange
#from new_reconst import partialOverlap



def Overlap(l1, r1, l2, r2): 
    # If one rectangle is on left side of other 
    if(l1['x'] > r2['x'] or l2['x'] > r1['x']): 
        return False
    if(l1['y'] > r2['y'] or l2['y'] > r1['y']): 
        return False
    return True

def detection(inputImg):
    global size
    global predictions
    global tfnet2
   
    predictions = []
    predictions = tfnet2.return_predict(inputImg)
#    print(type(predictions))
    
#    filter the results obtained from model
    newImage = np.copy(inputImg)
    
    for i in range(len(predictions)):
        top_x = predictions[i]['topleft']['x']
        top_y = predictions[i]['topleft']['y']

        btm_x = predictions[i]['bottomright']['x']
        btm_y = predictions[i]['bottomright']['y']

        newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 10)

    cv2.imwrite("detected.jpg",newImage)
    newImage = np.copy(inputImg)
    size = newImage.shape
    sorted(predictions, key=lambda i: (i['topleft']['x'], i['topleft']['y']))
    
    for result in predictions:
        for r in predictions:
            if result!=r:
#                print('disticnt values'+ str(r['label'] + " " + str(result['label'])))
                if result["label"]==r["label"] and Overlap(result['topleft'], result['bottomright'], r['topleft'], r['bottomright']):
#                    print("match")
                    if result['confidence']>r['confidence']:
                        predictions.remove(r)
                    else:
                        predictions.remove(result)
#    print(predictions)
    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
       
        label = result['label'] + " " + str(round(confidence, 3))
        newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 1)
        newImage = cv2.putText(newImage, label, (top_x, top_y+20), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (0, 0, 0), 1, cv2.LINE_AA)
        size = newImage.shape
    return (newImage , predictions)

#    initialize variables for detection 
options = {"model": "cfg/tiny-yolo-voc-3c.cfg",
           "load": 8625,
           "gpu": 1.0,
           "threshold":0.5}
tfnet2 = TFNet(options)

'''
for processing single file
'''

original_img = cv2.imread("result/app/input/IMG_20200113_223126__01.jpg")
original_img = cv2.resize(original_img, (800, 400), interpolation = cv2.INTER_AREA)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
cv2.imwrite('temp.jpg', original_img)
#fig, ax = plt.subplots(figsize=(15, 15))
#ax.imshow(original_img)
outimg = detection(original_img)
fig, bx = plt.subplots(figsize=(15, 15))
bx.imshow(outimg[0])
print("wire_detection")
comp = wire_detection(outimg[1])
#f = open("result\\app\\components\\"+str(file_name)+".txt", 'w')
print("-------------------------------------------------------")
print(comp)
print("\n-------------------------------------------------------")
#if (comp is not None):
#    comp = reconstruct(comp, (800, 400))
#print("---------------------------------------------------------")
#print("comp")
#for c in comp:
#    print(c['label'], c['tlx'], c['tly'], c['brx'], c['bry'])
#print("-----------------------------------------------------------\n")
rimage=cv2.imread("out.jpg")
fig, cx = plt.subplots(figsize=(15, 15))
cx.imshow(rimage)
get_ipython().magic('reset -sf')
