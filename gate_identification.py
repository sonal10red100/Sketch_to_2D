# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 12:21:31 2020

@author: Chandraprakash Sharm
"""


import matplotlib.pyplot as plt
import numpy as np

from darkflow.net.build import TFNet
import cv2
import pprint as pp

def Overlap(l1, r1, l2, r2): 
    # If one rectangle is on left side of other 
    if(l1['x'] > r2['x'] or l2['x'] > r1['x']): 
        return False
    if(l1['y'] > r2['y'] or l2['y'] > r1['y']): 
        return False
    return True

def boxing(original_img , predictions):
    newImage = np.copy(original_img)
    
#    prediction = np.array([])
    
    sorted(predictions, key=lambda i: (i['topleft']['x'], i['topleft']['y']))
    
    for result in predictions:
        for r in predictions:
            if result!=r:
#                print('disticnt values')
                if result["label"]==r["label"] and Overlap(result['topleft'], result['bottomright'], r['topleft'], r['bottomright']):
                    print("match")
                    if result['confidence']>r['confidence']:
                        predictions.remove(r)
                    else:
                        predictions.remove(result)
#                if result['label']=="not" or r["label"]=="not":
#                    if (internal_to(result["topleft"], result["bottomright"], r["topleft"], r["bottomright"])):
#                        x = {}
#                        x["label"] = str("nand") if (r['label']=='and' or result['label']=='and') else str("nor")
#                        x["confidence"] = min(r["confidence"], result["confidence"])
#                        tf = {}
#                        tf['x'] = min(r['topleft']['x'], result['toopleft']['x'])
#                        tf['y'] = min(r['topleft']['y'], result['toopleft']['y'])
#                        x["topleft"]=tf
#                        br = {}
#                        br['x'] = max(r['topleft']['x'], result['toopleft']['x'])
#                        br['y'] = max(r['topleft']['y'], result['toopleft']['y'])
#                        x["bottomright"] = br
#                        print(x)
#                        
#                        predictions.remove(r)
#                        predictions.remove(result)
#                        np.append(predictions, [x])
                    
            
    print(predictions)
    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
       
        label = result['label'] + " " + str(round(confidence, 3))
        
        newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
        newImage = cv2.putText(newImage, label, (top_x, top_y+100), cv2.FONT_HERSHEY_COMPLEX_SMALL , 5, (0, 0, 0), 3, cv2.LINE_AA)
        
        
        
    return newImage

options = {"model": "cfg/tiny-yolo-voc-3c.cfg",
           "load": 6000,
           "gpu": 1.0,
          "threshold":0.5}

tfnet2 = TFNet(options)


original_img = cv2.imread("dataset/images/IMG_20200119_191711.jpg")
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
results = tfnet2.return_predict(original_img)
print(results)
#print(len(results))

fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(original_img)

fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(boxing(original_img, results))