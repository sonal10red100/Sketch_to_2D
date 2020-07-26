# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 22:38:39 2020

@author: Maitreyi Sharma
"""
import cv2
import os
import glob
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet
import numpy as np
import pprint as pp


#from IPython import get_ipython
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
for processing multiple images
'''
comp = []
img_dir = "result\\app\\input\\" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)

for f1 in files:
    img = cv2.imread(f1)
    img = cv2.resize(img, (800, 400), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('temp.jpg', img)
    darr = detection(img)
    print(str(f1))
    file_name = os.path.basename(str(f1))[0:os.path.basename(str(f1)).find('.')]
    cv2.imwrite("result\\app\\output\\"+str(file_name)+".jpg",darr[0])
    
    comp = wire_detection(darr[1])
    comp = reconstruct(comp, (800, 400))
#    f = open("result\\app\\components\\"+str(file_name)+".txt", 'w')
#    for c in comp:
#        f.write(str(c)+"\n")
#    f.close()
    rimage=cv2.imread("out.jpg")
#    fig, cx = plt.subplots(figsize=(15, 15))
#    cx.imshow(rimage)
    cv2.imwrite("result\\app\\circuit\\"+str(file_name)+".jpg", rimage)
#    wimg = cv2.imread("result/app/wires_endpoint/ob.jpg")
#    cv2.imwrite("result\\app\\wires_endpoints\\" + str(file_name) + ".jpg", wimg)
#    get_ipython().magic('reset -sf')
'''
    for processing single file
    '''
    
#    original_img = cv2.imread("dataset/images/IMG_20200113_221848.jpg")
#    original_img = cv2.resize(original_img, (800, 400), interpolation = cv2.INTER_AREA)
#    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
#    cv2.imwrite('temp.jpg', original_img)
#    #fig, ax = plt.subplots(figsize=(15, 15))
#    #ax.imshow(original_img)
#    outimg = detection(original_img)
#    fig, bx = plt.subplots(figsize=(15, 15))
#    bx.imshow(outimg[0])
#    print("wire_detection")
#    comp = wire_detection(outimg[1])
#    #f = open("result\\app\\components\\"+str(file_name)+".txt", 'w')
#    print("-------------------------------------------------------")
#    print(comp)
#    print("\n-------------------------------------------------------")
#    if (comp is not None):
#        reconstruct(comp, (800, 400))
#    rimage=cv2.imread("out.jpg")
#    fig, cx = plt.subplots(figsize=(15, 15))
#    cx.imshow(rimage)
#    get_ipython().magic('reset -sf')
