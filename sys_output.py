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
from final_bool import gen_expression


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
cnt=0
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.resize(img, (800, 400), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('temp.jpg', img)
    darr = detection(img)
#    print(str(f1))
    file_name = os.path.basename(str(f1))[0:os.path.basename(str(f1)).find('.')]
#    cv2.imwrite("result\\app\\component_output\\"+str(file_name)+".jpg",darr[0])
    
    comp = wire_detection(darr[1])
    comp = reconstruct(comp, (800, 400))
#    f = open("result\\app\\components\\"+str(file_name)+".txt", 'w')
#    f.write("[")
#    for c in comp:
#        f.write(str(c)+',')
#    f.write("]")
#    f.close()
    exp = gen_expression(comp)
    f=open('result\\app\\logic_expression_detect\\' + str(file_name)+".txt", 'w')
    f.write(exp)
    wire_count = 0
    for c in comp:
        if c['label'] == 'wire':
            wire_count += 1
    f.write("\n"+str(wire_count))
    f.close()
#    print("********************************************************")
#    print(file_name, exp, wire_count)
#    print("*********************************************************")
#    if cnt==3:
#        break
#    cnt+=1
print('Finished')
#    f.close()
    
#    rimage=cv2.imread("out.jpg")
#    fig, cx = plt.subplots(figsize=(15, 15))
#    cx.imshow(rimage)
#    cv2.imwrite("result\\app\\circuit\\"+str(file_name)+".jpg", rimage)
#    wimg = cv2.imread("result/app/wires_endpoint/ob.jpg")
#    cv2.imwrite("result\\app\\wires_endpoints\\" + str(file_name) + ".jpg", wimg)
#    get_ipython().magic('reset -sf')

