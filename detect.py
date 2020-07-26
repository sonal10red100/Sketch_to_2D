# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:16:33 2020

@author: Chandraprakash Sharm
"""

import networkx as nx
import tkinter.filedialog
import tkinter
from tkinter import *
from tkinter.ttk import *
import numpy as np
import cv2
from tokToolTip import CreateToolTip
#import pprint as pp
#import matplotlib.pyplot as plt
#from darkflow.net.build import TFNet
#from detect.py import detection
from PIL import Image
from PIL import ImageTk

from PIL import ImageDraw
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet
import pprint as pp

#from lines import interpolate_pixels_along_line
from new import wire_detection
from bool_exp import expression
from new_reconst import reconstruct
from new_reconst import not_gate
from new_reconst import and_gate
from new_reconst import or_gate
from new_reconst import wires
from new_reconst import construct_graph
from new_reconst import connected
from new_reconst import yrange
from new_reconst import partialOverlap

dimensions = (800, 400)


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
    
    options = {"model": "cfg/tiny-yolo-voc-3c.cfg",
               "load": 8625,
               "gpu": 1.0,
               "threshold":0.5}
    tfnet2 = TFNet(options)
    predictions = tfnet2.return_predict(inputImg)
    print(type(predictions))
#    filter the results obtained from model
    
    newImage = np.copy(inputImg)
    
    for i in range(len(predictions)):
        top_x = predictions[i]['topleft']['x']
        top_y = predictions[i]['topleft']['y']

        btm_x = predictions[i]['bottomright']['x']
        btm_y = predictions[i]['bottomright']['y']

        newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 10)

    cv2.imwrite("detected.jpg",newImage)
    size = newImage.shape
    sorted(predictions, key=lambda i: (i['topleft']['x'], i['topleft']['y']))
    newImage = np.copy(inputImg)
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
    print(predictions)
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
#        cv2.imwrite("out1.jpg", newImage)
    return newImage



#To adjust the size of the image
def adjust(img):
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print(resized.shape)
    return (resized)

#Input image selection
def inputImage():
    global panelA
    global path

    path=tkinter.filedialog.askopenfilename()
    if len(path)>0:
        #reading the image and pre-processing it for display
        image=cv2.imread(path)
        image=cv2.resize(image, dimensions, interpolation = cv2.INTER_AREA)
        cv2.imwrite('temp.jpg', image)
        
        
        image = cv2.imread('temp.jpg')
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=adjust(image)
        image=Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        
        print(path)
#        display input image when image is selected.
        if panelA is None:
            panelA=tkinter.Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
            
        else:
            panelA.configure(image=image)
            panelA.image=image

        
#processing started and detected gate display
def detectComponent():
    global panelB
    global path
    
#    path=tkinter.filedialog.askopenfilename()
    
    image=cv2.imread('temp.jpg')
    if image is not None:
        
#        cv2.imwrite('temp.jpg', image)
        
#        image = cv2.imread('temp.jpg')
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        oimage=detection(image)
        oimage=adjust(oimage)
        oimage=Image.fromarray(oimage)
        oimage = ImageTk.PhotoImage(oimage)
        
        if panelB is None:
            panelB=tkinter.Label(image=oimage)
            panelB.image = oimage
            panelB.pack(side="right", padx=10, pady=10)
            
        else:
            panelB.configure(image=oimage)
            panelB.image=oimage
            
#    else:
#            pop up error dialog for improper file selection

def reconstructComponent():
    global panelC
    global predictions
#    global path
#    global size
#    reconstruct(predictions, size)
#    components = [{'label': 'and', 'confidence': 0.81399184, 'topleft': {'x': 1253, 'y': 204}, 'bottomright': {'x': 1897, 'y': 1295}}, {'label': 'and', 'confidence': 0.7143245, 'topleft': {'x': 2597, 'y': 903}, 'bottomright': {'x': 3453, 'y': 1979}},{'label':'wire', 'dir':'h', 'topleft':{'x': 545, 'y':565}, 'bottomright':{'x':1253, 'y':545}},
#    {'label':'wire', 'dir':'h', 'topleft':{'x': 517, 'y':929}, 'bottomright':{'x':1253, 'y':873}},{'label':'wire', 'dir':'h', 'topleft':{'x': 1897, 'y':601}, 'bottomright':{'x':2209, 'y':593}},{'label':'wire', 'dir':'v', 'topleft':{'x': 2209, 'y':593}, 'bottomright':{'x':2317, 'y':1353}},{'label':'wire', 'dir':'h', 'topleft':{'x': 2317, 'y':1353}, 'bottomright':{'x':2597, 'y':1413}},{'label':'wire', 'dir':'h', 'topleft':{'x': 933, 'y':1769}, 'bottomright':{'x':2597, 'y':1661}},{'label':'wire', 'dir':'h', 'topleft':{'x': 3453, 'y':1301}, 'bottomright':{'x':3717, 'y':1301}}]
# 
#    components =   [{'label': 'and', 'confidence': 0.79810315, 'topleft': {'x': 1485, 'y': 244}, 'bottomright': {'x': 2348, 'y': 1256}}, {'label': 'or', 'confidence': 0.76538694, 'topleft': {'x': 2912, 'y': 849}, 'bottomright': {'x': 3914, 'y': 1664}}, {'label': 'not', 'confidence': 0.73433167, 'topleft': {'x': 3917, 'y': 1143}, 'bottomright': {'x': 4128, 'y': 1360}},
#                {'label':'wire', 'topleft':{'x': 521, 'y':673}, 'bottomright':{'x':1485, 'y':625}},{'label':'wire', 'topleft':{'x': 533, 'y':937}, 'bottomright':{'x':1485, 'y':857}},{'label':'wire', 'topleft':{'x': 2348, 'y':697}, 'bottomright':{'x':2529, 'y':677}},{'label':'wire', 'topleft':{'x': 2529, 'y':677}, 'bottomright':{'x':2529, 'y':1189}},{'label':'wire', 'topleft':{'x': 2529, 'y':1189}, 'bottomright':{'x':2912, 'y':1141}},{'label':'wire', 'topleft':{'x': 481, 'y':1529}, 'bottomright':{'x':2912, 'y':1353}},{'label':'wire', 'topleft':{'x': 4128, 'y':1189}, 'bottomright':{'x':4597, 'y':1169}}]      
#    print("REconstruciton step 1")
#    print(predictions)
    components = wire_detection(predictions)
#    components = predictions
    print("print components")
    print(components)
    reconstruct(components, dimensions)
#    print("completed")
#    print(size)
    rimage=cv2.imread("out.jpg")
#    print("reconstructing")
    components.clear()
    if rimage is not None:
        rimage=cv2.cvtColor(rimage, cv2.COLOR_BGR2RGB)
#        print('read image')
        rimage=adjust(rimage)
        rimage=Image.fromarray(rimage)
        rimage=ImageTk.PhotoImage(rimage)
        
        if panelC is None:
            panelC=tkinter.Label(image=rimage)
            panelC.image = rimage
            panelC.pack(side="bottom", padx=10, pady=10)
            
        else:
            panelC.configure(image=rimage)
            panelC.image=rimage

#    if panelD is None:
#        panelD=tkinter.Label(image=rimage)
#        
#        panelD.pack(side="top", padx=10, pady=10)
#        
#    else:
#        panelD.configure(image=rimage)
#        panelD.image=rimage

def logic_expression():  
    global components
    
    exp = expression(component)
    
#size = (2176, 4600, 3)
predictions = None
path=""
# optimisations for improved response time.
#if __name__ == '__main__':
#    options = {"model": "cfg/tiny-yolo-voc-3c.cfg",
#               "load": 8625,
#               "gpu": 1.0,
#               "threshold":0.5}
#    tfnet2 = TFNet(options)

root = tkinter.Tk()
root.title("Sketch to 2D circuit converter")
root.minsize(350, 150)
panelA = None
panelB = None
panelC = None

print("done")

btn1 = tkinter.Button(root, text="Select an image", command=inputImage)
print("done1")
btn1.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
print("done2")
btn1_ttp = CreateToolTip(btn1, "Image selection for processing")
print("tooltip done")

btn2 = tkinter.Button(root, text="Detect Components", command=detectComponent)
print("done3")
btn2.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
print("done4")
btn2_ttp = CreateToolTip(btn2, "Press to detect components in the image")
print("tooltip done")

btn3 = tkinter.Button(root, text="Reconstruct Circuit", command=reconstructComponent)
print("done5")
btn3.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
print("done6")
btn3_ttp = CreateToolTip(btn3, "Convert the circuit to 2D")
print("tooltip done")

btn4 = tkinter.Button(root, text="Logical Expression", command=logic_expression)
print("done7")
btn4.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
print("done6")
btn4_ttp = CreateToolTip(btn4, "Generate Logical Expression for the image")
print("tooltip done")

# kick off the GUI
root.mainloop()
print("done8")