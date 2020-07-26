# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:16:33 2020

@author: Chandraprakash Sharm
"""

import tkinter.filedialog
import tkinter
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import tkinter.font as font 
import numpy as np
import cv2
import os
from PIL import Image
from PIL import ImageTk

from PIL import ImageDraw
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet
import pprint as pp

from tokToolTip import CreateToolTip
from simulation_table import Table

from new import wire_detection
from final_bool import gen_expression
from new_reconst import reconstruct
from table import gen_truth_table
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
def adjust(img, re_true):
    if re_true == True:
        scale_w_percent = 74 # percent of original size
        scale_h_percent = 95
        width = int(img.shape[1] * scale_w_percent / 100)
        height = int(img.shape[0] * scale_h_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        print(resized.shape)
        return (resized)
    else:
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
    global panelIn
    global path
    global inImgLabel

    path=tkinter.filedialog.askopenfilename()
    if len(path)>0:
        #reading the image and pre-processing it for display
        image=cv2.imread(path)
        image=cv2.resize(image, dimensions, interpolation = cv2.INTER_AREA)
        cv2.imwrite('temp.jpg', image)
        
        
        image = cv2.imread('temp.jpg')
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=adjust(image, False)
        image=Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        
        print(path)
#        display input image when image is selected.
        if panelIn is None:
#            panelIn_title.pack(side="top", fill="both", expand="no", padx="10", pady="10")
#            panelIn=tkinter.Label(image=image, textvariable=inImgLabel, compound=tkinter.BOTTOM, font="Helvetica 12", bg='#e6f2ff')
            panelIn.image = image
#            panelIn.pack(side="left", padx=10, pady=10)
            
        else:
            panelIn.configure(image=image)
            panelIn.image=image

        
#processing started and detected gate display
def detectComponent():
    global panelDe
    global path
    global predictions
    global deImgLabel
    
#    path=tkinter.filedialog.askopenfilename()
    image=cv2.imread('temp.jpg')
    if image is not None:
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        oimage=detection(image)
        if (predictions == []):
            tkinter.messagebox.showwarning("warning", "Sorry.Could not identify any components.")
            print("no detection")
        else:
            oimage=adjust(oimage, False)
            oimage=Image.fromarray(oimage)
            oimage = ImageTk.PhotoImage(oimage)
            
            if panelDe.image is None:
#                panelDe_title.pack(side="top", fill="both", expand="no", padx="10", pady="10")
#                panelDe=tkinter.Label(image=oimage, textvariable=deImgLabel, compound=tkinter.BOTTOM, font="Helvetica 12", bg='#e6f2ff')
                panelDe.configure(image=oimage)
                panelDe.image=oimage
#                panelDe.pack(side="right", padx=10, pady=10)
                
            else:
                panelDe.configure(image=oimage)
                panelDe.image=oimage
                
    else:
        tkinter.messagebox.showerror("Error", "No image selected.")
#            pop up error dialog for improper file selection

def reconstructComponent():
    global panelRe
    global predictions
    global components
    global reImgLabel
    
    img = cv2.imread('temp.jpg')
    if img is None:
        tkinter.messagebox.showinfo("Information","Please select image and detect its components")
    else:
        res = messagebox.askquestion("Confirm", "Are all gates correctly detected?")
        print(res)
        if (res == 'yes'):
            components = wire_detection(predictions)
            print(components)
            if(predictions==[] or components==[] or predictions == None or components == None):
                tkinter.messagebox.showwarning("warning","No components to reconstruct the circuit.")
            else:
                components = reconstruct(components, dimensions)
                rimage=cv2.imread("out.jpg")
#                components.clear()
                if rimage is not None:
                    rimage=cv2.cvtColor(rimage, cv2.COLOR_BGR2RGB)
                    rimage=adjust(rimage, True)
                    rimage=Image.fromarray(rimage)
                    rimage=ImageTk.PhotoImage(rimage)
                    
                    if panelRe.image is None:
#                        panelRe_title.pack(side="bottom", fill="both", expand="no", padx="10", pady="10")
#                        panelRe=tkinter.Label(image=rimage, textvariable=reImgLabel, compound=tkinter.BOTTOM, font="Helvetica 12", bg='#e6f2ff')
                        panelRe.configure(image=rimage)
                        panelRe.image=rimage
#                        panelRe.pack(side="bottom", padx=10, pady=10)
                        
                    else:
                        panelRe.configure(image=rimage)
                        panelRe.image=rimage
                    tkinter.messagebox.showinfo("Information", "Reconstructed file is saved as out.jpg")

def logic_expression():  
    global components
    global predictions
    global var2
    global exp
    global expression 
    
    print("---------------------------------------------------------")
    print("components")
#    for c in components:
#        print(c['label'], c['topleft']['x'], c['topleft']['y'], c['bottomright']['x'], c['bottomright']['y'], c['tlx'], c['tly'], c['brx'], c['bry'])
#    print("-----------------------------------------------------------\n")
    img = cv2.imread('temp.jpg')
    if img is None:
        tkinter.messagebox.showinfo("Information", "Please select image, detect and reconstruct it.")
    elif(components==[] or predictions == None or predictions == [] or components == None):
            tkinter.messagebox.showwarning("warning", "No components identified to generate expression.")
    else:
        expression = gen_expression(components)
        print("from expression generation code ",expression)
        text = "Logical Expression " + str(expression) + " "
        var2.set(text)
        exp.config(fg='#003366')
        lst = []
        print("after adding to the label",expression)
#        gen_truth_table(expression)
        
#        exp.pack(side="top", fill="both", expand="yes", padx="0", pady="0")
    
def simulateExp():
    global expression
    print("simulation", expression)
    if(expression == None):
        tkinter.messagebox.showwarning("Warning", "Could not find expression for simulation")
    elif(expression == ""):
        tkinter.messagebox.showinfo("Information", "No expression found")
    else:
        gen_truth_table(expression)
    

def Help():
    print("help tab")
    os.startfile('help.txt')

expression = None
predictions = None
components = None
path=""
options = {"model": "cfg/tiny-yolo-voc-3c.cfg",
           "load": 8625,
           "gpu": 1.0,
           "threshold":0.45}
tfnet2 = TFNet(options)

# optimisations for improved response time.

if __name__ == '__main__':


    root = tkinter.Tk()
    root.title("DLC Rebuilder")
    root.minsize(1060, 600)
    root.geometry("1060x550+50+20")
    root.config(background='white')
    
    fontText = font.Font(size=12, weight='bold')
    fontTitle = font.Font(size=24, weight='bold')
    
    menu = Menu(root)
    root.config(menu=menu)
    helpmenu = Menu(menu) 
    menu.add_cascade(label='Help', menu=helpmenu) 
    helpmenu.add_command(label='About')
    helpmenu.add_command(label='Help', command=Help)
    
    
    print("done")
    var1 = tkinter.StringVar()
    var2 = tkinter.StringVar()
    inImgLabel = tkinter.StringVar()
    deImgLabel = tkinter.StringVar()
    reImgLabel = tkinter.StringVar()
    inImgLabel.set('Input Sketch')
    deImgLabel.set('Components Detected')
    reImgLabel.set('Reconstructed Ciruit')
    var2.set('No Expression')    
    var1.set('DLC Rebuilder')
    exp = tkinter.Label(root, textvariable = var2, bg='#ffffff', relief = "groove", borderwidth=5)
    exp['font'] = fontText
    
    title = tkinter.Label(root, textvariable = var1, bg='#ffffff')
    title['font'] = fontTitle
    title.place(x = 10, y = 10, width = 1060, height = 40)
    
#    exp.pack(side="top", fill="both", expand="yes", padx="0", pady="0")
#    exp.pack_forget()  
    img = None
    panelIn=tkinter.Label(image=img, textvariable=inImgLabel, compound=tkinter.BOTTOM, font="Helvetica 12", bg='#e6f2ff', relief = "groove", borderwidth=5)
    panelIn.image = img

    panelDe = tkinter.Label(image=img, textvariable=deImgLabel, compound=tkinter.BOTTOM, font="Helvetica 12", bg='#e6f2ff', relief = "groove", borderwidth=5)
    panelDe.image = img

    panelRe = tkinter.Label(image=img, textvariable=reImgLabel, compound=tkinter.BOTTOM, font="Helvetica 15", bg='#e6f2ff', relief = "groove", borderwidth=5)
    panelRe.image = img
    
    btn0 = tkinter.Button(root, text="Expression Calculator", command=simulateExp, bg='#24a0ed', fg ='#ffffff')
    btn0['font'] = fontText
    btn0_ttp = CreateToolTip(btn0, "Simulate truth table for the expression")
                          
    btn1 = tkinter.Button(root, text="Logical Expression", command=logic_expression, bg='#24a0ed', fg ='#ffffff')
    btn1['font'] = fontText
    btn1_ttp = CreateToolTip(btn1, "Generate Logical Expression for the image")
    
    btn2 = tkinter.Button(root, text="Reconstruct Circuit", command=reconstructComponent, bg='#24a0ed', fg ='#ffffff')
    btn2['font'] = fontText
    btn2_ttp = CreateToolTip(btn2, "Convert the circuit to 2D")

    btn3 = tkinter.Button(root, text="Detect Components", command=detectComponent, bg='#24a0ed', fg ='#ffffff')
    btn3['font'] = fontText
    btn3_ttp = CreateToolTip(btn3, "Press to detect components in the image")
    
    btn4 = tkinter.Button(root, text="Select Image", command=inputImage, bg='#24a0ed', fg ='#ffffff')
    btn4['font'] = fontText
    btn4_ttp = CreateToolTip(btn4, "Image selection for processing")

    # co ordinates for buttons
    hor_padding = 10
    btn_width = 200
    btn_start_x = 10
    btn_start_y = 10 + 50
    btn_height = 35
    
    # co ordinates for panels
    panIn_x = 10
    panIn_y = 55 + 50
    panIn_width = 412
    panIn_height = 237
    
    panDe_x = panIn_x
    panDe_y = panIn_y + panIn_height + 10
    panDe_width = panIn_width
    panDe_height =  238
    
    panRe_x = 10 + panIn_x + panIn_width
    panRe_y = 100 + 50
    panRe_width = 620
    panRe_height = 440
   
    # co ordinates for expression
    exp_x = 10 + panIn_x + panIn_width
    exp_y = 55 + 50
    exp_width = 620
    exp_height = 35
    
    
    
    # placing componenets
    btn4.place(x = btn_start_x, y = btn_start_y, width = btn_width, height = 35)
    btn3.place(x = btn_start_x + hor_padding + btn_width, y = btn_start_y, width = btn_width, height = 35)
    btn2.place(x = (btn_start_x + (hor_padding + btn_width)*2), y = btn_start_y, width = btn_width, height = 35)
    btn1.place(x = (btn_start_x + (hor_padding + btn_width)*3), y = btn_start_y, width = btn_width, height = 35)
    btn0.place(x = (btn_start_x + (hor_padding + btn_width)*4), y = btn_start_y, width = btn_width, height = 35)
    
    exp.place(x = exp_x, y = exp_y, width = exp_width, height = exp_height)
    
    panelIn.place(x = panIn_x, y = panIn_y, width = panIn_width, height = panIn_height)
    panelDe.place(x = panDe_x, y = panDe_y, width = panDe_width, height = panDe_height)
    panelRe.place(x = panRe_x, y = panRe_y, width = panRe_width, height = panRe_height)
    
    # kick off the GUI
    root.mainloop()
#    os.remove('temp.jpg')
    print("done8")


#if __name__ == '__main__':
#
#
#    root = tkinter.Tk()
#    root.title("Sketch to 2D circuit converter")
#    root.minsize(350, 200)
#    root.config(background='white')
#    
#    fontText = font.Font(size=15, weight='bold')
#    
#    menu = Menu(root)
#    root.config(menu=menu)
#    helpmenu = Menu(menu) 
#    menu.add_cascade(label='Help', menu=helpmenu) 
#    helpmenu.add_command(label='About')
#    helpmenu.add_command(label='Help', command=Help)
#    
#    panelIn = None
#    panelDe = None
#    panelRe = None
#    
#    #panelD = None
#    print("done")
#    var1 = tkinter.StringVar()
#    var2 = tkinter.StringVar()
#    inImgLabel = tkinter.StringVar()
#    deImgLabel = tkinter.StringVar()
#    reImgLabel = tkinter.StringVar()
#    var1.set('Logic Circuit Generator')
#    inImgLabel.set('Input Sketch')
#    deImgLabel.set('Components Detected')
#    reImgLabel.set('Reconstructed Ciruit')
#    
#    
#    
#    title = tkinter.Label(root, textvariable = var1, bg='#ffffff')
#    title['font'] = fontText    
#    title.pack(side="top", fill="both", expand="yes", padx="10", pady="10")
#    
#    exp = tkinter.Label(root, textvariable = var2, bg='#ffffff')
#    exp['font'] = fontText
#    exp.pack(side="top", fill="both", expand="yes", padx="0", pady="0")
##    exp.pack_forget()  
#    
#    btn1 = tkinter.Button(root, text="Logical Expression & Expression Simulation", command=logic_expression, bg='#85C1E9', fg ='#ffffff')
#    print("done1")
#    btn1.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
#    print('done_btn1')
#    btn1['font'] = fontText
#    print("done2")
#    btn1_ttp = CreateToolTip(btn1, "Generate Logical Expression for the image")
#    print("tooltip done")
#    
#    btn2 = tkinter.Button(root, text="Reconstruct Circuit", command=reconstructComponent, bg='#85C1E9', fg ='#ffffff')
#    print("done3")
#    btn2.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
#    print('done_btn2')
#    btn2['font'] = fontText
#    print("done4")
#    btn2_ttp = CreateToolTip(btn2, "Convert the circuit to 2D")
#    print("tooltip done")
#    
#    btn3 = tkinter.Button(root, text="Detect Components", command=detectComponent, bg='#85C1E9', fg ='#ffffff')
#    print("done5")
#    btn3.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
#    print('done_btn3')
#    btn3['font'] = fontText
#    print("done6")
#    btn3_ttp = CreateToolTip(btn3, "Press to detect components in the image")
#    print("tooltip done")
#    
#    btn4 = tkinter.Button(root, text="Select Image", command=inputImage, bg='#85C1E9', fg ='#ffffff')
#    print("done7")
#    btn4.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
#    print('done_btn4')
#    btn4['font'] = fontText
#    print("done6")
#    btn4_ttp = CreateToolTip(btn4, "Image selection for processing")
#    print("tooltip done")
#    
#    # kick off the GUI
#    root.mainloop()
##    os.remove('temp.jpg')
#    print("done8")
