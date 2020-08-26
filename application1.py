# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:34:38 2020

@author: HP
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:16:33 2020

@author: Chandraprakash Sharm
"""
import subprocess
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
import tensorflow as tf

from PIL import ImageDraw
import matplotlib.pyplot as plt
#from darkflow.net.build import TFNet
import pprint as pp

import speech_recognition as sr 
import pyttsx3  
import time
  
# Initialize the recognizer  
r = sr.Recognizer() 

from utils import visualization_utils as vis_util
from utils import label_map_util

from tokToolTip import CreateToolTip
from simulation_table import Table

from new import wire_detection
from final_bool import gen_expression
from new_reconst import reconstruct
from table import gen_truth_table
from trial import gen_logisim
dimensions = (800, 400)

flag=0

MODEL_NAME='C:/Users/HP/Desktop/sonal/Custom-Faster-RCNN-Using-Tensorfow-Object-Detection-API-master/dataset/inf_graph3'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#Path to label.pbtxt
PATH_TO_LABELS='C:/Users/HP/Desktop/sonal/Custom-Faster-RCNN-Using-Tensorfow-Object-Detection-API-master/dataset/label.pbtxt'

category_index = {1: {'id': 1, 'name': 'and'}, 2: {'id': 2, 'name': 'not'}, 3:{'id': 3, 'name': 'or'}} #label_map_util.create_category_index_from_labelmap('C://Programs//Anaconda//Tensorflow//models//research//object_detection//training//objectdetection.pbtxt', use_display_name=True)



detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

def Overlap(l1, r1, l2, r2): 
    # If one rectangle is on left side of other 
    if(l1['x'] > r2['x'] or l2['x'] > r1['x']): 
        return False
    if(l1['y'] > r2['y'] or l2['y'] > r1['y']): 
        return False
    return True

def detection(inputImg, image_path):
    
    global predictions
    
    print(inputImg.size)       
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)

    coordinates = vis_util.return_coordinates(
                        image_np,
                        np.squeeze(output_dict['detection_boxes']),
                        np.squeeze(output_dict['detection_classes']).astype(np.int32),
                        np.squeeze(output_dict['detection_scores']),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.30)
  
    a=os.path.basename(image_path)
    #b=os.path.splitext(a)[0]
  
    print(a,':',coordinates)
    newImage = np.copy(inputImg)
    predictions=[]
    for i in range(len(coordinates)):
          label=coordinates[i][5]
          #confidence=coordinates[i][4]
          top_x=coordinates[i][2]
          top_y=coordinates[i][0]
          btm_x=coordinates[i][3]
          btm_y=coordinates[i][1]
          
          
          newImage = cv2.rectangle(newImage, (top_x-5, top_y-5), (btm_x+5, btm_y+5), (255,0,0), 1)
          newImage = cv2.putText(newImage, label, (top_x, top_y+20), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (0, 0, 0), 1, cv2.LINE_AA)
          
          predictions.append({'label':label,'topleft':{'x':top_x, 'y':top_y},'bottomright':{'x':btm_x, 'y':btm_y}})
    cv2.imwrite('newImage.jpg',newImage)
    
    return (newImage,predictions)      



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

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


#Input image selection
def inputImage():
    global panelIn
    global path
    global inImgLabel
    global inp_img_path

    path=tkinter.filedialog.askopenfilename()
    if len(path)>0:
        #reading the image and pre-processing it for display
        image=cv2.imread(path)
        image=cv2.resize(image, dimensions, interpolation = cv2.INTER_AREA)
        cv2.imwrite('temp.jpg', image)
        inp_img_path=path
        print('-------'+path)
        
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
            print('image selected')
            panelIn.configure(image=image)
            panelIn.image=image

        
#processing started and detected gate display
def detectComponent():
    global panelDe
    global path
    global deImgLabel
    
#    path=tkinter.filedialog.askopenfilename()
    image=cv2.imread('temp.jpg')
    if image is not None:
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print('seeeeeeeeeeeeee')
        print(inp_img_path)
        darr=detection(image,inp_img_path)
        oimage=darr[0]
        predictions=darr[1]
        print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
        print(predictions)
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
                print('if+++++++++++')
#                panelDe.pack(side="right", padx=10, pady=10)
                
            else:
                print('else+++++++++++')
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
    global flag
    
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
                flag=1

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
    
def logisim():
    if flag==0:
        print('Circuit not yet reconstructed') 
        tkinter.messagebox.showinfo("Information", "Circuit not yet reconstructed")
    
    else:
        print('open in logisim')
        print(components)
        gen_logisim(components)
        filename='C:/Users/HP/Desktop/sonal/Custom-Faster-RCNN-Using-Tensorfow-Object-Detection-API-master/dataset/app_testing/q1.circ'
        subprocess.Popen(["C:/Users/HP/logisim-win-2.7.1.exe", filename])
    
def SpeakText(command):       
    # Initialize the engine 
    engine = pyttsx3.init() 
    engine.say(command)  
    engine.runAndWait()     
    
def speech_to_text():
    print('YAYYYYYYY')
    MyText=""
    image_selected=0
    components_detected=0
    circuit_reconstructed=0
    log_exp=0
    truth_tbl=0
    while(1):     
          
        # Exception handling to handle 
        # exceptions at the runtime 
        try: 
              
            # use the microphone as source for input. 
            with sr.Microphone() as source2: 
                  
                # wait for a second to let the recognizer 
                # adjust the energy threshold based on 
                # the surrounding noise level  
                r.adjust_for_ambient_noise(source2, duration=0.2) 
                  
                #listens for the user's input  
                audio2 = r.listen(source2) 
                  
                # Using ggogle to recognize audio 
                MyText = r.recognize_google(audio2) 
                MyText = MyText.lower() 
      
                print("Did you say =====> "+MyText)
                if(MyText=="stop"):
                    break
                if(MyText=="select image"):
                    image_selected=1
                    btn4.invoke()
                    break
                elif(MyText=="detect components"):
                    components_detected=1
                    btn3.invoke()
                    break
                elif(MyText=="reconstruct circuit"):
                    circuit_reconstructed=1
                    btn2.invoke()
                    break
                elif(MyText=="logical expression"):
                    log_exp=1
                    btn1.invoke()
                    break
                elif(MyText=="expression calculator"):
                    truth_tbl=1
                    btn0.invoke()
                    break
                else:
                    SpeakText("No such functionality Please try to speak again!")
                    break
                                  
        except sr.RequestError as e: 
            print("Could not request results; {0}".format(e)) 
              
        except sr.UnknownValueError: 
            print("unknown error occured")

    if(image_selected==1):
        tkinter.messagebox.showinfo("Information","Image selected. Proceed...")
        speech_to_text()
    if(components_detected==1):
        tkinter.messagebox.showinfo("Information","Components detected. Proceed...")
        speech_to_text()
    if(circuit_reconstructed==1):
        tkinter.messagebox.showinfo("Information","Circuit reconstructed. Proceed...")
        speech_to_text()
    if(log_exp==1):
        tkinter.messagebox.showinfo("Information","Logical expression generated. Proceed...")
        speech_to_text()
    #if(truth_tbl==1):
    #   tkinter.messagebox.showinfo("Information","Truth table generated. Process completed!")
        
        
        
        
     
                    

        

expression = None
#predictions = None
components = None
path=""
#options = {"model": "cfg/tiny-yolo-voc-3c.cfg",
#          "load": 8625,
#           "gpu": 1.0,
#          "threshold":0.45}
#tfnet2 = TFNet(options)

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
    exp = tkinter.Label(root, textvariable = var2, bg='#ffffff',relief = "groove", borderwidth=5)
    exp['font'] = fontText
  
    
    title = tkinter.Label(root, textvariable = var1, bg='#ffffff')
    title['font'] = fontTitle
    title.place(x = 10, y = 10, width = 920, height = 40)
    title['bg']="yellow"
    
    #Speech to text
    photo = ImageTk.PhotoImage(Image.open("C:/Users/HP/Desktop/sonal/Custom-Faster-RCNN-Using-Tensorfow-Object-Detection-API-master/dataset/app_testing/speaker.png"))
    btn_sp=tkinter.Button(root, text="", command=speech_to_text, image = photo)
    btn_sp.place(x=1000,y=10,width=50,height=40)
    btn_sp= CreateToolTip(btn_sp, "Speak to operate functionalities")
    
    
    photo_logisim = ImageTk.PhotoImage(Image.open("C:/Users/HP/Desktop/sonal/Custom-Faster-RCNN-Using-Tensorfow-Object-Detection-API-master/dataset/app_testing/logisim-icon.png"))
    btn_logisim=tkinter.Button(root, text="Open in Logisim", command=logisim, image = photo_logisim)
    btn_logisim.place(x=940,y=10,width=50,height=40)
    btn_logisim = CreateToolTip(btn_logisim, "Open in logisim")
    
    
    
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


