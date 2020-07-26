# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:33:51 2020

@author: Chandraprakash Sharm
"""
from PIL import Image, ImageDraw
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pprint as pp


scale_factor = 8
and_width = 333 // scale_factor
and_height = 333 // scale_factor
or_width = 390 // scale_factor
or_height = 396 // scale_factor
not_width = 122 // scale_factor
not_height = 124 // scale_factor
hor_gap = 10

and_input1 = {'x': 0, 'y': 117 // scale_factor}
and_input2 = {'x': 0, 'y': 217 // scale_factor}
and_output = {'x': 333 // scale_factor, 'y': 167 // scale_factor}
or_input1 = {'x': 45 // scale_factor, 'y': 114 // scale_factor}
or_input2 = {'x': 45 // scale_factor, 'y': 274 // scale_factor}
or_output = {'x': 390 // scale_factor, 'y': 198 // scale_factor}
not_input = {'x': 0, 'y': 62 // scale_factor}
not_output = {'x': 122 // scale_factor, 'y': int(62 / scale_factor)}


def partialOverlap(l1, r1, l2, r2): 
    if (r1['x']<r2['x'] and l1['x']<l2['x'] ):
        if l2['x'] <= r1['x'] :
            return True
        elif l2['x'] - r1['x']<=15:
            return True
    else:
        return False
    

def yrange(nl, nb, xl, xb):
#    not_width  = (nb['y']-nl['y'])//2 + nl['y']
    centery_gate = ((xb['y']-xl['y'])//2)+xl['y']
    if nl['y'] < centery_gate and nb['y'] > centery_gate:
        return True
    else:
        return False

def connected(a, b, g, ai, bi):
    if a["label"]=="wire" and b["label"]=="wire" and (a['bottomright'] == b['topleft']) or (a['topleft']==b['bottomright']):
        return 4
    elif (a['label'] != 'wire' and  b["label"]=="wire") and  abs(a['bottomright']['x']-b['topleft']['x'])<=hor_gap:
        return 3
    elif a['label'] == 'wire' and  b['label']!='wire' and abs(a['bottomright']['x']-b['topleft']['x'])<=hor_gap:
        lt = g.edges(bi, data="weight")
#        print(str())
#        print(lt)
        cnt = 0
        for l in lt:
            if (l[-1]==1):
                cnt+=1
        if (cnt%2 == 1):
            return 2
        else:
            return 1
#    elif a['label'] == 'not' and b['label'] != 'wire' and partialOverlap(a['topleft'], a['bottomright'], b['topleft'], b['bottomright']) and yrange(b['topleft'], b['bottomright'], a['topleft'], a['bottomright']):
##    not gate to input
#        lt = g.edges(b, data="weight"   )
#        cnt = 0
#        for l in lt:
#            if (l[-1]==6):
#                cnt+=1
#        if (cnt == 1):
#            return 7
#        else:
#            return 6
    elif b['label'] == 'not' and a['label'] != 'wire' and partialOverlap(a['topleft'], a['bottomright'], b['topleft'], b['bottomright']) :
#        not gate to output
        return 5
    else:
        return 0

def construct_graph(components, G):  
    i = 0
    components = sorted(components, key=lambda i: (i['topleft']['y'], i['topleft']['x']))
#    print(components)
#    img = cv2.imread('temp.jpg')
    for c in components:
        G.add_node(i,label=c['label'], topleft=c['topleft'], bottomright=c['bottomright'], tlx = 0, tly = 0, brx = 0, bry = 0)
        i = i+1            
        #pin pointing corner points of componenets
#        cv2.circle(img, (c['topleft']['x'], c['topleft']['y']), 5, (0, 255, 0), -1)
#        cv2.circle(img, (c['bottomright']['x'], c['bottomright']['y']), 5, (0, 255, 0), -1)
        
#    fig, ax = plt.subplots(figsize=(15, 15))
#    ax.imshow(img)
    #forming edges
#    print(G.nodes(data = True))
    
    for c1 in components:
        i = components.index(c1)
        for c2 in components:
            j = components.index(c2)
            if c1!=c2:
                weight = int(connected(c1, c2, G, i, j))
#                print(str(weight)+" "+ str(i)+" "+str(j))
                if (weight==0) is False:
                    if (G.has_edge(j, i) or G.has_edge(j, i))==False:
                        G.add_edge(i, j, weight=weight)
    
    Nodes = G.nodes(data=True)
    for n in Nodes:
        comp = n[1]
        if (comp['label']=="and"):
            x = G.node[n[0]]['topleft']['x']
            y = G.node[n[0]]['topleft']['y']
            
            G.node[n[0]]['tlx'] =x
            G.node[n[0]]['tly'] =y
            
            G.node[n[0]]['brx'] =x+and_width
            G.node[n[0]]['bry'] =y+and_height
            print("and tlx:"+str(G.node[n[0]]['tlx'])+" tly:"+str(G.node[n[0]]['tly'])+" brx:"+ str(G.node[n[0]]['brx'])+" bry: "+str(G.node[n[0]]['bry']))
                    
        elif (comp['label']=="or"):
#            print(comp['topleft'])
            x = G.node[n[0]]['topleft']['x']
            y = G.node[n[0]]['topleft']['y']
            
            G.node[n[0]]['tlx'] =x
            G.node[n[0]]['tly'] =y
            
            G.node[n[0]]['brx'] =x+or_width
            G.node[n[0]]['bry'] =y+or_height
#            print(G.node[n[0]])
            print("or tlx:"+str(G.node[n[0]]['tlx'])+" tly:"+str(G.node[n[0]]['tly'])+" brx:"+ str(G.node[n[0]]['brx'])+" bry: "+str(G.node[n[0]]['bry']))
            
        else:
            edgs = G.edges(n[0], data="weight")
            for edg in edgs:
                weight = edg[-1]
                dest = edg[1]
                if(weight == 1) or weight==2:
#                    print("processing input wire")
                    x = G.node[n[0]]['bottomright']['x']                    
                    if(G.node[dest]['label']=='and'):
                        y = G.node[edg[1]]['topleft']['y'] + and_input1['y'] + (100//scale_factor)*((weight+1)%2)
                        G.node[n[0]]['bry'] = y
                        G.node[n[0]]['brx'] = x + and_input1['x']                        
                    elif(G.node[dest]['label']=='or'):
                        y = G.node[edg[1]]['topleft']['y'] + or_input1['y'] + (160//scale_factor)*((weight+1)%2)
                        G.node[n[0]]['bry'] = y
                        G.node[n[0]]['brx'] = x + or_input1['x']
#                    elif(G.node[dest]['label']=='not'):
#                        y = G.node[edg[1]]['topleft']['y'] + not_input['y'] 
#                        G.node[n[0]]['bry'] = y
#                        G.node[n[0]]['brx'] = x
                        print("wires "+str(G.node[n[0]]))
                    G.node[n[0]]['tly'] = G.node[n[0]]['bry'] 
                
                elif (weight == 5):
                    if(G.node[dest]['label']=='and'):
                        x = G.node[dest]['topleft']['x'] + and_width
                        y = G.node[dest]['topleft']['y'] + and_output['y']
                        G.node[n[0]]['tlx'] = x 
                        G.node[n[0]]['tly'] = y - not_output['y']
                        G.node[n[0]]['brx'] = x + not_width
#                        G.node[n[0]]['bry'] = G.node[n[0]]['tly'] + not_height
                        G.node[n[0]]['bry'] = y - not_output['y'] + not_height
                        print("Not with "+str(G.node[dest]['label'])+"\nG.node[n[0]] :" +str(G.node[n[0]]) )                  
#                        print('output'+str(G.node[n[0]]['bry'])+' '+str(n[0]))
                    if(G.node[dest]['label']=='or'):
                        x = G.node[dest]['topleft']['x'] + or_width
                        y = G.node[dest]['topleft']['y'] + or_output['y']
                        G.node[n[0]]['tlx'] = x 
                        G.node[n[0]]['tly'] = y - not_output['y']
                        G.node[n[0]]['brx'] = x + not_width
                        G.node[n[0]]['bry'] = G.node[n[0]]['tly'] + not_height
                
                elif(weight == 3):
#                    print("processing output wire")
                                
                    if(G.node[dest]['label']=='and'):
                        x = G.node[dest]['topleft']['x'] + and_width
                        y = G.node[dest]['topleft']['y'] + and_output['y']
                        G.node[n[0]]['bry'] = y
                        G.node[n[0]]['tlx'] = x
                        G.node[n[0]]['tly'] = y
#                        print('output'+str(G.node[n[0]]['bry'])+' '+str(n[0]))
                                                                
                        
                    if(G.node[dest]['label']=='or'):
                        x = G.node[dest]['topleft']['x'] + or_width
                        y = G.node[dest]['topleft']['y'] + or_output['y']
                        G.node[n[0]]['bry'] = y
                        G.node[n[0]]['tlx'] = x
                        G.node[n[0]]['tly'] = y
#                        print("output wire "+str(G.node[n[0]]))
                        
                    if(G.node[dest]['label']=='not'):
                        x = G.node[dest]['brx']
                        y = G.node[dest]['tly'] + not_output['y']
                        G.node[n[0]]['bry'] = y
                        G.node[n[0]]['tlx'] = x
                        G.node[n[0]]['tly'] = y
        print("\nFixing gates and input and output wires")
        print("-----------------------------------------------------------------------------")
        print(Nodes)
        print("-----------------------------------------------------------------------------\n")
                        
    InternalWire = G.nodes(data=True)
    for n in InternalWire :
        att = n[1]
        if(att['label'] == 'wire'):
            edgs = G.edges(n[0], data=True)
#            print(edgs)
            for e in edgs:
                weight = e[-1]['weight']
                dest = e[1]
                if weight==4:
#                   
                    if (G.node[n[0]]['bottomright'] == G.node[dest]['topleft']):
                        src = n[0]
                        sink = dest
#                        print("src" + str(src))
#                        print("sink" + str(sink))
                    elif (G.node[n[0]]['topleft']==G.node[dest]['bottomright']):
                        src = dest
                        sink = n[0]
#                        print("src" + str(src))
#                        print("sink" + str(sink))
                   
                    if (G.node[src]['brx']==0 and G.node[sink]['tlx']==0):
                        G.node[src]['brx'] = G.node[src]['bottomright']['x']
                        G.node[sink]['tlx'] = G.node[sink]['topleft']['x']
                    elif (G.node[src]['brx']!=0 and G.node[sink]['tlx']==0):
                        G.node[sink]['tlx'] = G.node[src]['brx'] 
                    elif (G.node[src]['brx']==0 and G.node[sink]['tlx']!=0):
                        G.node[src]['brx'] = G.node[sink]['tlx'] 
                    if (G.node[src]['bry']==0 and G.node[sink]['tly']==0):
                        G.node[src]['bry'] = G.node[src]['bottomright']['y']
                        G.node[sink]['tly'] = G.node[sink]['topleft']['y']
                    elif (G.node[src]['bry']!=0 and G.node[sink]['tly']==0):
                        G.node[sink]['tly'] = G.node[src]['bry'] 
                    elif (G.node[src]['bry']==0 and G.node[sink]['tly']!=0):
                        G.node[src]['bry'] = G.node[sink]['tly']
                    
#                   print(str(n[0])+str(att['topleft'])+str(att['bottomright']))
#                    print(str(G.node[n[0]]['tlx'])+","+str(G.node[n[0]]['tly'])+" "+str(G.node[n[0]]['brx'])+","+str(G.node[n[0]]['bry']))
#                    print(str(dest) +str(G.node[dest]['topleft'])+str(G.node[dest]['bottomright']))
#                    print(str(G.node[dest]['tlx'])+","+str(G.node[dest]['tly'])+" "+str(G.node[dest]['brx'])+","+str(G.node[dest]['bry']))

        if G.node[n[0]]['brx']==0:
            G.node[n[0]]['brx'] = G.node[n[0]]['bottomright']['x']
        if G.node[n[0]]['bry']==0:
            G.node[n[0]]['bry'] = G.node[n[0]]['bottomright']['y']
        if G.node[n[0]]['tlx']==0:
            G.node[n[0]]['tlx'] = G.node[n[0]]['topleft']['x']
        if G.node[n[0]]['tly']==0:
            G.node[n[0]]['tly'] = G.node[n[0]]['topleft']['y']
    print("\n-------------------------------------------------------------------")
    print("nodes")
    print(G.nodes(data = True))
    print("---------------------------------------------------------------------\n")
    print("edges")
    print(G.edges(data = True))
    print("---------------------------------------------------------------------\n")

def not_gate(x, y, bg):
    img = Image.open('NOT_gate.png', 'r')
    img_w, img_h = img.size
    img = img.resize((img_w//scale_factor, img_h//scale_factor))
    offset = (x, y)
    bg.paste(img, offset)
    
def or_gate(x, y, bg):
    img = Image.open('Symbol-OR-Gate.png', 'r')
    img_w, img_h = img.size
    img = img.resize((img_w//scale_factor, img_h//scale_factor))
    offset = (x, y)
    bg.paste(img, offset)

def and_gate(x, y, bg):
    img = Image.open('AND-gate-Symbol.png', 'r')
    img_w, img_h = img.size
    img = img.resize((img_w//scale_factor, img_h//scale_factor))
    offset = (x, y)
    bg.paste(img, offset)

def wires(lx, ly, rx, ry, bg):
    cood = [(lx, ly), (rx, ry)]
    bg1 = ImageDraw.Draw(bg)
    bg1.line(cood, fill ="black", width = 2)
    return bg

def reconstruct(components, size):
    G = nx.Graph()
    construct_graph(components, G)
#    nx.draw(G)
    background = Image.new('RGB', (size[0], size[1]), (255, 255, 255))
    circuit_elements = G.nodes(data = True)
    for r in circuit_elements:
        r = r[1]
        if r['label']=='not':
            not_gate(r['tlx'], r['tly'], background)
        elif r['label'] == 'or':
            or_gate(r['tlx'], r['tly'], background)
        elif r['label'] == 'and':
            and_gate(r['tlx'], r['tly'], background)
        elif r['label'] == 'wire':
            bg = wires(r['tlx'], r['tly'], r['brx'], r['bry'], background)
            
    background.resize((size[1], size[0]))
    background.save('out.jpg')  
#    img = cv2.imread('out.jpg')
#    fig, ax = plt.subplots(figsize=(15, 15))
#    ax.imshow(img)         


#img = cv2.imread('temp.jpg')
#fig, ax = plt.subplots(figsize=(15, 15))
#ax.imshow(img)

#components = [{'label': 'and', 'confidence': 0.8446919, 'topleft': {'x': 198, 'y': 92}, 'bottomright': {'x': 377, 'y': 308}}, {'label': 'or', 'confidence': 0.84445304, 'topleft': {'x': 490, 'y': 219}, 'bottomright': {'x': 662, 'y': 367}}, {'label': 'not', 'confidence': 0.8945223, 'topleft': {'x': 302, 'y': 164}, 'bottomright': {'x': 388, 'y': 234}}, {'label': 'wire', 'topleft': {'x': 70, 'y': 307}, 'bottomright': {'x': 301, 'y': 307}}, {'label': 'wire', 'topleft': {'x': 387, 'y': 190}, 'bottomright': {'x': 412, 'y': 190}}, {'label': 'wire', 'topleft': {'x': 412, 'y': 190}, 'bottomright': {'x': 412, 'y': 272}}, {'label': 'wire', 'topleft': {'x': 412, 'y': 272}, 'bottomright': {'x': 491, 'y': 272}}, {'label': 'wire', 'topleft': {'x': 491, 'y': 272}, 'bottomright': {'x': 491, 'y': 278}}]
#reconstruct(components, (800, 400))
#components =   [{'label': 'and', 'confidence': 0.79810315, 'topleft': {'x': 1485, 'y': 244}, 'bottomright': {'x': 2348, 'y': 1256}}, {'label': 'or', 'confidence': 0.76538694, 'topleft': {'x': 2912, 'y': 849}, 'bottomright': {'x': 3914, 'y': 1664}}, {'label': 'not', 'confidence': 0.73433167, 'topleft': {'x': 3917, 'y': 1143}, 'bottomright': {'x': 4128, 'y': 1360}},
#       (4600, 2176)         {'label':'wire', 'topleft':{'x': 521, 'y':673}, 'bottomright':{'x':1485, 'y':625}},{'label':'wire', 'topleft':{'x': 533, 'y':937}, 'bottomright':{'x':1485, 'y':857}},{'label':'wire', 'topleft':{'x': 2348, 'y':697}, 'bottomright':{'x':2529, 'y':677}},{'label':'wire', 'topleft':{'x': 2529, 'y':677}, 'bottomright':{'x':2529, 'y':1189}},{'label':'wire', 'topleft':{'x': 2529, 'y':1189}, 'bottomright':{'x':2912, 'y':1141}},{'label':'wire', 'topleft':{'x': 481, 'y':1529}, 'bottomright':{'x':2912, 'y':1353}},{'label':'wire', 'topleft':{'x': 4128, 'y':1189}, 'bottomright':{'x':4597, 'y':1169}}]      
#
#components =[{'label': 'and', 'confidence': 0.84423566, 'topleft': {'x': 190, 'y': 100}, 'bottomright': {'x': 356, 'y': 304}}, {'label': 'wire', 'topleft': {'x': 64.0, 'y': 157.0}, 'bottomright': {'x': 189.0, 'y': 157.0}}, {'label': 'wire', 'topleft': {'x': 356.0, 'y': 186.0}, 'bottomright': {'x': 478.0, 'y': 186.0}}, {'label': 'wire', 'topleft': {'x': 64.0, 'y': 251.0}, 'bottomright': {'x': 189.0, 'y': 251.0}}]
#reconstruct(components, (800, 400))
#img = cv2.imread('out.jpg')
#fig, ax = plt.subplots(figsize=(15, 15))
#ax.imshow(img)

