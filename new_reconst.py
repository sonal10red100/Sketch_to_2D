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
#hor_gap = 0

and_input1 = {'x': 0, 'y': 117 // scale_factor}
and_input2 = {'x': 0, 'y': 217 // scale_factor}
and_output = {'x': 333 // scale_factor, 'y': 167 // scale_factor}
or_input1 = {'x': 45 // scale_factor, 'y': 114 // scale_factor}
or_input2 = {'x': 45 // scale_factor, 'y': 274 // scale_factor}
or_output = {'x': 390 // scale_factor, 'y': 198 // scale_factor}
not_input = {'x': 0, 'y': 62 // scale_factor}
not_output = {'x': 122 // scale_factor, 'y': int(62 / scale_factor)}

G = nx.Graph()

def partialOverlap(l1, r1, l2, r2): 
    if (r1['x']<r2['x'] and l1['x']<l2['x'] ):
        if l2['x'] <= r1['x'] :
            return True
    elif abs(l2['x'] - r1['x'])<=100:
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

def connected(a, b, ai, bi):
    if a["label"]=="wire" and b["label"]=="wire" and (a['bottomright'] == b['topleft']) or (a['topleft']==b['bottomright']):
        edgs = G.edges(ai)
        if len(edgs)==2:
            return 0
        return 4
    elif (a['label'] != 'wire' and  b["label"]=="wire") and  abs(a['bottomright']['x']-b['topleft']['x'])<=1:
        edgs = G.edges(ai, data=True)
        for e in edgs:
            if (e[-1]['weight'] == 3):
                return 0
        return 3
    elif a['label'] == 'wire' and  b['label']!='wire' and abs(a['bottomright']['x']-b['topleft']['x'])<=1:
        if(b['label']=='not'):
            return 0
        lt = G.edges(bi, data="weight")
#        print(str())
#        print(lt)
#        if (len(lt)>=3):
#            return 0
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

def construct_graph(components):  
    i = 0
    gate_nodes = []
    components = sorted(components, key=lambda i: (i['topleft']['y'], i['topleft']['x']))
#    img = cv2.imread('temp.jpg')
#    img = cv2.imread('result/app/input/IMG_20200113_222400.jpg')
#    img = cv2.resize(img, (800, 400), interpolation = cv2.INTER_AREA)
    for c in components:
        if(c['label']=='and' or c['label']=='or' or c['label']=='not'):
            gate_nodes.append(i)
        G.add_node(i,label=c['label'], topleft=c['topleft'], bottomright=c['bottomright'], tlx = 0, tly = 0, brx = 0, bry = 0)
        i = i+1   
#        if (c['label']=='wire'):
#            cv2.circle(img, (int(c['topleft']['x']), int(c['topleft']['y'])), 5, (0, 255, 0), -1)
#            cv2.circle(img, (int(c['bottomright']['x']), int(c['bottomright']['y'])), 5, (0, 255, 0), -1)
        
#    cv2.imwrite('result/app/wires_endpoint/ob.jpg', img)    
#    fig, ax = plt.subplots(figsize=(15, 15))
#    ax.imshow(img)
    #forming edges
#    print(G.nodes(data = True))
    
    for c1 in components:
        i = components.index(c1)
        for c2 in components:
            j = components.index(c2)
            if c1!=c2:
                weight = int(connected(c1, c2, i, j))
#                print(str(weight)+" "+ str(i)+" "+str(j))
                if (weight==0) is False:
                    if (G.has_edge(j, i) or G.has_edge(i, j))==False:
                        G.add_edge(i, j, weight=weight)

# add ouput wire in case no output for any gate     
    for g in gate_nodes:
#        print('gate ', g)
        wire = [0, 0, 0]
        not_connect = 0
        edgs = G.edges(g, data = True)
        for e in edgs:
#            print(e)
            try:
                wire[e[-1]['weight']-1] = 1
            except:
                if(G.nodes[e[1]]['label']=='not' or G.nodes[g]['label'] == 'not'):
                    not_connect = 1
#        print("wire ",wire, "not count : ", not_cnt)
        if ( wire[0] == 0 and G.node[g]['label']!='not'):
#            print("Input 1 wire missing")
            i = G.number_of_nodes()
            G.add_node(i, label='wire', topleft={'x': 50, 'y':0}, bottomright={'x': 0, 'y':0}, tlx=0, tly=0, brx=0, bry=0)
            G.add_edge(i, g, weight = 1)
        if ( wire[1] == 0 and G.node[g]['label']!='not'):
#            print("Input 2 wire missing")
            i = G.number_of_nodes()
            G.add_node(i, label='wire', topleft={'x': 50, 'y':0}, bottomright={'x': 0, 'y':0}, tlx=0, tly=0, brx=0, bry=0)
            G.add_edge(i, g, weight = 2)
        if( wire[2] == 0 ):
#            print("output wire missing")
            if(G.node[g]['label']!='not' and not_connect==0):
                i = G.number_of_nodes()
                G.add_node(i, label='wire', topleft={'x': 0, 'y':0}, bottomright={'x': 750, 'y':0}, tlx=0, tly=0, brx=0, bry=0)
                G.add_edge(i, g, weight = 3)
            elif(G.nodes[g]['label']=='not' and not_connect==1):
                i = G.number_of_nodes()
                G.add_node(i, label='wire', topleft={'x': 0, 'y':0}, bottomright={'x': 750, 'y':0}, tlx=0, tly=0, brx=0, bry=0)
                G.add_edge(i, g, weight = 3)

    print("*********************************************\n")
    edgs = G.edges(data=True)
    for e in edgs:
        print(e)
    print("*********************************************\n")
# Fixing gates in the circuit.    
# gates fixed
    Nodes = G.nodes(data=True)
    for n in Nodes:
        comp = n[1]
        node_index = n[0]
        if (comp['label']=="and"):
            x = G.node[n[0]]['topleft']['x']
            if x<0:
                x = 0
            y = G.node[n[0]]['topleft']['y']
            if y<0:
                y = 0
            G.node[n[0]]['tlx'] =x
            G.node[n[0]]['tly'] =y
            G.node[n[0]]['brx'] =x+and_width
            G.node[n[0]]['bry'] =y+and_height
#            print("and tlx:"+str(G.node[n[0]]['tlx'])+" tly:"+str(G.node[n[0]]['tly'])+" brx:"+ str(G.node[n[0]]['brx'])+" bry: "+str(G.node[n[0]]['bry']))
        elif (comp['label']=="or"):
#            print(comp['topleft'])
            x = G.node[n[0]]['topleft']['x']
            if x<0:
                x = 0
            y = G.node[n[0]]['topleft']['y']
            if y<0:
                y = 0
            G.node[n[0]]['tlx'] =x
            G.node[n[0]]['tly'] =y
            G.node[n[0]]['brx'] =x+or_width
            G.node[n[0]]['bry'] =y+or_height
#            print(G.node[n[0]])
#            print("or tlx:"+str(G.node[n[0]]['tlx'])+" tly:"+str(G.node[n[0]]['tly'])+" brx:"+ str(G.node[n[0]]['brx'])+" bry: "+str(G.node[n[0]]['bry']))    
        elif (comp['label']=='not'):
            edgs = G.edges(node_index, data="weight")
#            print(edgs)
            for e in edgs:
                dest_node = e[1]
                weight = e[-1]
                if(weight == 5):
                    if(G.node[dest_node]['label']=='and'):
                        x = G.node[dest_node]['topleft']['x'] + and_width
                        y = G.node[dest_node]['topleft']['y'] + and_output['y']
                        G.node[n[0]]['tlx'] = x 
                        G.node[n[0]]['tly'] = y - not_output['y']
                        G.node[n[0]]['brx'] = x + not_width
#                        G.node[n[0]]['bry'] = G.node[n[0]]['tly'] + not_height
                        G.node[n[0]]['bry'] = y - not_output['y'] + not_height
#                        print("Not with "+str(G.node[dest_node]['label'])+"\nG.node[n[0]] :" +str(G.node[n[0]]) )                  
#                        print('output'+str(G.node[n[0]]['bry'])+' '+str(n[0]))
                    if(G.node[dest_node]['label']=='or'):
                        x = G.node[dest_node]['topleft']['x'] + or_width
                        y = G.node[dest_node]['topleft']['y'] + or_output['y']
                        G.node[n[0]]['tlx'] = x 
                        G.node[n[0]]['tly'] = y - not_output['y']
                        G.node[n[0]]['brx'] = x + not_width
                        G.node[n[0]]['bry'] = G.node[n[0]]['tly'] + not_height

# Processing output and input wires
    for w in Nodes:
        index = w[0]
        comp = w[1]
        if(comp['label'] == 'wire'):
            edgs = G.edges(index, data=True)
            for e in edgs:
                wire_node = e[0]
                gate_node = e[1]
                weight = e[-1]['weight']
                gate = G.node[gate_node]['label']
                x = G.node[gate_node]['topleft']['x']
                y = G.node[gate_node]['topleft']['y']
                
                if(weight == 1 or weight == 2):
                    if(gate == 'and'):
                        G.node[wire_node]['brx'] = x 
                        G.node[wire_node]['bry'] = y + and_input1['y'] + (100//scale_factor)*((weight+1)%2)
                        G.node[wire_node]['tly'] = y + and_input1['y'] + (100//scale_factor)*((weight+1)%2)
#                        print("and tlx:"+str(G.node[index]['tlx'])+" tly:"+str(G.node[index]['tly'])+" brx:"+ str(G.node[index]['brx'])+" bry: "+str(G.node[index]['bry']))
                    if(gate == 'or'):
                        G.node[wire_node]['brx'] = x 
                        G.node[wire_node]['bry'] = y + or_input1['y'] + (160//scale_factor)*((weight+1)%2)
                        G.node[wire_node]['tly'] = y + or_input1['y'] + (160//scale_factor)*((weight+1)%2)
                elif(weight == 3):
##                    print("processing output wire")   
                    if(gate=='and'):
                        G.node[wire_node]['bry'] = y + and_output['y']
                        G.node[wire_node]['tlx'] = x + and_width
                        G.node[wire_node]['tly'] = y + and_output['y']
#                        print('output'+str(G.node[n[0]]['bry'])+' '+str(n[0]))
                                                                
                    elif(gate=='or'):
                        G.node[wire_node]['bry'] = y + or_output['y']
                        G.node[wire_node]['tlx'] = x + or_width
                        G.node[wire_node]['tly'] = y + or_output['y']
#                        print("output wire "+str(G.node[n[0]]))
                        
                    elif(gate=='not'):
                        x = G.node[gate_node]['tlx']
                        y = G.node[gate_node]['tly']
                        G.node[wire_node]['bry'] = y + not_output['y']
                        G.node[wire_node]['tlx'] = x + not_width
                        G.node[wire_node]['tly'] = y + not_output['y']
#                print("-----------------------------------------------------")
#                print(G.node[wire_node])
#                print(G.node[gate_node])
#                print("----------------------------------------------------\n")
                  
# Processing internal wires
    for n in Nodes :
        att = n[1]
        if(att['label'] == 'wire'):
            edgs = G.edges(n[0], data=True)
#            print(edgs)
            for e in edgs:
                weight = e[-1]['weight']
                dest = e[1]
                if weight==4:                   
                    if (G.node[n[0]]['bottomright'] == G.node[dest]['topleft']):
                        src = n[0]
                        sink = dest
                    elif (G.node[n[0]]['topleft'] == G.node[dest]['bottomright']):
                        src = dest
                        sink = n[0]
                    else:
                       print('Satisfies none')
                       break
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
    for n in Nodes:
        if G.node[n[0]]['brx']==0:
            G.node[n[0]]['brx'] = G.node[n[0]]['bottomright']['x']
        if G.node[n[0]]['bry']==0:
            G.node[n[0]]['bry'] = G.node[n[0]]['bottomright']['y']
        if G.node[n[0]]['tlx']==0:
            G.node[n[0]]['tlx'] = G.node[n[0]]['topleft']['x']
        if G.node[n[0]]['tly']==0:
            G.node[n[0]]['tly'] = G.node[n[0]]['topleft']['y']
    
# removing extra lines being created 
# in case it isnt identified. 
    node_to_remove = []
    for n in Nodes:
        try:
            i = gate_nodes.index(n[0])
        except:
            flag = 0
            for g in gate_nodes:
                for path in nx.all_simple_paths(G, n[0], g):
                    flag = 1
                    break
                if(flag == 1):
                    break
            if (flag == 0):
                node_to_remove.append(n[0])
    for n in node_to_remove:
        G.remove_node(n)
    gate_nodes.clear()
    
##    print("edges")
##    print(G.edges(data = True))
##    print("---------------------------------------------------------------------\n")

def improve_graph():
    print('imporve')
    wires = []
    gates = []
    Nodes = G.nodes(data=True)
    for n in Nodes:
        if n[1]['label']=='wire':
            wires.append(n[0])
        elif n[1]['label']=='and' or n[1]['label']=='or' or n[1]['label'] == 'not':
            gates.append(n[0])
    no_of_wires = len(wires)
    print('Gates : ', gates)
    print('Wires : ', wires)
    for w in wires:
        for g in gates:
#            shortest path from wire to gate
            path_exist = False
            try:
                path = nx.shortest_path(G, w, g)[:-1]
                path_exist = True
            except:
                continue
            
            if(len(path)>=2):
#                discard path with gate in between
                found_gate = False
                for node in path:
                    for gate in gates:
                        if(node == gate):
                            found_gate = True
                            break
                    if found_gate == True:
                        break
                    
                if (found_gate == False):
                    start_node = path[0]
                    end_node = path[-1]
                    
                    degree_start_node = len(G.edges(start_node))
                    degree_end_node = len(G.edges(end_node))
                    
                    weight_start = 0
                    weight_end = 0
                    for ed in G.edges(end_node, data=True):
                        weight_end += ed[-1]['weight']
                    for ed in G.edges(start_node, data=True):
                        weight_start += ed[-1]['weight']
                    print('path', w, g, path)
#               process internal connectiopn wires                                    
#                    if(len(path)>3 ):
                    if (weight_start == 7 and weight_end in range(5, 7) and len(path)>3):
                        current_node = path[1]
                        G.node[current_node]['bry'] = G.node[end_node]['tly']
                        G.node[end_node]['tlx'] = G.node[start_node]['brx']
                        for i in range(1, len(path)-1):
                            G.remove_edge(path[i], path[i+1])
                            print('loop')
                        G.add_edge(current_node, end_node, weight=4)
                        print(w, g, path)
#                    process input or output wires
                    
                    elif ( weight_end in range(5, 7) and weight_start == 4 ):
                        print('input wire condition')
                        if G.node[end_node]['tlx'] != G.node[start_node]['tlx']:
                            G.node[end_node]['tlx'] = G.node[start_node]['tlx']
                        for i in range(0, len(path)-1):
                            G.remove_edge(path[i], path[i+1])
                            print('loop')
#                            G.remove_edge(start_node, end_node)
                    elif( weight_end == 7 and weight_start == 4 ):
                        print('output wire condition')
                        if G.node[start_node]['brx'] != G.node[end_node]['brx']:
                            G.node[start_node]['brx'] = G.node[end_node]['brx']
                        for i in range(0, len(path)-1):
                            G.remove_edge(path[i], path[i+1])
                            print('loop')
#                            G.remove_edge(start_node, end_node)
                    
#                    print('current_node', current_node, G.node[current_node]['tlx'], G.node[current_node]['tly'], G.node[current_node]['brx'], G.node[current_node]['bry'])
#                    print('end_node', end_node, G.node[end_node]['tlx'], G.node[end_node]['tly'], G.node[end_node]['brx'], G.node[end_node]['bry'])
#           
    node_to_remove = []
    for n in Nodes:
        try:
            i = gates.index(n[0])
        except:
            flag = 0
            for g in gates:
                for path in nx.all_simple_paths(G, n[0], g):
                    flag = 1
                    break
                if(flag == 1):
                    break
            if (flag == 0):
                node_to_remove.append(n[0])
    for n in node_to_remove:
        G.remove_node(n)
    
    
    
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
    
    construct_graph(components)
#    nx.draw(G)
    improve_graph()
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
    components.clear()
    for n in G.nodes(data=True):
        components.append(n[1])
    print("reconstructed!!")
    
#    print("\n-------------------------------------------------------------------")
#    print("nodes")
#    for n in G.nodes(data=True):
#        print(str(n[0]) + " "+str(n[1]['label'])+
#              " "+
#              str(n[1]['tlx'])+" "+ 
#              str(n[1]['tly'])+" "+
#              str(n[1]['brx'])
#              +" "+ str(n[1]['tly']))
#    print("---------------------------------------------------------------------\n")
#    
    G.clear()
    
#    img = cv2.imread('out.jpg')
#    fig, ax = plt.subplots(figsize=(15, 15))
#    ax.imshow(img)         
    
    return components

#
#components = [{'label': 'and', 'confidence': 0.8526694, 'topleft': {'x': 235, 'y': 29}, 'bottomright': {'x': 420, 'y': 244}}, {'label': 'or', 'confidence': 0.7925187, 'topleft': {'x': 499, 'y': 154}, 'bottomright': {'x': 663, 'y': 310}}, {'label': 'not', 'confidence': 0.89816254, 'topleft': {'x': 668, 'y': 198}, 'bottomright': {'x': 736, 'y': 258}}, {'label': 'wire', 'topleft': {'x': 82, 'y': 247}, 'bottomright': {'x': 84.0, 'y': 247}}, {'label': 'wire', 'topleft': {'x': 84.0, 'y': 247}, 'bottomright': {'x': 84.0, 'y': 276.0}}, {'label': 'wire', 'topleft': {'x': 84.0, 'y': 276.0}, 'bottomright': {'x': 500, 'y': 276.0}}, {'label': 'wire', 'topleft': {'x': 500, 'y': 276.0}, 'bottomright': {'x': 500, 'y': 260.0}}, {'label': 'wire', 'topleft': {'x': 736, 'y': 212}, 'bottomright': {'x': 800, 'y': 212}}, {'label': 'wire', 'topleft': {'x': 90, 'y': 157}, 'bottomright': {'x': 229.0, 'y': 157}}, {'label': 'wire', 'topleft': {'x': 229.0, 'y': 157}, 'bottomright': {'x': 229.0, 'y': 173.0}}, {'label': 'wire', 'topleft': {'x': 229.0, 'y': 173.0}, 'bottomright': {'x': 236, 'y': 173.0}}, {'label': 'wire', 'topleft': {'x': 420, 'y': 121}, 'bottomright': {'x': 434.0, 'y': 121}}, {'label': 'wire', 'topleft': {'x': 434.0, 'y': 121}, 'bottomright': {'x': 434.0, 'y': 213.0}}, {'label': 'wire', 'topleft': {'x': 434.0, 'y': 213.0}, 'bottomright': {'x': 468.0, 'y': 213.0}}, {'label': 'wire', 'topleft': {'x': 468.0, 'y': 213.0}, 'bottomright': {'x': 468.0, 'y': 174.0}}, {'label': 'wire', 'topleft': {'x': 468.0, 'y': 174.0}, 'bottomright': {'x': 500, 'y': 174.0}}, {'label': 'wire', 'topleft': {'x': 89, 'y': 113}, 'bottomright': {'x': 236, 'y': 113}}]
##components = [{'label': 'and', 'confidence': 0.7562369, 'topleft': {'x': 214, 'y': 38}, 'bottomright': {'x': 360, 'y': 232}}, {'label': 'and', 'confidence': 0.8221032, 'topleft': {'x': 523, 'y': 159}, 'bottomright': {'x': 668, 'y': 369}}, {'label': 'wire', 'topleft': {'x': 667, 'y': 243.0}, 'bottomright': {'x': 667, 'y': 243.0}}, {'label': 'wire', 'topleft': {'x': 667, 'y': 243.0}, 'bottomright': {'x': 667, 'y': 286.0}}, {'label': 'wire', 'topleft': {'x': 667, 'y': 286.0}, 'bottomright': {'x': 708.0, 'y': 286.0}}, {'label': 'wire', 'topleft': {'x': 708.0, 'y': 286.0}, 'bottomright': {'x': 708.0, 'y': 263.0}}, {'label': 'wire', 'topleft': {'x': 708.0, 'y': 263.0}, 'bottomright': {'x': 719, 'y': 263.0}}, {'label': 'wire', 'topleft': {'x': 92, 'y': 171}, 'bottomright': {'x': 136, 'y': 171}}, {'label': 'wire', 'topleft': {'x': 138, 'y': 161}, 'bottomright': {'x': 215, 'y': 161}}, {'label': 'wire', 'topleft': {'x': 359, 'y': 107}, 'bottomright': {'x': 413.0, 'y': 107}}, {'label': 'wire', 'topleft': {'x': 413.0, 'y': 107}, 'bottomright': {'x': 413.0, 'y': 237.0}}, {'label': 'wire', 'topleft': {'x': 413.0, 'y': 237.0}, 'bottomright': {'x': 524, 'y': 237.0}}, {'label': 'wire', 'topleft': {'x': 57, 'y': 94}, 'bottomright': {'x': 215, 'y': 94}}]
#comp = reconstruct(components, (800, 400))
#img = cv2.imread('out.jpg')
#fig, ax = plt.subplots(figsize=(15, 15))
#ax.imshow(img)         
