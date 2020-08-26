# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:15:41 2020

@author: HP
"""

import xml.etree.ElementTree
from xml.etree.ElementTree import Element, SubElement, Comment, tostring, ElementTree


# Open original file
def gen_logisim(components):
    #f= open("q1.circ","w+")
    root = Element('project')
    root.attrib['source']='2.7.1'
    root.attrib['version']='1.0'
    
    lib0=SubElement(root,'lib')
    lib0.attrib['desc']='#Wiring'
    lib0.attrib['name']='0'
    
    lib1=SubElement(root,'lib')
    lib1.attrib['desc']='#Gates'
    lib1.attrib['name']='1'
    
    lib2=SubElement(root,'lib')
    lib2.attrib['desc']='#Plexers'
    lib2.attrib['name']='2'
    
    lib3=SubElement(root,'lib')
    lib3.attrib['desc']='#Arithmetic'
    lib3.attrib['name']='3'
    
    lib4=SubElement(root,'lib')
    lib4.attrib['desc']='#Memory'
    lib4.attrib['name']='4'
    
    lib5=SubElement(root,'lib')
    lib5.attrib['desc']='#I/O'
    lib5.attrib['name']='5'
    
    lib6=SubElement(root,'lib')
    lib6.attrib['desc']='#Base'
    lib6.attrib['name']='6'
    tool=SubElement(lib6,'tool')
    tool.attrib['name']='Text Tool'
    a1=SubElement(tool,'a')
    a1.attrib['name']='text'
    a1.attrib['val']=''
    
    a2=SubElement(tool,'a')
    a2.attrib['name']='font'
    a2.attrib['val']='SansSerif plain 12'
    
    a3=SubElement(tool,'a')
    a3.attrib['name']='halign'
    a3.attrib['val']='center'
    
    a4=SubElement(tool,'a')
    a4.attrib['name']='valign'
    a4.attrib['val']='base'
    
    main=SubElement(root,'main')
    main.attrib['name']='main'
    
    options=SubElement(root,'options')
    aa1=SubElement(options,'a')
    aa1.attrib['name']='gateUndefined'
    aa1.attrib['val']='ignore'
    aa2=SubElement(options,'a')
    aa2.attrib['name']='simlimit'
    aa2.attrib['val']='1000'
    aa3=SubElement(options,'a')
    aa3.attrib['name']='simrand'
    aa3.attrib['val']='0'
    
    mappings=SubElement(root,'mappings')
    tool1=SubElement(mappings,'tool')
    tool1.attrib['lib']='6'
    tool1.attrib['map']='Button2'
    tool1.attrib['name']='Menu Tool'
    tool2=SubElement(mappings,'tool')
    tool2.attrib['lib']='6'
    tool2.attrib['map']='Button3'
    tool2.attrib['name']='Menu Tool'
    tool3=SubElement(mappings,'tool')
    tool3.attrib['lib']='6'
    tool3.attrib['map']='Ctrl Button1'
    tool3.attrib['name']='Menu Tool'
    
    toolbar=SubElement(root,'toolbar')
    tl1=SubElement(toolbar,'tool')
    tl1.attrib['lib']='6'
    tl1.attrib['name']='Poke Tool'
    tl2=SubElement(toolbar,'tool')
    tl2.attrib['lib']='6'
    tl2.attrib['name']='Edit Tool'
    tl3=SubElement(toolbar,'tool')
    tl3.attrib['lib']='6'
    tl3.attrib['name']='Text Tool'
    
    aaa1=SubElement(tl3,'a')
    aaa1.attrib['name']='text'
    aaa1.attrib['val']=''
    
    aaa2=SubElement(tl3,'a')
    aaa2.attrib['name']='font'
    aaa2.attrib['val']='SansSerif plain 12'
    
    aaa3=SubElement(tl3,'a')
    aaa3.attrib['name']='halign'
    aaa3.attrib['val']='center'
    
    aaa4=SubElement(tl3,'a')
    aaa4.attrib['name']='valign'
    aaa4.attrib['val']='base'
    
    sep=SubElement(toolbar,'sep')
    
    ttl1=SubElement(toolbar,'tool')
    ttl1.attrib['lib']='0'
    ttl1.attrib['name']='Pin'
    aaa0=SubElement(ttl1,'a')
    aaa0.attrib['name']='tristate'
    aaa0.attrib['val']='false'
    
    
    ttl2=SubElement(toolbar,'tool')
    ttl2.attrib['lib']='0'
    ttl2.attrib['name']='Pin'
    
    aaa20=SubElement(ttl2,'a')
    aaa20.attrib['name']='facing'
    aaa20.attrib['val']='west'
    
    aaa21=SubElement(ttl2,'a')
    aaa21.attrib['name']='output'
    aaa21.attrib['val']='true'
    
    aaa22=SubElement(ttl2,'a')
    aaa22.attrib['name']='labelloc'
    aaa22.attrib['val']='east'
    
    ttl3=SubElement(toolbar,'tool')
    ttl3.attrib['lib']='1'
    ttl3.attrib['name']='NOT Gate'
    
    ttl4=SubElement(toolbar,'tool')
    ttl4.attrib['lib']='1'
    ttl4.attrib['name']='AND Gate'
    
    ttl5=SubElement(toolbar,'tool')
    ttl5.attrib['lib']='1'
    ttl5.attrib['name']='OR Gate'
    
        
    new_tag =SubElement(root, 'circuit')
    new_tag.attrib['name'] = 'main' # must be str; cannot be an int
    
    sub1=SubElement(new_tag, 'a')
    sub1.attrib['name']='circuit'
    sub1.attrib['val']='main'
    
    sub2=SubElement(new_tag, 'a')
    sub2.attrib['name']='clabel'
    sub2.attrib['val']=''
    
    sub3=SubElement(new_tag, 'a')
    sub3.attrib['name']='clabelup'
    sub3.attrib['val']='east'
    
    sub4=SubElement(new_tag, 'a')
    sub4.attrib['name']='clabelfont'
    sub4.attrib['val']='SansSerif plain 12'
    
    for i in range(len(components)):
        components[i]['tlx']=int(components[i]['tlx'])
        components[i]['tly']=int(components[i]['tly'])
        components[i]['brx']=int(components[i]['brx'])
        components[i]['bry']=int(components[i]['bry'])
        
    
    for i in range(len(components)):
        if(((abs(int(components[i]['tlx']/10))*10)-components[i]['tlx']) < 5 ):
            components[i]['tlx']=(int(components[i]['tlx']/10))*10
            print('+++++++++++++')
            print(int(components[i]['tlx']/10))
            print('+++++++++++++')
            
        else:
            components[i]['tlx']=(int(components[i]['tlx'])/10)*10+10
        if(((abs(int(components[i]['tly']/10))*10)-components[i]['tly']) < 5 ):
            components[i]['tly']=(int(components[i]['tly']/10))*10
        else:
            components[i]['tly']=(int(components[i]['tly']/10))*10+10
        if(((abs(int(components[i]['brx']/10))*10)-components[i]['brx']) < 5 ):
            components[i]['brx']=(int(components[i]['brx']/10))*10
        else:
            components[i]['brx']=(int(components[i]['brx']/10))*10+10
        if(((abs(components[i]['bry']/10)*10)-components[i]['bry']) < 5 ):
            components[i]['bry']=(int(components[i]['bry']/10))*10
        else:
            components[i]['bry']=(int(components[i]['bry']/10))*10+10
            
    print('-------------------')        
    print(components)
    tlx=0
    pos=[]
    for i in range(len(components)):
        if(components[i]['label']=='not'):
            tlx=components[i]['tlx']
            pos.append(i)
            #components[i]['brx']=components[i]['brx']+40
            print('found')
    print(pos)
    '''
    for i in range(len(pos)):
        for j in range(len(components)):
            if(components[pos[i]]['brx']==components[j]['tlx']):
                components[j]['tlx']=components[j]['tlx']+40                
                
            
    for i in range(len(pos)):
        for j in range(len(components)):    
            if(components[j]['tlx']==components[pos[i]]['brx'] and components[pos[i]]['brx']!=0):
                print('yes')
                wire_tly=components[pos[i]]['tly']+(components[pos[i]]['bry']-components[pos[i]]['tly'])/2
                if((abs(wire_tly-int(wire_tly/10)*10)) < 5):
                    wire_tly=int(wire_tly/10)*10
                else:
                    wire_tly=int(wire_tly/10)*10+10
                wire_tlx=components[pos[i]]['brx']
                wire_brx=wire_tlx+20
                wire_bry=wire_tly
                components.append({'label':'wire','topleft':{'x':wire_tlx, 'y':wire_tly},'bottomright':{'x':wire_brx, 'y':wire_bry},'tlx':wire_tlx,'tly':wire_tly,'brx':wire_brx,'bry':wire_bry})
            
    for i in range(len(pos)):
        components[pos[i]]['brx']=components[pos[i]]['brx']+50
        
    
    print('!!!!!!!!!!!!!!!!!!!')
    print(components)            
    print('!!!!!!!!!!!!!!!!!!!')

    for i in range(len(pos)):
        for j in range(len(components)):
            tly=components[pos[i]]['tly']+(components[pos[i]]['bry']-components[pos[i]]['tly'])/2                
            if(((components[pos[i]]['brx']==components[j]['tlx']) and (tly!=components[j]['tly'])) or (components[pos[i]]['brx']!=components[j]['tlx'])):
                print('trueeeeeeeeeeeeeeeeee')    
                if(abs((int(components[pos[i]]['brx']/10)*10) - (components[pos[i]]['brx'])) < 5):
                    tlx=int(components[pos[i]]['brx']/10)*10
                    print(tlx)
                else:
                    tlx=int(components[pos[i]]['brx']/10)*10+10
                if(abs(int(tly/10)*10 - tly) < 5):
                    tly=int(tly/10)*10
                else:
                    tly=int(tly/10)*10+10 
                components.append({'label':'wire','tlx':tlx,'tly':tly,'brx':components[j]['tlx'],'bry':components[j]['tly']})                                
        
                '''
    for i in range(len(pos)):
        not_loc=components[pos[i]]['tly']+(components[pos[i]]['bry']-components[pos[i]]['tly'])/2
        if(not_loc%10 ==5):
            not_loc=not_loc+5
        for j in range(len(components)):
            if( (components[j]['tly']==not_loc) and (components[j]['tlx']==components[pos[i]]['brx']) ):
                components[j]['tlx']=components[j]['tlx']+30
            elif( components[j]['tly']!=not_loc and ((components[j]['tlx']==components[pos[i]]['brx'])) ):
                components[j]['tlx']=components[j]['tlx']+30
                components.append({'label':'wire','tlx':components[pos[i]]['brx']+30,'tly':not_loc,'brx':components[j]['tlx'],'bry':components[j]['tly']})
    for i in range(len(pos)):
        for j in range(len(components)):
            if( (components[j]['label']!='wire') and ((components[pos[i]]['brx']-30)<components[j]['brx']) ):
                components[pos[i]]['tlx']=components[pos[i]]['tlx']+30
                components[pos[i]]['brx']=components[pos[i]]['brx']+30
                break
                
    for i in range(len(pos)):
        for j in range(len(components)):
            if(components[j]['label']!='wire'):
                comp_pos=components[pos[i]]['brx']-70
                if(comp_pos<components[j]['brx']):
                    wire_tlx=components[j]['brx']
                    wire_tly=components[j]['tly']+(components[j]['bry']-components[j]['tly'])/2
                    if(wire_tly%10==5):
                        wire_tly=wire_tly+5
                    wire_brx=components[pos[i]]['brx']-30
                    wire_bry=components[pos[i]]['tly']+(components[pos[i]]['bry']-components[pos[i]]['tly'])/2
                    if(wire_bry%10==5):
                        wire_bry=wire_bry+5
                    components.append({'label':'wire','tlx':wire_tlx,'tly':wire_tly,'brx':wire_brx,'bry':wire_bry})
                    break
                
    for i in range(len(components)):
        if(components[i]['label']=='wire'):
            v=SubElement(new_tag,'wire')
            tlx=str(int(components[i]['tlx']))
            tly=str(int(components[i]['tly']))
            fr='('+tlx+','+tly+')'
            v.attrib['from']=fr
            brx=str(int(components[i]['brx']))
            bry=str(int(components[i]['bry']))
            to='('+brx+','+bry+')'
            v.attrib['to']=to
        else:
            v=SubElement(new_tag,'comp')
            v.attrib['lib']='1'
            x=str(int(components[i]['brx']))
            y=int(components[i]['tly']+(components[i]['bry']-components[i]['tly'])/2)
            if(abs((int(y/10)*10) - y)==5):
                y=str(y+5)
                print('yyyyyy')
                print(y)
            else:
                y=str(y)
            v.attrib['loc']='('+x+','+y+')'
            if(components[i]['label']=='and'):
                v.attrib['name']='AND Gate'
            elif(components[i]['label']=='or'):
                v.attrib['name']='OR Gate'
            else:
                v.attrib['name']='NOT Gate'

                
                
            
            
         
            

    xml.etree.ElementTree.dump(root)
    tree = ElementTree(root)
    tree.write(open('q1.circ','w+'), encoding='unicode')

components=[{'label': 'and', 'topleft': {'x': 254, 'y': 33}, 'bottomright': {'x': 391, 'y': 239}, 'tlx': 254, 'tly': 33, 'brx': 295, 'bry': 74}, {'label': 'not', 'topleft': {'x': 372, 'y': 97}, 'bottomright': {'x': 430, 'y': 164}, 'tlx': 295, 'tly': 46, 'brx': 310, 'bry': 61}, {'label': 'wire', 'topleft': {'x': 99, 'y': 101}, 'bottomright': {'x': 255, 'y': 101}, 'tlx': 99, 'tly': 47, 'brx': 254, 'bry': 47}, {'label': 'wire', 'topleft': {'x': 429, 'y': 125}, 'bottomright': {'x': 462.0, 'y': 125}, 'tlx': 310, 'tly': 53, 'brx': 462.0, 'bry': 53}, {'label': 'wire', 'topleft': {'x': 462.0, 'y': 125}, 'bottomright': {'x': 462.0, 'y': 225.0}, 'tlx': 462.0, 'tly': 53, 'brx': 462.0, 'bry': 175}, {'label': 'wire', 'topleft': {'x': 114, 'y': 156}, 'bottomright': {'x': 255, 'y': 156}, 'tlx': 114, 'tly': 59, 'brx': 254, 'bry': 59}, {'label': 'and', 'topleft': {'x': 552, 'y': 161}, 'bottomright': {'x': 689, 'y': 359}, 'tlx': 552, 'tly': 161, 'brx': 593, 'bry': 202}, {'label': 'not', 'topleft': {'x': 671, 'y': 221}, 'bottomright': {'x': 722, 'y': 282}, 'tlx': 593, 'tly': 174, 'brx': 608, 'bry': 189}, {'label': 'wire', 'topleft': {'x': 462.0, 'y': 225.0}, 'bottomright': {'x': 553, 'y': 225.0}, 'tlx': 462.0, 'tly': 175, 'brx': 552, 'bry': 175}, {'label': 'wire', 'topleft': {'x': 550.0, 'y': 281.0}, 'bottomright': {'x': 553, 'y': 281.0}, 'tlx': 302, 'tly': 187, 'brx': 552, 'bry': 187}, {'label': 'wire', 'topleft': {'x': 0, 'y': 0}, 'bottomright': {'x': 750, 'y': 0}, 'tlx': 608, 'tly': 181, 'brx': 750, 'bry': 181}]
#components=[{'label': 'and', 'topleft': {'x': 242, 'y': 129}, 'bottomright': {'x': 351, 'y': 284}, 'tlx': 242, 'tly': 129, 'brx': 283, 'bry': 170}, {'label': 'not', 'topleft': {'x': 335, 'y': 180}, 'bottomright': {'x': 371, 'y': 222}, 'tlx': 283, 'tly': 142, 'brx': 298, 'bry': 157}, {'label': 'wire', 'topleft': {'x': 370, 'y': 190}, 'bottomright': {'x': 411.0, 'y': 190}, 'tlx': 298, 'tly': 149, 'brx': 411.0, 'bry': 149}, {'label': 'wire', 'topleft': {'x': 411.0, 'y': 190}, 'bottomright': {'x': 411.0, 'y': 272.0}, 'tlx': 411.0, 'tly': 149, 'brx': 411.0, 'bry': 242}, {'label': 'wire', 'topleft': {'x': 89, 'y': 218}, 'bottomright': {'x': 243, 'y': 218}, 'tlx': 89, 'tly': 143, 'brx': 242, 'bry': 143}, {'label': 'or', 'topleft': {'x': 480, 'y': 228}, 'bottomright': {'x': 649, 'y': 363}, 'tlx': 480, 'tly': 228, 'brx': 528, 'bry': 277}, {'label': 'wire', 'topleft': {'x': 411.0, 'y': 272.0}, 'bottomright': {'x': 481, 'y': 272.0}, 'tlx': 411.0, 'tly': 242, 'brx': 480, 'bry': 242}, {'label': 'wire', 'topleft': {'x': 70, 'y': 302}, 'bottomright': {'x': 481, 'y': 302}, 'tlx': 70, 'tly': 262, 'brx': 480, 'bry': 262}, {'label': 'wire', 'topleft': {'x': 50, 'y': 0}, 'bottomright': {'x': 0, 'y': 0}, 'tlx': 50, 'tly': 155, 'brx': 242, 'bry': 155}, {'label': 'wire', 'topleft': {'x': 0, 'y': 0}, 'bottomright': {'x': 750, 'y': 0}, 'tlx': 528, 'tly': 252, 'brx': 750, 'bry': 252}]
gen_logisim(components)
