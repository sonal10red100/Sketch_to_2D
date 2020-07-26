# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:18:13 2020

@author: Chandraprakash Sharm
"""

import os
import glob

from final_bool import gen_expression

img_dir = "result\\app\\logic_expression_detect\\*.txt" # Enter Directory of all images 
#data_path = os.path.join(img_dir,'*g')


files = glob.glob(img_dir)
cnt=0

wire_cnt_gd = 0
wire_cnt_r = 0
score = 0
total_score = 0 
exp_gd = ""
exp_r = ""
for f1 in files:
    wire_cnt_gd = 0
    wire_cnt_r = 0
    score = 0
    score_exp = 0
    score_wire = 0
    deviation = 0
    
    fr = open(f1, 'r')
    exp_r = fr.readline()[:-1]
    print(exp_r)
    wire_cnt_r = int(fr.readline())
    fr.close()
    
    
    file_name = os.path.basename(str(f1))[0:os.path.basename(str(f1)).find('.')]
    fgd = open("result\\app\\ground_truth\\"+str(file_name)+".txt", 'r')
    exp_gd = fgd.readline()[:-1]
    print(exp_gd)
    wire_cnt_gd = int(fgd.readline())
    fgd.close()
#    if(is_digit(wire_cnt_gd))
    
#    print('gd ', wire_cnt_gd, exp_gd, wire_cnt_r, exp_r)
    if(exp_r != ""):
        if(exp_r == exp_gd):
            score_exp = 4
#        if( exp_r[-1] != "\'"):
#            if( exp_r[1:-1] == exp_gd ):
#                score_exp = 2
#        elif exp_r == exp_gd:
#            score_exp = 2
#        print('Expression Matches')
    
    deviation = abs(wire_cnt_r - wire_cnt_gd)/wire_cnt_gd
    score_wire = ( 1-deviation )*6
#    
    score = score_exp + score_wire
#    print(score, score_wire, score_exp)
    print(file_name)
    
    total_score += score
    cnt+=1
#    if cnt==2:
#        break

print(total_score / 562)
print('finished')
#gdFile.close()