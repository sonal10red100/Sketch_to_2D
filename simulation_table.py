# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:56:55 2020

@author: Maitreyi Sharma
"""

class Table:
	def __init__(self,root,total_rows,total_columns, lst): 
#		print(lst)        
#		print('sssssssssssssssss')
		print(total_rows)
			# code for creating table 
		for i in range(total_rows): 
			for j in range(total_columns):
				self.e = Entry(root, width=10, fg='black', font=('Arial',16,'bold'))
				self.e.grid(row=i, column=j)
				self.e.insert(END, lst[i][j])