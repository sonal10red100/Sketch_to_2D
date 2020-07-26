import cv2 as cv2
import os
import numpy as np

weightMatrix = []
finalListOfPoints = []
'''
predictions=[{'label': 'and', 'confidence': 0.77202564, 'topleft': {'x': 246, 'y': 52}, 'bottomright': {'x': 424, 'y': 217}}, {'label': 'and', 'confidence': 0.59216535, 'topleft': {'x': 575, 'y': 193}, 'bottomright': {'x': 696, 'y': 319}}]
'''
def generate_straight_lines(arr,x1,x2,y1,y2):
	    

	threshold=16
	for i in range(len(arr)-1):	
		if(abs(arr[i][1] - arr[i+1][1]) < threshold): 
			#print("y closer",arr[i][1],' ',arr[i+1][1])
			arr[i+1]=list(arr[i+1])
			arr[i+1][1]=arr[i][1]
			arr[i+1]=tuple(arr[i+1])		
		elif(abs(arr[i][0] - arr[i+1][0]) < threshold):
			#print("x closer",arr[i][0],' ',arr[i+1][0])
			arr[i+1]=list(arr[i+1])
			arr[i+1][0]=arr[i][0]
			arr[i+1]=tuple(arr[i+1])
	
	order=""
	for i in range(len(arr)-1):
		if(arr[i][1]>arr[i+1][1]):
			order="dec"
		elif(arr[i][1]<arr[i+1][1]):
			order="inc"
	
	'''print("++++++++++")
	for i in range(len(arr)):
	print(arr[i])
	print("++++++++++")
	'''

	if(order=="inc"):
		arr=sorted(arr , key=lambda k: [k[0],k[1]])
	
	res_list = [] 
	for i in range(len(arr)): 
		if(arr[i] not in arr[i + 1:]): 
			res_list.append(arr[i])

	for i in range(len(res_list)):
		arr[i]=res_list[i] 

	for i in range(len(arr)):
		print(arr[i])	
	if(len(arr)>1):	
		if(arr[0][0]==arr[1][0]):
			arr.remove(arr[0])
		
	for i in range(len(arr)-1):
		if((arr[i][0]!=arr[i+1][0]) and (arr[i][1]!=arr[i+1][1])):
			d1=abs(arr[i][0] - arr[i+1][0])
			d2=abs(arr[i][1] - arr[i+1][1])	
			if(d1 <= d2):
				print('x bef : arr[i]',arr[i],'arr[i+1]',arr[i+1])
				arr[i+1]=list(arr[i+1])
				arr[i+1][0]=arr[i][0]
				arr[i+1]=tuple(arr[i+1])
				print('x : arr[i]',arr[i],'arr[i+1]',arr[i+1])
								
			else:
				arr[i+1]=list(arr[i+1])
				arr[i+1][1]=arr[i][1]
				arr[i+1]=tuple(arr[i+1])
				print('arr[i]',arr[i],'arr[i+1]',arr[i+1])
	print('len',len(arr))
	final_list = [] 
	for i in range(len(arr)): 
		if(arr[i] not in arr[i + 1:]): 
			final_list.append(arr[i])

	for i in range(len(final_list)):
		arr[i]=final_list[i] 
	print('len',len(arr))
	for i in range(len(arr)):
		print(arr[i])	

	

	i=0
	p1=0
	p2=0
	while i < (len(arr)-1):
		m=i
		p1=arr[i]
		while (arr[i][1] == arr[i+1][1]):
			if((i+2) == len(arr)):	
				i=i+1					
				break
			i=i+1
		p2=arr[i]
		print('p1',p1,'p2',p2)
		print('x1',x1,'y1',y1)

		
		xx1=p1[0]+x1
		yy1=p1[1]+y1
		xx2=p2[0]+x1
		yy2=p2[1]+y1
		
		finalListOfPoints.append((xx1,yy1))
		finalListOfPoints.append((xx2,yy2))

		weightMatrix.append({'label':'wire','topleft':{'x':xx1, 'y':yy1},'bottomright':{'x':xx2, 'y':yy2}})
		p1=p2
		if((i+1) == len(arr)):
			break
		while (arr[i][0] == arr[i+1][0]):
			if((i+2) == len(arr)):	
				i=i+1					
				break
			i=i+1
		p2=arr[i]
		print('p1',p1,'p2',p2)
		print('x1',x1,'y1',y1)
		xx1=p1[0]+x1
		yy1=p1[1]+y1
		xx2=p2[0]+x1
		yy2=p2[1]+y1
			
		finalListOfPoints.append((xx1,yy1))
		finalListOfPoints.append((xx2,yy2))

		weightMatrix.append({'label':'wire','topleft':{'x':xx1, 'y':yy1},'bottomright':{'x':xx2, 'y':yy2}})	
		p1=p2
		if(m==i):
			i=i+1

	#np.append(predictions,weightMatrix)
	print(weightMatrix)
	print(len(weightMatrix))
	    
	   


def shi_tomasi(image,x1,x2,y1,y2,h,w):
	
	#Converting to grayscale
	gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	(thresh, bw) = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
	

	#Specifying maximum number of corners as 1000
	# 0.01 is the minimum quality level below which the corners are rejected
	# 10 is the minimum euclidean distance between two corners
	corners_img = cv2.goodFeaturesToTrack(bw,1000,0.01,20)
	
	#corners_img = np.int0(corners_img)
	print('----------------------',type(corners_img))
	try:
		corners_list=[]
		for corners in corners_img:		   
			x,y = corners.ravel()
			print('x :',x,'y :',y)
			corners_list.append((x,y))
			#Circling the corners in green
				#cv2.circle(image,(x,y),3,[0,255,0],-1)
			
		corners_list.append((0,0))
		z=0
		print('w h',w,h)
		for i in range(len(corners_list)):
			if(corners_list[i][0]==w):
				z=1
				break
		    
		sorted_corners_list=sorted(corners_list)
		first_y=sorted_corners_list[0][1]
		sorted_corners_list.insert(0,(0,first_y))
		last_y=sorted_corners_list[len(sorted_corners_list)-1][1]
		sorted_corners_list.append((w,last_y))
		generate_straight_lines(sorted_corners_list,x1,x2,y1,y2)
	except:
		print('None type error occurred')
	    
	print(finalListOfPoints)

	    #cv2.imshow('image',image)
	    #cv2.waitKey(0)

def operateBox(img,x1,y1,x2,y2):
	boxImg=img[y1:y2,x1:x2]
	h,w=boxImg.shape[:2]
	cv2.imwrite('ob.jpg',boxImg)
	#cv2.imshow('sdffdgyh',boxImg)
	#cv2.waitKey(0)
	new_img = cv2.imread('ob.jpg')
	shi_tomasi(new_img,x1,x2,y1,y2,h,w)		
	
	


def wire_detection(predictions):
	#global path
	#global predictions
	#load a simple image
	img = cv2.imread('temp.jpg')
        #print('PATHHHHHHHHHHHHHHHH',path)
	img=cv2.resize(img,(800,400))
	del weightMatrix[:]
	#replace with bounding box coords
	for i in range(len(predictions)):
		xx1=predictions[i]['topleft']['x']
		xx4=predictions[i]['bottomright']['x']
		yy1=predictions[i]['topleft']['y']
		yy2=predictions[i]['bottomright']['y']
		bb_area=(yy2-yy1)*(xx4-xx1)
		#print('bb area:-------------->',bb_area)
		
		s=str(bb_area)
		if(len(s)==4):
			x1=predictions[i]['topleft']['x']-10
			if(x1<0):
				x1=3
			x2=predictions[i]['topleft']['x']-10
			if(x2<0):
				x2=3						
			x3=predictions[i]['bottomright']['x']+12
			if(x3>800):
				x3=797
			x4=predictions[i]['bottomright']['x']+12
			if(x4>800):
				x4=797				
			y1=predictions[i]['topleft']['y']-10
			if(y1<0):
				y1=3
			y2=predictions[i]['bottomright']['y']+10
			if(y2>400):
				y2=397
			y3=predictions[i]['bottomright']['y']+10
			if(y3>400):
				y3=397
			y4=predictions[i]['topleft']['y']-10
			if(y4<0):
				y4=3			
			
			predictions[i]['topleft']['x']=x1
			predictions[i]['bottomright']['x']=x3
			predictions[i]['topleft']['y']=y1
			predictions[i]['bottomright']['y']=y2
			
			pts = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], np.int32)
			cv2.fillPoly(img, [pts], (255,255,255))
				
		else:

			x1=predictions[i]['topleft']['x']-18
			if(x1<0):
				x1=3
			x2=predictions[i]['topleft']['x']-18
			if(x2<0):
				x2=3
			x3=predictions[i]['bottomright']['x']+18
			if(x3>800):
				x3=797
			x4=predictions[i]['bottomright']['x']+18
			if(x4>800):
				x4=797		
			y1=predictions[i]['topleft']['y']-15
			if(y1<0):
				y1=3
			y2=predictions[i]['bottomright']['y']+16
			if(y2>400):
				y2=397
			y3=predictions[i]['bottomright']['y']+16
			if(y3>400):
				y3=397
			y4=predictions[i]['topleft']['y']-15
			if(y4<0):
				y4=3
			predictions[i]['topleft']['x']=x1
			predictions[i]['bottomright']['x']=x3
			predictions[i]['topleft']['y']=y1
			predictions[i]['bottomright']['y']=y2
			
			pts = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], np.int32)
			cv2.fillPoly(img, [pts], (255,255,255))
					
	
	#cv2.imshow('remove gates',img)
	#cv2.waitKey(0)

	
	lower = np.array([0,0,0]) 
	#upper = np.array([240,72,136])
	upper = np.array([162,128,141])


	mask = cv2.inRange(img, lower, upper)
	res = cv2.bitwise_and(img,img, mask=mask)
	#Replace all black portion to white
	res[np.where(res==[0])]=[255]
	


	img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY);
	for i in range(400):
		for j in range(800):
			k=img[i,j]
			if(k>70 and i>300):
				img[i,j]=255
	
	#img[np.where(img>[80])]=[255]
	#cv2.imshow('Replace all black portion to white',img)
	#cv2.waitKey(0)
	
		
	kernel = np.ones((3,3), np.uint8)
	img = cv2.erode(img,kernel,iterations=1)   
	#cv2.imshow('erosion', img)   
	#cv2.waitKey(0) 

	ret,thresh = cv2.threshold(img,127,255,0)
	canny_img = cv2.Canny(img,100,200)
	im2, contours, hierarchy = 	cv2.findContours(canny_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	'''im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)

	cv2.drawContours(img, contours, -1, (0, 255, 0), 3) 
	cv2.imshow('Contours', img) 
	cv2.waitKey(0) 
	'''	
	for item in range(len(contours)):
		cnt = contours[item]
		if len(cnt)>20:
			print(len(cnt))
			M = cv2.moments(cnt)
			#cx = int(M['m10']/M['m00'])
			#cy = int(M['m01']/M['m00'])
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),1)
			#cv2.imshow('initial image',img)
			#cv2.waitKey(0)
			if(x==0 and y==0 and (x+w)==800 and (y+h)==400):
				continue;
			x1=x
			y1=y
			x2=x+w
			y2=y+h
			print('x1:',x1,'y1:',y1,'x2:',x2,'y2:',y2)
			operateBox(img,x1,y1,x2,y2)        
	#cv2.destroyAllWindows()
	pre = predictions + weightMatrix
	return pre
	#cv2.imshow('initial image',img)
	#cv2.waitKey(0)

#path='/home/sonal/Desktop/q.jpg'



#wire_detection()
#np.append(predictions,weightMatrix)

#print(predictions)
'''
t=10	    
for i in range(len(finalListOfPoints)-1):
	for j in range(i+1,len(finalListOfPoints)-1):
		if((abs(finalListOfPoints[i][1]-finalListOfPoints[j][1]) < t) and ((abs(finalListOfPoints[i][1]-finalListOfPoints[j][1]) > 0) and (abs(finalListOfPoints[i][0]-finalListOfPoints[j][0]) < t) and (abs(finalListOfPoints[i][0]-finalListOfPoints[j][0]) > 0))):
			for k in range(len(weightMatrix)):	
				a1=weightMatrix[k]['topleft']['x']
				b1=weightMatrix[k]['topleft']['y']
				if((a1==finalListOfPoints[j][0]) and (b1==finalListOfPoints[j][1])):
					weightMatrix[k]['topleft']['x']=finalListOfPoints[i][0]	
			  		weightMatrix[k]['topleft']['y']=finalListOfPoints[i][1]
			finalListOfPoints[j]=list(finalListOfPoints[j])
			finalListOfPoints[j][1]=finalListOfPoints[i][1]
			finalListOfPoints[j][0]=finalListOfPoints[i][0]
			finalListOfPoints[j]=tuple(finalListOfPoints[j])


img = cv2.imread(path)
img=cv2.resize(img,(800,400))	
for i in range(len(weightMatrix)):	
	a1=weightMatrix[i]['topleft']['x']
	b1=weightMatrix[i]['topleft']['y']
	a2=weightMatrix[i]['bottomright']['x']
	b2=weightMatrix[i]['bottomright']['y']

	
	cv2.circle(img,(a1,b1),3,[0,255,0],-1)
	cv2.circle(img,(a2,b2),3,[0,255,0],-1)
'''
#cv2.imshow('check',img)
#cv2.waitKey(0)



		    	

