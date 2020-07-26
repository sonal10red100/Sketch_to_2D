
import matplotlib.pyplot as plt
import numpy as np

from darkflow.net.build import TFNet
import cv2
import pprint as pp

def boxing(original_img , predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        print(predictions)
        label = result['label'] + " " + str(round(confidence, 3))
        if confidence > 0.5:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 2)
            newImage = cv2.putText(newImage, label, (top_x, top_y+20), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (0, 0, 0), 1, cv2.LINE_AA)
            print(top_x)
            print(top_y)
            print(btm_x)
            print(btm_y)
            print('size')
            print(newImage.shape)
    return newImage

options = {"model": "cfg/tiny-yolo-voc-3c.cfg",
           "load": 8625,
           "gpu": 1.0,
          "threshold":0.5}

tfnet2 = TFNet(options)


original_img = cv2.imread("IMG_20200526_183731.jpg")
#original_img = cv2.imread("result/attachments/IMG_20200113_222352.jpg")
original_img = cv2.resize(original_img, (800, 400), interpolation = cv2.INTER_AREA)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
results = tfnet2.return_predict(original_img)
print(results)
#print(len(results))

fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(original_img)

fig, ax = plt.subplots(figsize=(15, 15))
outimg = boxing(original_img, results)
cv2.imwrite("comp_detect_img.jpg", outimg   )
ax.imshow(outimg)
