# coding=utf8
import argparse
from email.mime import image
import cv2 
from sklearn.cluster import KMeans
from collections import Counter, OrderedDict
import numpy as np
import argparse
import time
# from mtcnn.mtcnn import MTCNN
import dlib
import matplotlib.pyplot as plt
import imutils
from imutils import face_utils
import PIL
from PIL import Image as im
import os
from skimage.color import rgb2lab, deltaE_cie76
import math

from sklearn.feature_extraction import img_to_graph



FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
])
def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	# create two copies of the input image -- one for the
	# overlay and one for the final output image
	overlay = image.copy()
	output = image.copy()
	# if the colors list is None, initialize it with a unique
	# color for each facial landmark region
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220)]
    	# loop over the facial landmark regions individually
	for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
		# grab the (x, y)-coordinates associated with the
		# face landmark
		(j, k) = FACIAL_LANDMARKS_IDXS[name]
		pts = shape[j:k]
		# check if are supposed to draw the jawline
		if name == "jaw":
			# since the jawline is a non-enclosed facial region,
			# just draw lines between the (x, y)-coordinates
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv2.line(overlay, ptA, ptB, colors[i], 2)
		# otherwise, compute the convex hull of the facial
		# landmark coordinates points and display it
		else:
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay, [hull], -1, colors[i], -1)
    	# apply the transparent overlay
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
	# return the output image
	return output
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)

arr = []
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# loop over the face parts individually
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		# clone the original image so we can draw on it, then
		# display the name of the face part on the image
		clone = image.copy()
		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)
		# loop over the subset of facial landmarks, drawing the
		# specific face part
		for (x, y) in shape[i:j]:
			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

    		# extract the ROI of the face region as a separate image
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		roi = image[y:y + h, x:x + w]
		roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
		# show the particular face part

# JUST COMMENTED OUTT

		# cv2.imshow("ROI", roi)
		cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
		arr.append(roi)

		# printin and its a blue smurf


		# data = im.fromarray(roi)
		# data.save('gfg_dummy_pic.jpg')
		print(roi)
		# cv2.imshow("Image", clone)
		# cv2.waitKey(0)
	# visualize all facial landmarks with a transparent overlay
	output = face_utils.visualize_facial_landmarks(image, shape)
	# cv2.imshow("Image", output)
	# cv2.waitKey(0)
data = im.fromarray(arr[5])
data.save('gfg_dummy_pic.png')
data2 = im.fromarray(arr[6])
data2.save('gfg_dummy2_pic.jpg')



# # the color detection of an image below


def show_img_compar(img_1, img_2 ):
    f, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis('off') #hide the axis
    ax[1].axis('off')
    f.tight_layout()
    plt.show()
# img = cv2.imread("gfg_dummy_pic.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_2 = cv2.imread("gfg_dummy2_pic.jpg")
# img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
img = PIL.Image.open('gfg_dummy2_pic.jpg').convert('RGB') 
img = np.array(img) 
# Convert RGB to BGR 
img = img[:, :, ::-1].copy() 
img2 = PIL.Image.open('gfg_dummy_pic.jpg').convert('RGB') 
img2 = np.array(img2) 
# Convert RGB to BGR 
img2 = img2[:, :, ::-1].copy() 
dim = (500, 300)

clt = KMeans(n_clusters=5)
clt.fit(img.reshape(-1,3))
clt.labels_
clt.cluster_centers_

err = []
def palette_perc(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)
    
    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_) # count how many pixels per cluster
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    perc = dict(sorted(perc.items()))
    
    #for logging purposes
    # print(perc)
    # shows the pixels of each one
    print(k_cluster.cluster_centers_)
    err.append(k_cluster.cluster_centers_[0])
    step = 0
    
    for idx, centers in enumerate(k_cluster.cluster_centers_): 
        palette[:, step:int(step + perc[idx]*width+1), :] = centers
        step += int(perc[idx]*width+1)

    return palette

  
clt_1 = clt.fit(img.reshape(-1, 3))
show_img_compar(img, palette_perc(clt_1))

clt_2 = clt.fit(img2.reshape(-1, 3))
show_img_compar(img2, palette_perc(clt_2))

# dictionary extraction, take the images key value pairs with file paths to compare with the shortest path

listo = {"LIGHT": ".../color_layers/peach.jpg", "BROWN": ".../color_layers/brown.jpg", 
"MID":".../color_layers/midtone.jpg", "DARK": ".../color_layers/ashbrown.jpg",
 "OLIVE" : ".../color_layers/olive.jpg" }


image = cv2.imread('pic.jpg')
print("The type of this input is {}".format(type(image)))
print("Shape: {}".format(image.shape))
plt.imshow(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')

# filtering images
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_colors(image, number_of_colors, show_chart):
	modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
	modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
	clf = KMeans(n_clusters = number_of_colors)
	labels = clf.fit_predict(modified_image)
	counts = Counter(labels)

	center_colors = clf.cluster_centers_
	# We get ordered colors by iterating through the keys
	ordered_colors = [center_colors[i] for i in counts.keys()]
	hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
	rgb_colors = [ordered_colors[i] for i in counts.keys()]

	if (show_chart):
		plt.figure(figsize = (8, 6))
		plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

	return rgb_colors

IMAGE_DIRECTORY = 'images'
COLORS = {
	"BROWN" : [92,62,23],
	"DARK" : [36, 27,6],
	"LIGHT" : [240,185,152]
}

min = 2345555555555
closest = ""
for e in COLORS:
	distance = math.sqrt(sum([(err[0][i] - COLORS[e][i]) ** 2 for i in range(3)]))
	if distance < min:
		min = distance
		closest = e
	print("The distance between {} and {} is {}".format(e, err[0], distance))
print(closest)



  
# Using cv2.imread() method
imagess = cv2.imread(listo[closest])
  
# Displaying the image
cv2.imshow(listo[closest], imagess)

cv2.waitKey(0)

# cv2.destroyAllWindows() simply destroys all the windows we created.

cv2.destroyAllWindows()

# The function cv2.imwrite() is used to write an image.




images = []

for file in os.listdir(IMAGE_DIRECTORY):
    if not file.startswith('.'):
        images.append(get_image(os.path.join(IMAGE_DIRECTORY, file)))



def match_image_by_color(image, color, threshold = 60, number_of_colors = 10): 
    
    image_colors = get_colors(image, number_of_colors, False)
    selected_color = rgb2lab(np.uint8(np.asarray([[color]])))

    select_image = False
    for i in range(number_of_colors):
        curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
        diff = deltaE_cie76(selected_color, curr_color)
        if (diff < threshold):
            select_image = True
    
    return select_image







