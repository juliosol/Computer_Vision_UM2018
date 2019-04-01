import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import scipy
import scipy.stats
import statistics
import math

random.seed(784)
np.random.seed(784)

image = cv2.imread('sweden.jpg')
presav_gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.jpg',presav_gray_image)
cv2.imshow('gray_sweden',presav_gray_image)
cv2.waitKey(0)         
cv2.destroyAllWindows()
blur_gray_image = cv2.blur(presav_gray_image,(20,20),0)
cv2.imwrite('gray_sweden.jpg',blur_gray_image)


## Initializing parameters for the Tabu search MRF algorithm
tabu_list = {}
init_temp = 0.1
total_iterations = 14
#alpha = 0.02
beta = 5
sigma = 6
#threshold = 0.0
original_image = presav_gray_image
regions = 2
height, width = original_image.shape

## Function that adds noise to an image following a Gaussian distribution
## print(blur_gray_desk)
def noisy(image, mean, variance):
	row,col= image.shape
	sigma = variance**2
	gauss = np.random.normal(mean,sigma,(row,col))
	gauss = gauss.reshape(row,col)
	pre_noisy = np.round(image + gauss,0)
	noisy = pre_noisy.astype(np.uint8)
	return noisy

gaussian_image = noisy(presav_gray_image,0,sigma)
cv2.imshow('gaussian gray_image',gaussian_image)
cv2.waitKey(0)         
cv2.destroyAllWindows()

## Color intervals for each of the regions:

regions_list = {}
for j in range(regions):
	regions_list[j] = list(range(255//regions*j,255//regions*(j+1)))

regions_parameters = np.zeros([regions,2])
for j in regions_list.keys():
	regions_parameters[j][0] = statistics.mean(regions_list[j])
	regions_parameters[j][1] = statistics.variance(regions_list[j])


## Function that identifies the label that a point corresponds to:

def label_identifier(current_image,i,j,regions_lst,regions_param,regions):
	current_value = current_image[i][j]
	#print(current_value)
	#print(regions_lst.items())
	if current_value > 250:
		label = regions - 1
	else:
		for item in range(regions):
			#print(regions_lst.items())
			current = list(regions_lst.items())[item][1]
			for element in current:
				if element == current_value:
					label = item
	return label

### Function that gives a particular color to each region of an image, according to the labels.

def color_segment(current_image,regions_lst,regions_param):
	row,col = current_image.shape
	for i in range(row):
		for j in range(col):
			current_value = current_image[i][j]
			if current_value > 250:
				label = regions - 1
			else:
				for item in range(regions):
					#print(regions_lst.items())
					current = list(regions_lst.items())[item][1]
					for element in current:
						if element == current_value:
							label = item
			current_image[i][j] = regions_param[label][0]
	#color_current_image = current_image.astype(np.uint8)
	return current_image

## First we compute the singleton energies in a given pixel.

def singleton(current_image,i,j,regions_lst,regions_param,regions):
	current_value = current_image[i][j]
	current_label = label_identifier(current_image,i,j,regions_lst,regions_param,regions)
	#print(current_label)
	mean_label,variance_label = regions_param[current_label]
	singleton_energy = np.log(1/math.sqrt(2*math.pi*variance_label)) - (current_value - mean_label)**2 / 2*variance_label
	return singleton_energy

### Now we write the function that returns the value for the doubleton energy. 

def doubleton(current_image, i,j,regions_lst, regions_param,regions,beta):
	doubleton_energy = 0
	height,width = current_image.shape
	current_value = current_image[i][j]
	current_label = label_identifier(current_image,i,j,regions_lst,regions_param,regions)
	if i !=(height - 1):
		if current_label == label_identifier(current_image,i+1,j,regions_lst,regions_param,regions):
			doubleton_energy = doubleton_energy - beta
		else:
			doubleton_energy = doubleton_energy + beta
	if j != (width - 1):
		if current_label == label_identifier(current_image,i,j+1,regions_lst,regions_param,regions):
			doubleton_energy = doubleton_energy - beta
		else:
			doubleton_energy = doubleton_energy + beta
	if i != 0:
		if current_label == label_identifier(current_image,i-1,j,regions_lst,regions_param,regions):
			doubleton_energy = doubleton_energy - beta
		else:
			doubleton_energy = doubleton_energy + beta
	if j != 0:
		if current_label == label_identifier(current_image,i,j-1,regions_lst,regions_param,regions):
			doubleton_energy = doubleton_energy - beta
		else:
			doubleton_energy = doubleton_energy + beta
	return doubleton_energy

## This function is used to compute the local energy at each pixel which will then be used for the 
## tabu search algorithm.

def local_energy(current_image,i,j,regions_lst,regions_param,regions,beta):
	local_energy = singleton(current_image,i,j,regions_lst,regions_param,regions) + doubleton(current_image,i,j,regions_lst,regions_param,regions,beta)
	return local_energy

## This function computes the total energy of an image (considering all pixels) which will also be
## used for the tabu search algorithm.

def total_energy(current_image,regions_lst,regions_param,regions,beta):
	height,width = current_image.shape
	singletons = 0
	doubletons = 0
	#print(height)
	#print(width)
	for i in range(height):
		for j in range(width):
			#print(i)
			#print(j)
			singletons = singletons + singleton(current_image,i,j,regions_lst,regions_param,regions)
			doubletons = doubletons + doubleton(current_image,i,j,regions_lst,regions_param,regions,beta)
	total_energy = singletons + doubletons/2
	return total_energy

#print(total_energy(gaussian_image,regions_list,regions_parameters,regions))

## This part of the code computes the update of the image from the 
## tabu search algorithm to minimize the energy of the image:
def image_update(current_image,regions_lst,regions_param,regions,beta,sigma,temp):
	""" This function gets as input a gray-scale image that is blurred, a mean
	and variance term in order to perturb a pixel with some noise. Then we 
	follow the tabu search algorithm to update regions in the image. 
	"""
	row,col = current_image.shape
	new_image = np.zeros((row,col))
	perturbed_image = noisy(current_image,0,sigma)
	for i in range(1,row):
		for j in range(1,col):
			#print(i)
			#print(j)
			perturbed_local_energy = local_energy(perturbed_image,i,j,regions_lst,regions_param,regions,beta)
			current_local_energy = local_energy(current_image,i,j,regions_lst,regions_param,regions,beta)
			if (perturbed_local_energy - current_local_energy) <= 0:
				new_image[i][j] = perturbed_image[i][j]
			elif (perturbed_local_energy - current_local_energy) > 0:
				if np.exp(-(perturbed_local_energy - current_local_energy)/temp) > np.random.uniform(0,1):
					new_image[i][j] = perturbed_image[i][j]
				else:
					new_image[i][j] = current_image[i][j]
	return new_image

## Some calls to the functions we have to test them.
original_image = presav_gray_image
current_image = gaussian_image
temp = init_temp
updated_image = image_update(current_image,regions_list,regions_parameters,regions,beta,sigma,temp)
preu_colored_updated_image = color_segment(updated_image,regions_list,regions_parameters)
colored_updated_image = preu_colored_updated_image.astype(np.uint8)

cv2.imshow('updated',colored_updated_image)
cv2.waitKey(0)         
cv2.destroyAllWindows()

## The following function is used to update the tabu list of the Tabu search algorithm:

def tabu_update(current_image,regions_lst,regions_param,regions,tabu_list,beta,sigma,temp):
	updated_image = image_update(current_image,regions_lst,regions_param,regions,beta,sigma,temp)
	updated_energy = total_energy(updated_image,regions_lst,regions_param,regions,beta)
	if len(tabu_list) == 0:
		tabu_list[0] = updated_image
	else:
		tabu_elts = len(tabu_list)
		i = 1
		for p in list(tabu_list):
			curr_tabu_energy = total_energy(tabu_list[p],regions_lst,regions_param,regions,beta)
			if updated_energy < curr_tabu_energy:
				if len(tabu_list.keys()) < 10:
					tabu_list[i] = updated_image
					i = i + 1
				else:
					tabu_list[p] = updated_image
			elif updated_energy >= curr_tabu_energy and np.exp(-updated_energy/temp) > np.random.uniform(0,1):
				if len(tabu_list.keys()) < 10:
					tabu_list[i] = updated_image
					i = i + 1
				else:
					tabu_list[p] = updated_image
	return updated_image, tabu_list


for j in range(total_iterations):
	print(j)
	#print(current_image)
	current_image, tabu_list = tabu_update(current_image,regions_list,regions_parameters,regions,tabu_list,beta,sigma,temp)
	temp = temp/(1 + np.log(1+j))
	#print(temp)
	print(total_energy(current_image,regions_list,regions_parameters,regions,beta))
	#print(tabu_list)
	#cv2.imshow('updated',current_image)
	#cv2.waitKey(0)         
	#cv2.destroyAllWindows()

#print(tabu_list)
preu_colored_current_image = color_segment(current_image,regions_list,regions_parameters)
colored_current_image = preu_colored_current_image.astype(np.uint8)
cv2.imwrite('chess_circle_triangle_2_14.jpg',colored_current_image)

cv2.imshow('color updated', colored_current_image)
cv2.waitKey(0)         
cv2.destroyAllWindows()

cv2.imshow('updated', preu_colored_current_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
