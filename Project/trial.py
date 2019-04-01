import numpy as np
import cv2
import random 
import statistics

regions = 4
regions_list = {}
for j in range(regions):
  regions_list[j] = list(range(255//regions*j,255//regions*(j+1)))

regions_parameters = np.zeros([regions,2])
for j in regions_list.keys():
  regions_parameters[j][0] = statistics.mean(regions_list[j])
  regions_parameters[j][1] = statistics.variance(regions_list[j])
#print(regions_parameter)

square = np.zeros([1000,1000])
for i in range(0,250):
  for j in range(1000):
    square[i][j] = regions_parameters[0][0]
for i in range(251,500):
  for j in range(1000):
    square[i][j] = regions_parameters[1][0]
for i in range(501,750):
  for j in range(1000):
    square[i][j] = regions_parameters[2][0]
for i in range(751,1000):
  for j in range(1000):
    square[i][j] = regions_parameters[3][0]
image_square = square.astype(np.uint8)
cv2.imshow('gaussian gray_image',image_square)
cv2.waitKey(0)         
cv2.destroyAllWindows()

#search_number = 171
#for item in range(regions):    # for name, age in list.items():  (for Python 3.x)
#    current = regions_list.items()[item][1]
#    for element in current:
#      if element == search_number:
#        print(item)
#print(regions_list.items()[0][1])

"""
This part was used to test different types of adding noise to the image for starting 
the algorithm.


import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import scipy
import scipy.stats

random.seed(784)
init_temp = random.randint(0,10)

desk = cv2.imread('desk.jpg')
presav_gray_desk = cv2.cvtColor(desk, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_desk.jpg',presav_gray_desk)
gray_desk = cv2.imread('gray_image.jpg')
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


## This part is for adding salt and pepper noise to the picture.
def sp_noise(image,prob):
  '''
  Add salt and pepper noise to image
  prob: Probability of the noise
  '''
  output = np.zeros(image.shape,np.uint8)
  thres = 1 - prob 
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      rdn = random.random()
      if rdn < prob:
        output[i][j] = 0
      elif rdn > thres:
        output[i][j] = 255
      else:
        output[i][j] = image[i][j]
  return output

noise_img = sp_noise(gray_desk,0.20)
cv2.imwrite('sp_noise.jpg', noise_img)
img2 = cv2.imread('sp_noise.jpg',0)
cv2.imshow('image',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

snr = scipy.stats.signaltonoise(noise_img, axis=None)
print("This is salt and pepper noise " + str(snr))

blur = cv2.blur(gray_desk,(30,30),0)

plt.imshow(blur)
plt.show()
snr_blur = scipy.stats.signaltonoise(blur,axis = None)
print('This is blurring from CV2 ' + str(snr_blur))

kernel = np.ones((10,10),np.float32)/100
dst = cv2.filter2D(gray_desk,-1,kernel)

plt.imshow(dst)
plt.show()
snr_dst = scipy.stats.signaltonoise(dst,axis = None)
print('This is blurring from CV2 ' + str(snr_dst))


#from PIL import Image, ImageFilter
#im = Image.open('desk.jpg')
#im.show()


#Applying Grayscale filter to image
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Saving filtered image to new file
#cv2.imwrite('graytest.jpg',gray)
#img2 = cv2.imread('graytest.jpg')
#cv2.imshow('imag',img2)
#cv2.waitkey(0)
#cv2.destroyAllWindows()

## This part is already working and showing some preliminary division 
## between background and foreground.


## This is the U(z,h,v) function defined in Patra's paper. This function measures
## 
def u_function(image,i,j,alpha,beta,threshold):
  #print(image[i][j])
  #print(image[i][j-1])
  horizontal_similarity = (image[i][j] -image[i][j-1])**2*(1-horizontal_funct(image,i,j,threshold))
  vertical_similarity = (image[i][j] -image[i-1][j])**2 * (1-vertical_funct(image,i,j,threshold))
  alpha_term = alpha*(horizontal_similarity + vertical_similarity)
  beta_term = beta*(vertical_funct(image,i,j,threshold) + horizontal_funct(image,i,j,threshold))
  similarity_pixels = alpha_term + beta_term
  return similarity_pixels

def up_function(original_image,image,i,j,alpha,beta,threshold,sigma):
  u_value = u_function(image,i,j,alpha,beta,threshold)
  orig_value = original_image[i][j]
  test_value = image[i][j]
  fit_level = (orig_value - test_value)**2
  up_value = fit_level/(2*sigma**2) + u_value
  return up_value
    
def image_update(original_image,image,sigma,alpha,beta,threshold,mean = 0):

  row,col = image.shape
  new_image = np.zeros((row,col))
  perturbed_image = noisy(image,mean,sigma)
  for i in range(1,row):
    for j in range(1,col):
      #print(i)
      #print(j)
      up_value_degraded = up_function(original_image,image,i,j,alpha,beta,threshold,sigma)
      up_value_before_degraded = up_function(original_image,image,i,j,alpha,beta,threshold,sigma)
      if (up_value_degraded - up_value_before_degraded) < 0:
        new_image[i][j] = perturbed_image[i][j]
      else:
        new_image[i][j] = image[i][j]
  return new_image

updated = image_update(presav_gray_image,gaussian_image,sigma,alpha, beta,threshold)
cv2.imshow('updated',updated)
cv2.waitKey(0)         
cv2.destroyAllWindows()

#print(updated)

## This is the energy function that can be used to compute the energy of an image.
def energy(image,alpha,beta,threshold):
  row,col = image.shape
  energy = 0
  for i in range(1,row):
    for j in range(1,col):
      horizontal_similarity = (image[i][j] -image[i][j-1])**2*(1-horizontal_funct(image,i,j,threshold))
      vertical_similarity = (image[i][j] -image[i-1][j])**2 * (1-vertical_funct(image,i,j,threshold))
      alpha_term = alpha*(horizontal_similarity + vertical_similarity)
      beta_term = beta*(vertical_funct(i,j) + horizontal_funct(i,j))
      energy = energy + alpha_term + beta_term
  return energy

######################################################################################################
######################################################################################################

###From here on the code is already fully working:

import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import scipy
import scipy.stats

random.seed(784)
np.random.seed(784)

image = cv2.imread('chess_circle_rectangle.jpg')
presav_gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.jpg',presav_gray_image)
cv2.imshow('gray_tangram',presav_gray_image)
cv2.waitKey(0)         
cv2.destroyAllWindows()
#gray_desk = cv2.imread('gray_image.jpg')

#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

blur_gray_image = cv2.blur(presav_gray_image,(20,20),0)
cv2.imwrite('gray_desk.jpg',blur_gray_image)
#cv2.imshow('blur gray_image',blur_gray_desk)
#cv2.waitKey(0)         
#cv2.destroyAllWindows()
#blur_gray_desk = cv2.imread('gray_desk.jpg')
#plt.imshow(blur_gray_desk)
#plt.show()

#snr_blur = scipy.stats.signaltonoise(blur_gray_desk,axis = None)
#print('This is blurring from CV2 ' + str(snr_blur))

## Initializing parameters for the Tabu search MRF algorithm
tabu_list = {}
init_temp = 0.1
total_iterations = 5
alpha = 0.01
beta = 5
sigma = 5
threshold = 0.5
original_image = presav_gray_image

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


## This are the horizontal and vertical functions used in the Patra paper, this behaves
## as a piecewise function that activates if some conditions of the pixel in the 
## image are satisfied.

def horizontal_funct(current_image,i,j,threshold):
  pixel1 = current_image[i][j]
  pixel2 = current_image[i][j-1]
  if abs(pixel1 - pixel2)>threshold:
    result = 1
  else:
    result = 0
  return result

def vertical_funct(current_image,i,j,threshold):
  pixel1 = current_image[i][j]
  pixel2 = current_image[i-1][j]
  if abs(pixel1 - pixel2)>threshold:
    result = 1
  else:
    result = 0
  return result

## This is the U(z,h,v) function defined in Patra's paper. This function measures
## 
def u_function(current_image,i,j,alpha,beta,threshold):
  #print(image[i][j])
  #print(image[i][j-1])
  horizontal_similarity = (current_image[i][j] - current_image[i][j-1])**2*(1-horizontal_funct(current_image,i,j,threshold))
  vertical_similarity = (current_image[i][j] - current_image[i-1][j])**2 * (1-vertical_funct(current_image,i,j,threshold))
  alpha_term = alpha*(horizontal_similarity + vertical_similarity)
  beta_term = beta*(vertical_funct(current_image,i,j,threshold) + horizontal_funct(current_image,i,j,threshold))
  similarity_pixels = alpha_term + beta_term
  return similarity_pixels

def up_function(original_image,current_image,i,j,alpha,beta,threshold,sigma):
  u_value = u_function(current_image,i,j,alpha,beta,threshold)
  orig_value = original_image[i][j]
  test_value = current_image[i][j]
  fit_level = (orig_value - test_value)**2
  up_value = fit_level/(2*sigma**2) + u_value
  return up_value
    
def image_update(original_image,current_image,sigma,alpha,beta,threshold,temp,mean = 0):
   #This function gets as input a gray-scale image that is blurred, a mean
  #and variance term in order to perturb a pixel with some noise. Then we 
  #follow the tabu search algorithm to update regions in the image. 
  row,col = current_image.shape
  new_image = np.zeros((row,col))
  perturbed_image = noisy(current_image,mean,sigma)
  for i in range(1,row):
    for j in range(1,col):
      #print(i)
      #print(j)
      up_value_degraded = up_function(original_image,perturbed_image,i,j,alpha,beta,threshold,sigma)
      up_value_before_degraded = up_function(original_image,current_image,i,j,alpha,beta,threshold,sigma)
      #print(up_value_degraded)
      #print(up_value_before_degraded)
      if (up_value_degraded - up_value_before_degraded) <= 0:
        new_image[i][j] = perturbed_image[i][j]
        #print(new_image[i][j])
      elif (up_value_degraded - up_value_before_degraded) > 0:
        if np.exp(-(up_value_degraded - up_value_before_degraded)/temp) > np.random.uniform(0,1):
          new_image[i][j] = perturbed_image[i][j]
          #print(new_image[i][j])
        else:
          new_image[i][j] = current_image[i][j]
          #print(new_image[i][j])
  return new_image


#cv2.imshow('updated',gaussian_image)
#cv2.waitKey(0)         
#cv2.destroyAllWindows()
updated = image_update(presav_gray_image,gaussian_image,sigma,alpha, beta,threshold,init_temp,mean = 0)
#print(gaussian_image.shape)
print(gaussian_image[100])
print(updated[100])
cv2.imshow('updated',updated)
cv2.waitKey(0)         
cv2.destroyAllWindows()

#print(updated)

## This is the energy function that can be used to compute the energy of an image.
def energy(current_image,alpha,beta,threshold):
  row,col = current_image.shape
  energy = 0
  for i in range(1,row):
    for j in range(1,col):
      horizontal_similarity = (current_image[i][j] - current_image[i][j-1])**2*(1-horizontal_funct(current_image,i,j,threshold))
      vertical_similarity = (current_image[i][j] - current_image[i-1][j])**2 * (1-vertical_funct(current_image,i,j,threshold))
      alpha_term = alpha*(horizontal_similarity + vertical_similarity)
      beta_term = beta*(vertical_funct(current_image,i,j,threshold) + horizontal_funct(current_image,i,j,threshold))
      energy = energy + alpha_term + beta_term
  return energy

## Main function to run the Hybrid Tabu Process described in Patra's paper.

def main(original_image,current_image,sigma,alpha,beta,threshold,tabu_list,temp,mean = 0):
  updated_image = image_update(original_image,current_image,sigma,alpha, beta,threshold,temp,mean = 0)
  #print(updated_image)
  updated_energy = energy(updated_image,alpha,beta,threshold)
  #tabu_energy_list = {}
  if len(tabu_list) == 0:
    tabu_list[0] = updated_image
  else:
    tabu_elts = len(tabu_list)
    i = 1
    for p in tabu_list.keys():
      curr_tabu_energy = energy(tabu_list[p],alpha,beta,threshold)
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

original_image = presav_gray_image
current_image = gaussian_image
temp = init_temp
#current_image, tabu_list = main(original_image,current_image,sigma,alpha,beta,threshold,tabu_list,temp,mean = 0)
#print(current_image)
#cv2.imshow('updated',current_image)
#cv2.waitKey(0)         
#cv2.destroyAllWindows()

#for j in range(total_iterations):
# print(j)
  #print(current_image)
# current_image, tabu_list = main(original_image,current_image,sigma,alpha,beta,threshold,tabu_list,temp,mean = 0)
# temp = temp/(1 + np.log(1+j))
# print(temp)
# print(energy(current_image,alpha,beta,threshold))
  #print(tabu_list)
  #cv2.imshow('updated',current_image)
  #cv2.waitKey(0)         
  #cv2.destroyAllWindows()

#print(tabu_list)
#cv2.imshow('updated',current_image)
#cv2.waitKey(0)         
#cv2.destroyAllWindows()

"""