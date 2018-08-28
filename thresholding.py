#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 19:05:54 2018

@author: janko
"""

import cv2
import numpy as np

EPSILON = 1.0

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    else:
        raise ValueError
    # Take the absolute value of the derivative or gradient
    sobel = np.abs(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8    
    sobel = (255.0*sobel/np.max(sobel)).astype(np.uint8)
    # Create the corresponding binary edge mask
    mask = np.logical_and(sobel >= thresh_min, sobel <= thresh_max)                
    return mask


def mag_thresh(img, sobel_kernel=3, thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude 
    sobel = np.sqrt(sobelx**2 + sobely**2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8    
    sobel = (255.0*sobel/np.max(sobel)).astype(np.uint8)
    # Create the corresponding binary edge mask
    mask = np.logical_and(sobel >= thresh_min, sobel <= thresh_max)
    return mask


def dir_thresh(img, sobel_kernel=3, thresh_min=0, thresh_max=np.pi/2):    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    angle = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary mask where direction thresholds are met
    mask = np.logical_and(angle >= thresh_min, angle <= thresh_max)
    # Return this mask as your binary_output image    
    return mask


def colour_thresh(img, channel='gray', thresh_min=0, thresh_max=255):
    if channel == 'red':
        img = img[:,:,0]
    elif channel == 'green':
        img = img[:,:,1]
    elif channel == 'blue':
        img = img[:,:,2]
    elif channel == 'gray':
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.mean(img, axis=2)
    elif channel == 'hue':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,0]
    elif channel == 'lightness':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,1]
    elif channel == 'saturation':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2]
        
    mask = np.logical_and(img >= thresh_min, img <= thresh_max)
    return mask
        
        
def adaptive_white_thresh(img, ref_img, num_slices, thresh_offset=50):
    ''' Performs adaptive thresholding on white colour.
    
    Divides the input image into several horizontal slices and uses the reference
    image to compute the threshold for each slice.
    
    Parameters
    ----------
    img : numpy array
        input image (RGB)
        
    ref_img : numpy array
        reference image (RGB), it is used to estimate the thresholds
        
    num_slices : int
        number of slices img and ref_img are divided into
        
    thresh_offset : float
        used to compute the threshold for each slice. The threshold is the mean
        pixel value (averaged over the three channels) plus thresh_offset
    '''
    r = img[:,:,0]    
    g = img[:,:,1]
    b = img[:,:,2]    
    
    [rows, cols, dims] = ref_img.shape
    mask = np.zeros_like(img[:,:,0])
    slice_height = int(rows/num_slices)
    for ii in range(0, num_slices):
        # Extract slice
        ref_img_slice_r = ref_img[ii*slice_height:(ii+1)*slice_height,:,0].astype(np.float32)
        ref_img_slice_g = ref_img[ii*slice_height:(ii+1)*slice_height,:,1].astype(np.float32)
        ref_img_slice_b = ref_img[ii*slice_height:(ii+1)*slice_height,:,2].astype(np.float32)
        
        # Only consider not purely black pixels (to avoid the influence of padded pixels)
        ref_mask_r = ref_img_slice_r == 0
        ref_mask_g = ref_img_slice_g == 0
        ref_mask_b = ref_img_slice_b == 0
        ref_mask = np.logical_and(ref_mask_r, ref_mask_b, ref_mask_g)
        ref_mask = (ref_mask == False)
        
        ref_mask_left = np.copy(ref_mask)
        ref_mask_right = np.copy(ref_mask)
        ref_mask_left[:,int(ref_mask.shape[1]/2):] = False
        ref_mask_right[:,:int(ref_mask.shape[1]/2)] = False
                        
        # Extract reference whiteness
        ref_mean_img_slice = (ref_img_slice_r + ref_img_slice_g + ref_img_slice_b)/3.0
        
        # Compute threshold for left and right side of the road
        th_left = np.mean(ref_mean_img_slice[ref_mask_left]) + thresh_offset
        th_right = np.mean(ref_mean_img_slice[ref_mask_right]) + thresh_offset
        
        if np.isnan(th_left):
            th_left = 1e6
        if np.isnan(th_right):
            th_right = 1e6
        
        r_slice = r[ii*slice_height:(ii+1)*slice_height]
        g_slice = g[ii*slice_height:(ii+1)*slice_height]
        b_slice = b[ii*slice_height:(ii+1)*slice_height]
        
        mask_left = np.logical_and(r_slice >= th_left, g_slice >= th_left, b_slice >= th_left) * ref_mask_left
        mask_right = np.logical_and(r_slice >= th_right, g_slice >= th_right, b_slice >= th_right) * ref_mask_right
        mask[ii*slice_height:(ii+1)*slice_height,:] = mask_left + mask_right
        
    return mask        



def adaptive_yellow_thresh(img, ref_img, num_slices, thresh_mult=1.5):
    ''' Performs adaptive thresholding on yellow colour.
    
    Divides the input image into several horizontal slices and uses the reference
    image to compute the threshold for each slice.
    
    Parameters
    ----------
    img : numpy array
        input image (RGB)
        
    ref_img : numpy array
        reference image (RGB), it is used to estimate the thresholds
        
    num_slices : int
        number of slices img and ref_img are divided into
        
    thresh_mult : float
        used to compute the threshold for each slice. The threshold is the mean
        yellowness ratio (R+G vs B) multiplied by thresh_mult
    '''
    r = img[:,:,0]    
    g = img[:,:,1]
    b = img[:,:,2]    
    
    [rows, cols, dims] = ref_img.shape
    mask = np.zeros_like(img[:,:,0])
    slice_height = int(rows/num_slices)
    for ii in range(0, num_slices):
        # Extract slice
        ref_img_slice_r = ref_img[ii*slice_height:(ii+1)*slice_height,:,0].astype(np.float32)
        ref_img_slice_g = ref_img[ii*slice_height:(ii+1)*slice_height,:,1].astype(np.float32)
        ref_img_slice_b = ref_img[ii*slice_height:(ii+1)*slice_height,:,2].astype(np.float32)
        
        # Only consider not purely black pixels (to avoid the influence of padded pixels)
        ref_mask_r = ref_img_slice_r == 0
        ref_mask_g = ref_img_slice_g == 0
        ref_mask_b = ref_img_slice_b == 0
        ref_mask = np.logical_and(ref_mask_r, ref_mask_b, ref_mask_g)
        ref_mask = (ref_mask == False)
        
        ref_mask_left = np.copy(ref_mask)
        ref_mask_right = np.copy(ref_mask)
        ref_mask_left[:,int(ref_mask.shape[1]/2):] = False
        ref_mask_right[:,:int(ref_mask.shape[1]/2)] = False
        
        # Extract yellowness ratio
        yellow_ratio = (ref_img_slice_r + ref_img_slice_g)/(ref_img_slice_b + EPSILON)
        yellow_ratio_left = np.mean(yellow_ratio[ref_mask_left])
        yellow_ratio_right = np.mean(yellow_ratio[ref_mask_right])
        
        # Compute threshold for left and right side of the road
        th_left = yellow_ratio_left * thresh_mult
        th_right = yellow_ratio_right * thresh_mult
        
        if np.isnan(th_left):
            th_left = 1e6
        if np.isnan(th_right):
            th_right = 1e6
        
        r_slice = r[ii*slice_height:(ii+1)*slice_height]
        g_slice = g[ii*slice_height:(ii+1)*slice_height]
        b_slice = b[ii*slice_height:(ii+1)*slice_height]
        
        yellowness = (r_slice + g_slice).astype(np.float32) / (b_slice.astype(np.float32) + EPSILON)
        
        mask_left = (yellowness > th_left) * ref_mask_left
        mask_right = (yellowness > th_right) * ref_mask_right
        mask[ii*slice_height:(ii+1)*slice_height,:] = mask_left + mask_right
        
    return mask   
