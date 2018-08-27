#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 20:31:17 2018

@author: janko
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass

DEFAULT_MIN_SCORE = 280
DEFAULT_SEARCH_RATIO = 1.0
DEFAULT_LEFT_RIGHT_SPLIT = 0.5
DEFAULT_OVERLAP = 0

def get_center(binary_img,
               kernel_width,
               min_score=DEFAULT_MIN_SCORE,
               reference=None,
               search_ratio=DEFAULT_SEARCH_RATIO):
    ''' Identifies the "hottest" region in a binary image.
    
    Computes center of mass of the window that contains the highest amount of
    true pixels. Uses a sliding window (kernel) whose height corresponds
    to the image height.
    
    Parameters
    ----------
    binary_img : numpy array
        binary image
        
    kernel_width : int
        width of the sliding window
        
    min_score : int
        minimum number of hot pixels contained by a window to actually
        determine its centre
        
    reference : float
        horizontal coordinate of the reference point whose neghbourhood is
        to be searched. Defaults to None which means the entire image width is
        searched.
    
    search_ratio : float
        ratio of the kernel width that is searched around the reference point.
        Applicable only when reference is not None.
        
    Returns
    -------
    center : numpy array
        (rows, cols) of the center of mass
        
    th : float
        number of hot pixels under centered window, i.e., the score of
        the hottest window
    '''
    th = 0    
    [rows, cols] = binary_img.shape
    
    # Set search region (horizontally)
    if reference is not None:
        start = int(np.max((reference - kernel_width * search_ratio, 0)))
        stop = int(np.min((reference + kernel_width * search_ratio, cols - kernel_width)))
    else:
        start = 0
        stop = cols - kernel_width
        
    final_ratio = None
    center = None
    for offset in range(start, stop):
        blk = binary_img[:, offset:offset+kernel_width]        
        score = np.sum(blk)
        if score > th and score > min_score:
            center_temp = np.asarray(center_of_mass(blk))  # rows, cols
            coords = np.where(blk == True)
            r_coords = np.asarray(coords[0])
            c_coords = np.asarray(coords[1])
            ratio = np.mean(np.abs(r_coords - center_temp[0])) / np.mean(np.abs(c_coords - center_temp[1]))
#            print score
#            print ratio
            if ratio < 1.0 or score > 0.75*blk.shape[0]*blk.shape[1]:
                continue
            th = score
            center = center_temp
            final_ratio = ratio
            center[1] = center[1] + offset  # account for the current window location
    #print final_ratio
    return center, th
    

def get_centers_from_slice(binary_img,
                           kernel_width,
                           min_score=DEFAULT_MIN_SCORE,
                           reference=None,
                           search_ratio=DEFAULT_SEARCH_RATIO,
                           left_right_split=DEFAULT_LEFT_RIGHT_SPLIT):
    ''' Identifies the left and right "hottest" regions in a binary image.
    
    Computes center of mass of the window that contains the highest amount of
    true pixels in the left and right parts of the image. Uses a sliding
    window (kernel) whose height corresponds to the image height.
    
    Parameters
    ----------
    binary_img : numpy array
        binary image
        
    kernel_width : int
        width of the sliding window
        
    min_score : int
        minimum number of hot pixels contained by a window to actually
        determine its centre
        
    reference : tuple
        (cols_left, cols_right) coordinates of the reference point whose
        neghbourhood is to be searched. Defaults to None which means the entire
        image width is searched.
    
    search_ratio : float
        ratio of the kernel width that is searched around the reference point.
        Applicable only when reference is not None.
        
    left_right_split : float
        ratio (with respect to image width) that separates left and right image
        regions (starting from left)
        
    Returns
    -------
    center_left : numpy array
        (rows, cols) of the hottest left center of mass
        
    center_right : numpy array
        (rows, cols) of the hottest right center of mass
        
    th : float
        number of hot pixels under centered window, i.e., the score of
        the hottest window
    '''
    # Generate masks for left and right image regions
    mask_left = np.zeros_like(binary_img)
    mask_left[:, 0:int(mask_left.shape[1]*(1-left_right_split))] = 1
    mask_left = (mask_left == 1)
    mask_right = (mask_left == 0)
    
    if reference is None:
        reference_left = None
        reference_right = None
    else:
        reference_left = reference[0]
        reference_right = reference[1]
    
    #Compute centers
    center_left, score_left = get_center(binary_img * mask_left,
                                         kernel_width,
                                         min_score,
                                         reference=reference_left,
                                         search_ratio=search_ratio)
    center_right, score_right = get_center(binary_img * mask_right,
                                           kernel_width,
                                           min_score,
                                           reference=reference_right,
                                           search_ratio=search_ratio)
    #plt.figure(), plt.imshow(binary_img * mask_left)
    return center_left, center_right


def get_centers(binary_img,
                kernel_width,
                num_slices,
                overlap=DEFAULT_OVERLAP,
                min_score=DEFAULT_MIN_SCORE,
                inter_reference=None,
                search_ratio=DEFAULT_SEARCH_RATIO,
                left_right_split=DEFAULT_LEFT_RIGHT_SPLIT):
    ''' Computes left and right lines positions for each slice.
    
    Parameters
    ----------
    binary_img : numpy array
        binary image
        
    kernel_width : int
        width of the sliding window
        
    num_slices : int
        number of vertical splits (slices) binary_img is to be divided into
        
    min_score : int
        minimum number of hot pixels contained by a window to actually
        determine its centre
        
    overlap : int
        overlap (in pixels) between two consecutive slices. Not implemented.
        
    inter_reference : tuple
        (cols_left, cols_right) coordinates of the reference point whose
        neghbourhood is to be searched. Defaults to None which means the entire
        image width is searched.
    
    search_ratio : float
        ratio of the kernel width that is searched around the reference point.
        Applicable only when reference is not None.
        
    left_right_split : float
        ratio (with respect to image width) that separates left and right image
        regions (starting from left)
        
    Returns
    -------
    center_left : numpy array
        (rows, cols) of the hottest left center of mass
        
    center_right : numpy array
        (rows, cols) of the hottest right center of mass            
    ''' 
    # Get height of image slices
    [rows, cols] = binary_img.shape
    height = rows/num_slices
    
    centers_left = []
    centers_right = []
    # Intra-frame references    
    reference_left = None
    reference_right = None

    for ii in range(0, num_slices):

        if inter_reference is not None:        
            reference_left = inter_reference[ii,0]
            reference_right = inter_reference[ii,1]
        
        # From bottom to top        
        binary_slice = binary_img[rows-(ii+1)*height:rows-ii*height]
        center_left, center_right = get_centers_from_slice(
                                binary_slice,
                                kernel_width,
                                min_score,
                                reference=(reference_left, reference_right),
                                search_ratio=search_ratio,
                                left_right_split=left_right_split)

        if center_left is not None:
            center_left[0] = center_left[0] + rows-(ii+1)*height
            if inter_reference is None:
                reference_left = center_left[1]
        
        if center_right is not None:
            center_right[0] = center_right[0] + rows-(ii+1)*height
            if inter_reference is None:
                reference_right = center_right[1]                    

        centers_left.append(center_left)
        centers_right.append(center_right)
        
        #print 'ref: ' + str(reference_left) + ' \t cen: ' + str(center_left[1])        
        
        # Remove Nones
        #centers_left = [x for x in centers_left if x is not None]
        #centers_right = [x for x in centers_right if x is not None]
    #return np.asarray(centers_left), np.asarray(centers_right)
    return centers_left, centers_right
        

def plot_centers(centers_left, centers_right):
    centers_left = np.asarray(centers_left)
    centers_right = np.asarray(centers_right)
    plt.plot(centers_left[:,1], centers_left[:,0], 'ro')
    plt.plot(centers_right[:,1], centers_right[:,0], 'ro')
    

def plot_fit_lines(centers_left, centers_right, img_shape):
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fit = np.polyfit(centers_left[:,0], centers_left[:,1], 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    
    right_fit = np.polyfit(centers_right[:,0], centers_right[:,1], 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    plt.plot(left_fitx, ploty, color='red')
    plt.plot(right_fitx, ploty, color='red')

