#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 14:12:38 2018

This scripts performs the whole pipeline for line detection. It receives a video
file and writes the output (frame by frame) into image folder.

@author: janko
"""

import os
import cv2
import pickle
import numpy as np
import thresholding as th
import matplotlib.pyplot as plt
from distortion import calibrate_camera, compute_perspective_distortion
from detection import get_centers, plot_centers, plot_fit_lines
from lines import Line
from tools import write_text_on_image, num2str

video_file = 'project_video.mp4'
DEFAULT_NUM_SLICES = 9  # Divison of warped binary mask into horizontal slices
kernel_width = 50       # Sliding window for line detection

# Distortion correction (skip it if already calculated)
if os.path.exists('data/distortion_correction.pckl'):
    f = open('data/distortion_correction.pckl', 'r')
    mtx = pickle.load(f)
    dist = pickle.load(f)
    M = pickle.load(f)
    Minv = pickle.load(f)
else:
    mtx, dist = calibrate_camera()
    M, Minv = compute_perspective_distortion()
    f = open('data/distortion_correction.pckl', 'wb')
    pickle.dump(mtx, f)
    pickle.dump(dist, f)
    pickle.dump(M, f)
    pickle.dump(Minv, f)
    
# Line objects to keep track of the detected data
leftLine = Line(niterations=3, height=720, width=1280, M=M)
rightLine = Line(niterations=3, height=720, width=1280, M=M)


cap = cv2.VideoCapture(video_file)
# Variables for inter-prediction filter. Initialised to None since for the very
# first frame there is no temporal information available.
inter_reference = None
plotyy = None
cnt = 0

while(cap.isOpened()):
    ret, img = cap.read()
    
    # Correct for camera distortion    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.undistort(img, mtx, dist, None, mtx)
    [rows, cols, dims] = img.shape
    
    # Use bilateral filtering to filter out noise but keeping the edges sharp
    blur = cv2.bilateralFilter(img,9,75,75)
    temp = cv2.warpPerspective(blur, M, (cols, rows), flags=cv2.INTER_LINEAR)
    
    # Get binary mask
    # As a reference for the adaptive thresholding we use the expected road surface
    # (this surface can be applied by applying direct and inverse perspective distortion
    # transform since after this operation everything outside the expected road surface
    # will be padded to zero).
    ref_img = cv2.warpPerspective(temp, Minv, (cols, rows), flags=cv2.INTER_LINEAR)    
    maskw = th.adaptive_white_thresh(blur, ref_img, num_slices=20, thresh_offset=50)
    masky = th.adaptive_yellow_thresh(blur, ref_img, num_slices=20, thresh_mult=1.50)
    mask = masky + maskw
    mask = mask > 0
    
    # Warp the mask
    temp = np.dstack((mask, mask, mask)).astype(np.uint8)*255
    warped = cv2.warpPerspective(temp, M, (cols, rows), flags=cv2.INTER_LINEAR)
    warped = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    warped = warped > 0
    
    # Find line points    
    centers_left, centers_right = get_centers(warped, kernel_width,
                                              num_slices=DEFAULT_NUM_SLICES,
                                              inter_reference=inter_reference)        
    
    # Inter frame references for line points (if available)
    if inter_reference is None:
        centers_left = [x for x in centers_left if x is not None]
        centers_right = [x for x in centers_right if x is not None]
        weights_left = np.ones(len(centers_left)).astype(np.float32)
        weights_right = np.ones(len(centers_right)).astype(np.float32)
        
    # If a line point is not found use the reference from the previous frame
    else:        
        weights_left = np.ones(DEFAULT_NUM_SLICES).astype(np.float32)
        weights_right = np.ones(DEFAULT_NUM_SLICES).astype(np.float32)
        for ii in range(0, len(centers_left)):
            if centers_left[ii] is None:
                centers_left[ii] = np.array([plotyy[ii], inter_reference[ii,0]])
                weights_left[ii] = 1.0
            if centers_right[ii] is None:
                centers_right[ii] = np.array([plotyy[ii], inter_reference[ii,1]])
                weights_right[ii] = 1.0
    centers_left = np.asarray(centers_left)
    centers_right = np.asarray(centers_right)
    
    # Obtain the second order polynomial that fits the detected line points
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fit = np.polyfit(centers_left[:,0], centers_left[:,1], 2, w=weights_left)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(centers_right[:,0], centers_right[:,1], 2, w=weights_right)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Update the line objects
    leftLine.update(left_fit, left_fitx, ploty, centers_left)
    rightLine.update(right_fit, right_fitx, ploty, centers_right)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Using the best fit (moving average) for the polynomial coefficients
    pts_left = np.array([np.transpose(np.vstack([leftLine.bestx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightLine.bestx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)    

    # Print Curvature and Offset data to each frame   
    curvature = np.mean((leftLine.radius_of_curvature, rightLine.radius_of_curvature))
    curvature_label = 'Curvature: ' + num2str(curvature) + 'm'
    
    offset = leftLine.line_base_pos + rightLine.line_base_pos    
    if offset < 0:
        offset_label = 'Vehicle is ' + num2str(-offset, 2) + 'm left from centre'
    else:
        offset_label = 'Vehicle is ' + num2str(offset, 2) + 'm right from centre'
    
    write_text_on_image(result, curvature_label, location=(50,50))
    write_text_on_image(result, offset_label, location=(50,100))
    cv2.imwrite('images/result_' + str(cnt) + '.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))    
    cnt = cnt + 1

    # Find line centers (for left and right lines) for each slice as reference
    # points for the current frame
    plotyy = np.arange(0, warped.shape[0], warped.shape[0]/DEFAULT_NUM_SLICES) + warped.shape[0]/DEFAULT_NUM_SLICES/2.0
    plotyy = np.flip(plotyy, axis=0)
        
    inter_reference_left = leftLine.best_fit[0]*plotyy**2 + leftLine.best_fit[1]*plotyy + leftLine.best_fit[2]
    inter_reference_right = rightLine.best_fit[0]*plotyy**2 + rightLine.best_fit[1]*plotyy + rightLine.best_fit[2]
    
    # Update the inter_reference so it can be used by next frame
    inter_reference_left = np.reshape(inter_reference_left, (DEFAULT_NUM_SLICES, 1))
    inter_reference_right = np.reshape(inter_reference_right, (DEFAULT_NUM_SLICES, 1))
    inter_reference = np.hstack((inter_reference_left, inter_reference_right))    

cap.release()
