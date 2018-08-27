#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 18:21:01 2018

@author: janko
"""

import cv2
import numpy as np

def check_line_separation(left_fit, right_fit, ploty):
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]    
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    separation = right_fitx - left_fitx    
    return np.max(separation) - np.min(separation)


def num2str(x, num_decimals=2):
    str_format = '%0.' + str(num_decimals) + 'f'
    return str_format % x


def write_text_on_image(img, text, location):
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX    
    fontScale              = 1.5
    fontColor              = (255,255,255)
    lineType               = 2
    
    cv2.putText(img, text, 
        location, 
        font, 
        fontScale,
        fontColor,
        lineType)
    return img

def warp_point(point, matrix):
    point = np.hstack((point, 1.0))
    point = np.matmul(matrix, point)
    return point[0:2]/point[-1]