#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 16:30:47 2018

This script conatins all the necessary tools for bot camera distrotion and
perspective corrections.

@author: janko
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_CHESSBOARD_FOLDER = 'camera_cal'
DEFAULT_NX = 9
DEFAULT_NY = 6

DEFAULT_SRC_POINTS = np.float32(
            [[575, 464],
             [707, 464],
             [258, 683],
             [1047, 683]])

DEFAULT_DST_POINTS = np.float32(
            [[375, 0],
             [950, 0],
             [375, 700],
             [950, 700]])

DEFAULT_TEST_IMAGE = 'test_images/straight_lines1.jpg'

def calibrate_camera(folder=DEFAULT_CHESSBOARD_FOLDER, nx=DEFAULT_NX, ny=DEFAULT_NY):
    ''' Performs camera calibration on a set of chessboard images.
    '''
    imgs = os.listdir(folder)
    
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Prepare object points (x, y, z)
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    
    for filename in imgs:
        img = cv2.imread(os.path.join(folder, filename))
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If found, store corners    
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
    
    # Calibration
    [rows, cols] = gray.shape   # All calibration images have the same size
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       (cols, rows),
                                                       None,
                                                       None)
    
    return mtx, dist
    

def show_camera_distortion(folder=DEFAULT_CHESSBOARD_FOLDER, nx=DEFAULT_NX, ny=DEFAULT_NY, img=None):
    ''' Shows example of camera calibration.
    '''
    mtx, dist = calibrate_camera(folder, nx, ny)
    imgs = os.listdir(folder)
    # Undistort
    if img is None:
        img = cv2.imread(os.path.join(folder, imgs[1]))
    else:
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    plt.subplot(121), plt.imshow(img)
    plt.subplot(122), plt.imshow(dst)




def compute_perspective_distortion(src_points=DEFAULT_SRC_POINTS, dst_points=DEFAULT_DST_POINTS):
    '''Computes transform matrices for perspective distortion between two sets of points.'''
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)    
    return M, Minv
    

def show_perspective_calibration(src_points=DEFAULT_SRC_POINTS,
                                 dst_points=DEFAULT_DST_POINTS,
                                 img=DEFAULT_TEST_IMAGE):
    ''' Shows example of perspective distortion correction.
    '''

    plt.figure()
    M, Minv = compute_perspective_distortion()
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    [rows, cols, dims] = img.shape
    plt.subplot(121)
    plt.imshow(img)
    for ii in range(0, src_points.shape[0]):
        plt.plot(src_points[ii,0], src_points[ii,1], 'ro')    
        
    warped = cv2.warpPerspective(img, M, (cols, rows), flags=cv2.INTER_LINEAR)
    
    plt.subplot(122)
    plt.imshow(warped)
    for ii in range(0, src_points.shape[0]):
        aux = np.hstack((src_points[ii,:], 1.0))
        aux = np.matmul(M, aux)
        aux = aux[0:2]/aux[-1]
        plt.plot(aux[0], aux[1], 'bo')
    
    
#show_perspective_calibration()
