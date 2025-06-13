"""iFish utils.

Company: VNPT-IT.
Filename: ifisheye.py.
Datetime: 10/04/2024.
Description: Utilities for applying fisheye effects to ordinary images. Inspired by https://github.com/Gil-Mor/iFish.git
"""
import cv2
import numpy as np
import os


def get_fish_xn_yn(source_x, source_y, radius, distortion):
    """
        Converting a pixel's coordinates to its corresponding coordinates in fisheye image
        - Params:
            source_x    : pixel's x-coordinate
            source_y    : pixel's y-coordinate
            radius      : pixel's distance from the image center
            distortion  : distortion coefficient

        - Returns:
            fish_x      : pixel's new x-coordinate 
            fish_y      : pixel's new y-coordinate
    """
    if 1 - distortion*(radius**2) == 0:
        fish_x = source_x 
        fish_y = source_y
    else:
        fish_x = source_x/(1 - distortion*(radius**2))
        fish_y = source_y/(1 - distortion*(radius**2))

    return fish_x, fish_y


def img_pad_square(img, pad_value=0):
    """
        Add padding to the image to make it become a squared image
        - Params:
            img         : the original image
            pad_value   : padding value
        
        - Returns:
            img         : padded image
    """
    height, width, channel = img.shape
    if width >= height:
        border_width = (width - height)//2
        img = cv2.copyMakeBorder(img, border_width, border_width, 0, 0, cv2.BORDER_CONSTANT, value=pad_value)
    else:
        border_width = (height - width)//2
        img = cv2.copyMakeBorder(img, 0, 0, border_width, border_width, cv2.BORDER_CONSTANT, value=pad_value)
    return img


def fish(img, distortion_coefficient):
    """
        Convert normal image to fisheye image using OpenCV remap for better performance
        - Params:
            img                     : the original image
            distortion_coefficient  : distortion coefficient (should be between 0-1)
        - Returns:
            dstimg                 : fisheye image
    """
    height, width = img.shape[:2]
    
    # Create coordinate matrices
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    # Normalize coordinates to [-1, 1]
    X = (2 * X - width) / width
    Y = (2 * Y - height) / height
    
    # Calculate radius
    R = np.sqrt(X**2 + Y**2)
    
    # Calculate new coordinates
    mask = (1 - distortion_coefficient * R**2) != 0
    X_new = np.zeros_like(X)
    Y_new = np.zeros_like(Y)
    
    X_new[mask] = X[mask] / (1 - distortion_coefficient * R[mask]**2)
    Y_new[mask] = Y[mask] / (1 - distortion_coefficient * R[mask]**2)
    
    # Convert back to pixel coordinates
    X_new = ((X_new + 1) * width / 2).astype(np.float32)
    Y_new = ((Y_new + 1) * height / 2).astype(np.float32)
    
    # Create remap matrices
    map_x = X_new
    map_y = Y_new
    
    # Apply remap
    dstimg = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    return dstimg


def reverse_fish_xn_yn(source_x, source_y, radius, distortion):
    """
        Converting a pixel's coordinates in fisheye image to its corresponding coordinates in the original image
        (The reverse function of get_fish_xn_yn)
        - Params:
            source_x    : pixel's x-coordinate
            source_y    : pixel's y-coordinate
            radius      : pixel's distance from the image center
            distortion  : distortion coefficient

        - Returns:
            fish_x      : pixel's new x-coordinate 
            fish_y      : pixel's new y-coordinate
    """
    if radius == 0:
        return source_x, source_y
    coefficient = (np.sqrt(1 + 4*distortion*(radius**2)) - 1)/(2*distortion*(radius**2))
    
    return source_x*coefficient, source_y*coefficient