#!/usr/bin/env python3
import cv2
import numpy as np

def reconstruct(img, it=5):
    
    nx, ny = np.shape(img);
    
    # random initializations
    H = img;
    W = np.random.rand(nx,ny)
    mu = np.zeros((nx,ny))
    X = np.zeros((nx,ny))

    for i in it:
        
        
        
        
    
    return 

def main():

    img = cv2.imread('image.png', cv2.IMREAD_COLOR);
    
main();