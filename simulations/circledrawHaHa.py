#!/usr/bin/env python3
import cv2
import numpy as np

def draw_circles(img, n=5):
    nx, ny, channel = np.shape(img);
    x = np.random.choice(nx, 5);
    y = np.random.choice(ny, 5);
    for i in range(n):
        print(x[i], y[i])
        cv2.circle(img, (y[i], x[i]), radius=5, color=(0, 255, 0), thickness=-1);
    
    return img;

def main():
    img = np.zeros((1024, 1024));
    cv2.imwrite('base.png', img);
    img = cv2.imread('base.png', cv2.IMREAD_COLOR);
    img = draw_circles(img, 5)
    
    img = cv2.imwrite('circles_drawn.png', img);
    
main();