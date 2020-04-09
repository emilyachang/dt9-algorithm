#!/usr/bin/env python3
import cv2
import numpy as np

def draw_circles(img, n=15):
    
    lam = 670; #nm
    pixel_size = 0.5;
    fs = 1/pixel_size;
    
#     low = -fs/2;
#     high = fs/2 * ((nx-2)/nx);
    
    nx, ny = np.shape(img);
    low = 0;
    high = fs/2 * ((nx-2)/nx) + fs/2;
    
    x = np.random.choice(nx, n).reshape((n, 1));
    y = np.random.choice(ny, n).reshape((n, 1));
    
    r = np.random.uniform(0.8, 1.5, n).reshape((n, 1));
    
    phase = np.random.uniform(0.25, 0.75, n) / (2*np.pi);
    phase = phase.reshape((n, 1));
    
    M = np.concatenate((x, y, np.round(r / pixel_size).astype(int), phase), axis=1);
    M = np.concatenate((M, np.exp(M[:,2] * 2j * np.pi).reshape(n, 1)), axis=1);
    
    for i in range(n):
        cv2.circle(img, (y[i], x[i]), radius=(np.round(r[i] / pixel_size)).astype(int), color=255, thickness=-1);
    
    img[img == 0] = 1;

    ft = np.fft.fft2(img);
    print(np.log10(abs(ft)).shape, np.max(np.log10(abs(ft))), np.min(np.log10(abs(ft))));
    
    return img, (np.log10(abs(ft)) - np.min(np.log10(abs(ft))))/(np.max(np.log10(abs(ft))) - np.min(np.log10(abs(ft)))) * 255;

def main():
    img = np.zeros((1000, 1000));
    cv2.imwrite('base.png', img);
    img = cv2.imread('base.png', cv2.IMREAD_COLOR);
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0

    img, fourier = draw_circles(img)
    
    img = cv2.imwrite('circles_drawn.png', img);
    fourier = cv2.imwrite('fourier.png', fourier)
    
main();