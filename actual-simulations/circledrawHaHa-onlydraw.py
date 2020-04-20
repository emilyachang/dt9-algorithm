#!/usr/bin/env python3
import cv2
import numpy as np

def draw_circles(img, n=15):
    """Draw circles to represent disk-shaped particles given a black grid. Returns image of drawn circles.
    
    Args:
        img - Array representing black grid.
        n - Number of circles to draw.
        
    Returns:
        img - Output image with circles drawn.
    """
    
    # declare parameters
    wv = 405; # laser wavelength (nm)
    pixel_size = 0.5;
    fs = 1/pixel_size;
    
    # size of imaging plane in image and frequency domain
    nx, ny = np.shape(img);
    low = -fs/2;
    highx = fs/2 * ((nx-2)/nx) ;
    highy = fs/2 * ((ny-2)/ny) ;
    
    # circle positioning and sizing
    x = np.random.choice(nx, n).reshape((n, 1));
    y = np.random.choice(ny, n).reshape((n, 1));
    r = np.random.uniform(0.8, 1.5, n).reshape((n, 1));
    
    # phase shift
    phase = np.random.uniform(0.25, 0.75, n) / (2*np.pi);
    phase = phase.reshape((n, 1));
    
    # complex optical wavefront I(x,y)
    M = np.concatenate((x, y, np.round(r / pixel_size).astype(int), phase), axis=1);
    M = np.concatenate((M, np.exp(M[:,2] * 2j * np.pi).reshape(n, 1)), axis=1);
    
    # draw circles
    for i in range(n):
        cv2.circle(img, (int(y[i]), int(x[i])), radius=int((np.round(r[i] / pixel_size))), color=255, thickness=-1);
    img[img == 0] = 1;
    
    return img;

def main():
    
    # test range of particle count
    for num_circles in [1,5,10,15,20,25]:
        
        # create base black image
        img = np.zeros((1000, 1000));
        cv2.imwrite('base.png', img);
        img = cv2.imread('base.png', cv2.IMREAD_COLOR);
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
        
        # draw circles
        img = draw_circles(img, num_circles)
        img = cv2.imwrite('circles_drawn%02.f.png' % (num_circles), img);
    
main();