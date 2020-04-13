#!/usr/bin/env python3
import cv2
import numpy as np

def draw_circles(img, n=15):
    
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

    # fourier transform
    ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)));
    ft2 = np.fft.fft2(img);
    
    print(np.log10(abs(ft)).shape, np.max(np.log10(abs(ft))), np.min(np.log10(abs(ft))));
    
    # fourier output for return
    fourierimg = (np.log10(abs(ft)) - np.min(np.log10(abs(ft))))/(np.max(np.log10(abs(ft))) - np.min(np.log10(abs(ft)))) * 255
    fourierimg2 = (np.log10(abs(ft2)) - np.min(np.log10(abs(ft2))))/(np.max(np.log10(abs(ft2))) - np.min(np.log10(abs(ft2)))) * 255
    # -------------------------------------

    # frequency coordinates
    kx = np.linspace(low,highx,nx)
    ky = np.linspace(low,highy,ny)
    
    # z distance sampling
    z_0 = np.random.uniform(800,2400);
    
    # transfer function
    trans = np.exp(z_0*1j*np.sqrt((2*np.pi*wv)**2 - kx**2 - ky**2))
    
    # intensity @ image sensor (convolution)
    I = trans * ft;
    I2 = trans * ft2;

    # inverse fourier to get image
    ift = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(I)));
    ift2 = np.fft.ifft2(I2);
    
    # image sensor output (I_z0(x,y))
    Iimg = (np.log10(abs(ift)) - np.min(np.log10(abs(ift))))/(np.max(np.log10(abs(ift))) - np.min(np.log10(abs(ift)))) * 255
    Iimg2 = (np.log10(abs(ift2)) - np.min(np.log10(abs(ift2))))/(np.max(np.log10(abs(ift2))) - np.min(np.log10(abs(ift2)))) * 255    
    # generate noise: N(0,[0.0125,0.03125])
    noise = np.random.normal(0,np.random.uniform(0.0125, 0.03125, 1), (nx, ny)) * 255
    
    # final image sensory output + noise (H(x,y))
    H = np.abs(Iimg) + noise
    H2 = np.abs(Iimg2) + noise
    
    return img, fourierimg, fourierimg2, H, H2;

def main():
    img = np.zeros((1000, 1000));
    cv2.imwrite('base.png', img);
    img = cv2.imread('base.png', cv2.IMREAD_COLOR);
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0

    img, fourier, fourier2, H, H2 = draw_circles(img)
    
    img = cv2.imwrite('circles_drawn.png', img);
    fourier = cv2.imwrite('fourier.png', fourier)
    fourier2 = cv2.imwrite('fourier2.png', fourier2)
    H = cv2.imwrite('image.png', H)
    H2 = cv2.imwrite('image2.png', H2)
    
main();