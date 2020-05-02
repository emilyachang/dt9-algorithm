#!/usr/bin/env python3
import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt
from skimage import transform as skt


def WAS_xfer(z, wave_len, img_size, pixel_dim):
    """Function to return the WAS transfer function at a given depth z.

    This function returns the coefficients of the transfer function in the
    Fourier domain for a fft of a given size.

    Args:
        z - The distance to apply the transfer function over.
        wave_len - The wavelength of the light.
        img_size - The dimensions of the transfer function to return.
        pixel_dim - A scalar describing the dimenion of the pixel. (Pixels are
        assumed to be square.).

    Returns:
        T_out - The transfer function coefficients (in Fourier domain).
    """

    n, m = img_size[0:2]
    sx = wave_len / (pixel_dim * m)
    sy = wave_len / (pixel_dim * n)
    iy, ix = np.ogrid[0:1 + n // 2, 0:1 + m // 2]
    x = (sx * ix) ** 2
    y = (sy * iy) ** 2
    del ix, iy
    # calculate 1 quadrant
    Kj_Z = z * 2.0j * np.pi / wave_len
    tmp = np.exp(Kj_Z * np.sqrt(1.0 - x - y))
    # and mirror
    tmp = np.concatenate((tmp[:, :-1], tmp[:, :0:-1]), axis=1)
    # and copy
    T_out = np.zeros((n, m), dtype='complex128')

    T_out[:n // 2, :] = tmp[:-1, :]
    T_out[n // 2:, :] = tmp[:0:-1, :]

    return T_out


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

def circles(circle_count):
    """Creates images with range of user specified circles.

    This function saves images with the number of circles specified by circle count
    at randomly selected locations into output .png files.

    Args:
        circle_count - An array with the number of circles to be drawn

    """
    # test range of particle count
    for num_circles in circle_count:
        
        # create base black image
        img = np.zeros((1000, 1000));
        cv2.imwrite('base.png', img);
        img = cv2.imread('base.png', cv2.IMREAD_COLOR);
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
        
        # draw circles
        img = draw_circles(img, num_circles)
        img = cv2.imwrite('circles_drawn%02.f.png' % (num_circles), img);


def createImage(I, z, pixsize):
    """Import ground truth image and return array containing hologram and reconstruction.

    This function returns arrays representing the images of the hologram and reconstruction
    for plotting ease given an input image at a specific distance from the image sensor.

    Args:
        I - The input ground truth image.
        z - The focal depth to reconstruct at.
        pixsize - The dimension of the image sensor pixels.

    Returns:
        H - The simulated hologram for the input image.
        I_down - The 2,2 downsampled hologram
        I_down2 - The 4,4 downsampled hologram
    """

    # I[I==255] = np.exp(np.pi*1.0j)
    I = np.pad(I, (100, 100))
    I[I == 0] = 1
    I[I > 1] = np.exp(np.pi * 1.0j)

    # Make the hologram by projecting to the image plane
    T = WAS_xfer(z, 405e-9, I.shape, pixsize)
    H = np.abs(ifft2(T * fft2(I)))

    # Scale the hologram to the range [0,16]
    H = H - np.min(H)
    H = 16 * H / np.max(H)

    # DOWNSAMPLING ! ------------------------------------------------------------------------------

    # create 1x downsampled image (1 um/pixel)
    Idown = skt.downscale_local_mean(H, (2, 2)) / 1
    Idown = Idown - np.min(Idown)
    Idown = 16 * Idown / np.max(Idown)

    # create 2x downsampled image (2 um/pixel)
    Idown2 = skt.downscale_local_mean(H, (4, 4)) / 1
    Idown2 = Idown2 - np.min(Idown2)
    Idown2 = 16 * Idown2 / np.max(Idown2)

    return H, Idown, Idown2

if __name__ == '__main__':

    circle_count = [1, 5, 10, 15, 20, 25];

    circles(circle_count);

    for c in circle_count:

        I = cv2.imread('circles_drawn%02.f.png' % (c), cv2.IMREAD_GRAYSCALE)
        I = I / 1

        # sample over varying distances
        z_range = np.linspace(500e-6, 8000e-6, num=16)

        for z in z_range:

            # get hologram arrays from algorithm
            I_hol, Idown_hol, Idown2_hol = createImage(I, z, 0.5e-6)

            img = cv2.imwrite('output-images/%02.f-particles/%02.f_circles_hol_focaldepth_%04.0fmicrons_full.png' % (c, c, z*10e5), I_hol);
            img = cv2.imwrite('output-images/%02.f-particles/%02.f_circles_hol_focaldepth_%04.0fmicrons_down2.png' % (c, c, z * 10e5), Idown_hol);
            img = cv2.imwrite('output-images/%02.f-particles/%02.f_circles_hol_focaldepth_%04.0fmicrons_down4.png' % (c, c, z * 10e5), Idown2_hol);