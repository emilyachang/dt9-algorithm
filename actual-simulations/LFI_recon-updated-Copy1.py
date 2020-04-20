""" A few functions for doing basic single-depth LFI reconstruction.

Based on the algorithm described in the paper:
    Haeffele, Stahl, Vanmeerbeeck, Vidal. "Efficient reconstruction of 
    holographic lens-free images by sparse phase recovery." MICCAI. 
    pp. 109-117. Springer, Cham, 2017.
    
    Apr 13, 2020

Modified by DT9 Beep Boop Team to add additional functions."""


import numpy as np
from scipy.fftpack import fft2, ifft2 
import cv2
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
           
    n,m = img_size[0:2]      
    sx  = wave_len / (pixel_dim * m)
    sy  = wave_len / (pixel_dim * n)
    iy, ix = np.ogrid[0:1+n//2,0:1+m//2]        
    x   = (sx * ix)**2
    y   = (sy * iy)**2
    del ix,iy
    # calculate 1 quadrant
    Kj_Z   = z * 2.0j * np.pi / wave_len
    tmp    = np.exp(Kj_Z * np.sqrt(1.0 - x - y))
    # and mirror 
    tmp    = np.concatenate((tmp[:,:-1],tmp[:,:0:-1]),axis=1)
    # and copy
    T_out = np.zeros((n,m),dtype='complex128')
        
    T_out[:n//2,:] = tmp[  :-1,:]
    T_out[n//2:,:] = tmp[:0:-1,:]
        
    return T_out

def SPR_recon(H, z, lam=1.0, wave_len=637e-9, pixel_dim=1.12e-6, num_iter=10):
    """ Function to run the sparse phase recovery reconstruction algorithm at 1 depth.
    
    Solves a problem of the form:
        min_{X, mu, W} 0.5*|H.*W - mu - conv2(X,T)|_F^2 + lam*|X|_1 s.t. abs(W[i,j])==1
        
    Args:
        H - The input hologram images.
        z - The focal depth to reconstruct at.
        lam - The weight of the sparse regularization constant.
        wave_len - The wavelength of the illumination light.
        pixel_dim - The dimension of the image sensor pixels.
        num_iter - The number of iterations of the algorithm to run.
        
    Returns:
        X - The (complex valued) reconstructed image
        X_ll - The loss-less reconstructed image. (I.e., X_ll, plus the
        background term and phase term) can be used to perfectly recover the
        hologram.
        mu - The background constant.
        W - The estimated phase of the hologram.
    """
    
    #Get the transfer function
    T = WAS_xfer(z, wave_len, H.shape, pixel_dim)
    
    X = np.zeros(H.shape, dtype = 'complex128')
    W = np.ones_like(X)
    mu = np.zeros(1, dtype = 'complex128')
    
    #First get the hologram back projected to the image plane.
    TdHw = ifft2(np.conj(T)*fft2(H*W))
    
    #Now run the main loop
    for i in range(num_iter):
        
        #Estimate the background term
        mu = np.mean(TdHw-X)/np.conj(T[0,0])
        
        #Estiamte the phase of the hologram
        W = ifft2(T*fft2(X))
        W = W+mu
        W = W/np.abs(W+1e-16)
        
        #Update the back projection of the hologram (+phase) to the focal plane
        TdHw = ifft2(np.conj(T)*fft2(H*W))
        
        #Update the reconstructed image
        X_ll = TdHw-mu*np.conj(T[0,0])
        X = X_ll/(np.abs(X_ll)+1e-16)*np.maximum(np.abs(X_ll)-lam,0)
    
    return X, X_ll, mu, W

def createImage(I,z,pixsize):
    """Import ground truth image and return array containing hologram and reconstruction.
    
    This function returns arrays representing the images of the hologram and reconstruction
    for plotting ease given an input image at a specific distance from the image sensor.
    
    Args:
        I - The input ground truth image.
        z - The focal depth to reconstruct at.
        pixsize - The dimension of the image sensor pixels.
        
    Returns:
        H - The simulated hologram for the input image.
        H_recon - The absolute value of the reconstructed image.
    """
    
    #I[I==255] = np.exp(np.pi*1.0j)
    I[I>1] = np.exp(np.pi*1.0j)

    #Make the hologram by projecting to the image plane
    T = WAS_xfer(z, 405e-9, I.shape, pixsize)
    H = np.abs(ifft2(T*fft2(I)))
    
    #Scale the hologram to the range [0,16]
    H = H-np.min(H)
    H = 16*H/np.max(H)
    H = (H-np.min(H))/(np.max(H)-np.min(H));

    #Now do the reconstruction; reconstruction has the background illumination (mu) subtracted out
    X, X_ll, mu, W = SPR_recon(H, z, wave_len=405e-9, pixel_dim=pixsize)
    H_recon = np.abs(X)
    H_recon = (H_recon-np.min(H_recon))/(np.max(H_recon)-np.min(H_recon));
    
    # DOWNSAMPLING ! ------------------------------------------------------------------------------
    
    # create 1x downsampled image (1 um/pixel)
    Idown = skt.downscale_local_mean(H, (2,2))/1
    #Idown = cv2.resize(Idown, dsize=(1000,1000), interpolation=cv2.INTER_NEAREST) 
    Idown = (Idown-np.min(Idown))/(np.max(Idown)-np.min(Idown));
    Xdown, X_lldown, mudown, Wdown = SPR_recon(Idown, z, wave_len=405e-9, pixel_dim=pixsize*2)
    Idown_recon = np.abs(Xdown)
    Idown_recon = (Idown_recon-np.min(Idown_recon))/(np.max(Idown_recon)-np.min(Idown_recon));
    
    # create 2x downsampled image (2 um/pixel)
    Idown2 = skt.downscale_local_mean(H, (4,4))/1
    #Idown2 = cv2.resize(Idown2, dsize=(1000,1000), interpolation=cv2.INTER_NEAREST) 
    Idown2 = (Idown2-np.min(Idown2))/(np.max(Idown2)-np.min(Idown2));
    Xdown2, X_lldown2, mudown2, Wdown2 = SPR_recon(Idown2, z, wave_len=405e-9, pixel_dim=pixsize*4)
    Idown2_recon = np.abs(Xdown2)   
    Idown2_recon = (Idown2_recon-np.min(Idown2_recon))/(np.max(Idown2_recon)-np.min(Idown2_recon));

    return H, H_recon, Idown, Idown_recon, Idown2, Idown2_recon

def compareRecon(im1, im2):
    """Compare image intensity values between two images.
    
    Prints maximum value in each respective image and in the difference between the two images.
    Plots the difference between the two images.
    
    Args:
        im1 - First image.
        im2 - Second image.
        
    Returns:
        
    """
    
    print(np.max(im1), np.max(im2))
    print(np.max(np.abs(im1-im2)))
    
    #plt.figure()
    #plt.title('Comparing Reconstructions')
    #plt.imshow(np.abs(im1-im2), cmap='gray');
    

if __name__ == '__main__':
    
    # original image (0.5 um/pixel)
    I = cv2.imread('circles_drawn.png',cv2.IMREAD_GRAYSCALE)
    I = I/1
    
    # sample over varying distances
    z_range = np.linspace(500e-6, 8000e-6, num=16)

    for z in z_range:     
        plt.figure(figsize=(10,12))
        plt.suptitle('%4.0fum Focal Depth' % (z*10e5))
        
        # get hologram and reconstruction arrays from algorithm
        I_hol, I_recon, Idown_hol, Idown_recon, Idown2_hol, Idown2_recon = createImage(I,z,0.5e-6)
        
        # plot hologram and reconstruction of data
        plt.subplot(3,2,1)
        plt.imshow(I_hol, cmap='gray')
        plt.title('Hologram (0.5 um/pixel)')
        plt.colorbar()
        plt.subplot(3,2,2)
        plt.imshow(I_recon, cmap='gist_earth')
        plt.title('Reconstruction (0.5 um/pixel)')
        plt.colorbar()
        
        # plot hologram and reconstruction of downsampled data
        plt.subplot(3,2,3)
        plt.imshow(Idown_hol, cmap='gray')
        plt.title('Downsampled (2,2) Hologram (1 um/pixel)')
        plt.colorbar()
        plt.subplot(3,2,4)
        plt.imshow(Idown_recon, cmap='gist_earth')
        plt.title('Downsampled (2,2) Reconstruction (1 um/pixel)')
        plt.colorbar()
        
        # plot hologram and reconstruction of downsampled data
        plt.subplot(3,2,5)
        plt.imshow(Idown2_hol, cmap='gray')
        plt.title('Downsampled (4,4) Hologram (2 um/pixel)')
        plt.colorbar()
        plt.subplot(3,2,6)
        plt.imshow(Idown2_recon, cmap='gist_earth')
        plt.title('Downsampled (4,4) Reconstruction (2 um/pixel)')
        plt.colorbar()
        
        # save figure
        plt.savefig('output-images/output-focaldepth_%04.0fmicrons.png' % (z*10e5))
        
        # compare reconstruction intensities
        #compareRecon(I_recon, Idown_recon)
        
        print('done')
    
    #plt.show()
    
    
