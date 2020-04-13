""" A few functions for doing basic single-depth LFI reconstruction.

Based on the algorithm described in the paper:
    Haeffele, Stahl, Vanmeerbeeck, Vidal. "Efficient reconstruction of 
    holographic lens-free images by sparse phase recovery." MICCAI. 
    pp. 109-117. Springer, Cham, 2017.
    
    Apr 13, 2020
"""

import numpy as np
from scipy.fftpack import fft2, ifft2 
import cv2


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
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    #Make a basic phase image with a simple square of shifted phase in the middle
    I = cv2.imread('circles_drawn.png',cv2.IMREAD_GRAYSCALE)
    I[I==255] = np.exp(np.pi*1.0j)
    
    #Make the hologram by projecting to the image plane
    z = 500e-6
    T = WAS_xfer(z, 637e-9, I.shape, 1.12e-6)
    H = np.abs(ifft2(T*fft2(I)))
    
    #Scale the hologram to the range [0,16]
    H = H-np.min(H)
    H = 16*H/np.max(H)
    
    #Now do the reconstruction
    X, X_ll, mu, W = SPR_recon(H, z)
    
    #Make an image of the hologram and the reconstruction
    #Note that the reconstruction has the background illumination (mu)
    #subtracted out.    
    plt.figure()
    plt.imshow(H)
    plt.title('Hologram')
    
    plt.figure()
    plt.imshow(np.abs(X))
    plt.title('Reconstruction')
    plt.show()