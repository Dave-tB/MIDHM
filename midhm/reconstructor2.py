import numpy as np

from skimage.restoration import unwrap_phase
from scipy.fft import dct

class Reconstructor2:

  @staticmethod
  def reconstruct2(img,
                  lambda_=488e-9, 
                  dx=1.12e-6,
                  dy=1.12e-6,
                  z=4e-2,
                  d=7e-3,
                  numIteration=1,
                  verbose=False):


    Ny, Nx = img.shape
    
    imgH = (img - ref) / (np.sqrt(ref))

    # method of subtracting the mean
    ##imgH = img - np.mean(img)

    # spatial sampling
    nx = np.arange(-Nx/2, Nx/2, dtype=float)
    ny = np.arange(-Ny/2, Ny/2, dtype=float)

    # spatial coordinates x,y
    x = nx*dx
    y = ny*dy
    X, Y = np.meshgrid(x, y)

    # spectral sampling interval
    dfx = 1/(Nx*dx)
    dfy = 1/(Ny*dy)

    # Spectral coordinates fx, fy
    fx = nx*dfx
    fy = ny*dfy
    Fx, Fy = np.meshgrid(fx, fy)


    # The measured hologram amplitude is used for the an initial guess
    # Creating initial complex-valued field distribution in the detector plane
    phase = np.zeros((Ny, Nx), dtype=complex)
    
    # Fourier Spectrum of the Sample
    FimgH = np.fft.fftshift(np.fft.fft2(imgH))
    # Creating wave propagation term
    prop = np.exp((1j*2*np.pi*d)*np.sqrt(1/(Lambda**2) - Fx**2 - Fy**2))


    detField = imgH * np.exp(1j * phase)

    # Fourier transform of the processed hologram with the DC term
    FimgH = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(detField)))

    # reconstruction of the transmission function, t
    t = (np.fft.ifft2(np.fft.fftshift(FimgH*prop)))
    amp = np.abs(t)
    phi = -np.angle(t)
    absorptn = -np.log(amp)

    n = 5
    R_TI = amp
    R_TI_padded = np.hstack([np.vstack([R_TI, np.zeros((n-1,R_TI.shape[1]))]),\
                                              np.zeros((R_TI.shape[0]+n-1,n-1))])
    
    mu = np.sum(np.lib.stride_tricks.sliding_window_view(R_TI_padded, (n,n)), axis=(2,3))/n**2

    UnimgH = unwrap_phase(phi)
    UnimgH2 = Reconstructor.Unwrap_TIE_DCT_Iter(phi)
    T = (abs(R_TI)**2 - mu)**2
    T_padded = np.hstack([np.vstack([T, np.zeros((n-1,T.shape[1]))]),\
                                        np.zeros((T.shape[0]+n-1,n-1))])
    
    V = np.sum(np.lib.stride_tricks.sliding_window_view(T_padded, (n,n)), axis=(2,3))/n**2
    
    tau = 0.002    # threshold
    mask = (V > tau).astype(int)
    R_S = R_TI*mask
    
    z = d
    G = np.exp((1j*2*np.pi*z)*np.sqrt(1/(Lambda**2) - Fx**2 - Fy**2))
    FimG = np.fft.fftshift(np.fft.fft2(R_S))
    t2 = (np.fft.ifft2(np.fft.fftshift(FimG*G)))
    amp2 = (abs(t2))                   # amplitude
    phi2 = np.angle(-t2)              # wrapped phase
    UnimgH3 = -unwrap_phase(phi2)     # unwrapped phase
    
    return t2, amp2, phi2, FimgH, UnimgH, UnimgH2, UnimgH3

  #
  # SEEMS NOT BEING USED
  #

  @staticmethod
  def Unwrap_TIE_DCT_Iter(phase_wrap):
    phi1 = Reconstructor.unwrap_TIE(phase_wrap)
    phi1 = phi1 + np.mean(phase_wrap) - np.mean(phi1) # adjust piston
    K1 = np.round((phi1 - phase_wrap) / (2 * np.pi)) # calculate integer K
    phase_unwrap = phase_wrap + 2 * K1 * np.pi
    residue = np.unwrap(phase_unwrap - phi1)
    phi1 = phi1 + Reconstructor.unwrap_TIE(residue)
    phi1 = phi1 + np.mean(phase_wrap) - np.mean(phi1) # adjust piston
    K2 = np.round((phi1 - phase_wrap) / (2 * np.pi)) # calculate integer K
    phase_unwrap = phase_wrap + 2 * K2 * np.pi
    residue = np.unwrap(phase_unwrap - phi1)
    N = 0
    c = 0
    while np.sum(np.abs(K2 - K1)) > 0 and c < 2:
        K1 = K2
        phic = Reconstructor.unwrap_TIE(residue)
        phi1 = phi1 + phic
        phi1 = phi1 + np.mean(phase_wrap) - np.mean(phi1) # adjust piston
        K2 = np.round((phi1 - phase_wrap) / (2 * np.pi)) # calculate integer K
        phase_unwrap = phase_wrap + 2 * K2 * np.pi
        residue = np.unwrap(phase_unwrap - phi1)
        N += 1
        c += 1
    return phase_unwrap, N

  @staticmethod
  def unwrap_TIE(phase_wrap):
    psi = np.exp(1j * phase_wrap)



    edx = np.hstack([np.zeros([psi.shape[0], 1]), 
                     Reconstructor.wrapToPi(np.diff(psi, axis=1)),
                     np.zeros([psi.shape[0], 1])])

    edy = np.vstack([np.zeros([1, psi.shape[1]]), 
                     # np.unwrap(np.diff(psi, axis=0), axis=0),  ## <<< not good
                     # unwrap_phase(np.diff(psi, axis=0)) ## <<< maybe
                     Reconstructor.wrapToPi(np.diff(psi, axis=0)),
                     np.zeros([1, psi.shape[1]])])

    lap = np.diff(edx, axis=1) + np.diff(edy, axis=0) # calculate Laplacian using the finite difference
    rho = np.imag(np.conj(psi) * lap) # calculate right hand side of Eq.(4) in the manuscript
    phase_unwrap = Reconstructor.solvePoisson(rho)
    return phase_unwrap

  @staticmethod
  def solvePoisson(rho):
    # solve the poisson equation using DCT
    # dctRho = np.fft.dct2(rho, norm='ortho')
    dctRho = dct(rho, norm='ortho')
    N, M = rho.shape
    I, J = np.meshgrid(np.arange(M), np.arange(N))
    dctPhi = dctRho / 2 / (np.cos(np.pi*I/M) + np.cos(np.pi*J/N) - 2)
    dctPhi[0, 0] = 0 # handling the inf/nan value
    # now invert to get the result
    # phi = np.fft.idct2(dctPhi, norm='ortho')
    phi = dct(dctPhi, norm='ortho')
    return phi



  @staticmethod
  def wrapToPi(x):

    return unwrap_phase(x)


    ## Does not work?

    # FROM https://stackoverflow.com/a/71914752
    xwrap = np.remainder(x, 2 * np.pi)
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
    mask1 = x < 0
    mask2 = np.remainder(x, np.pi) == 0
    mask3 = np.remainder(x, 2 * np.pi) != 0
    xwrap[mask1 & mask2 & mask3] -= 2 * np.pi
    return xwrap
