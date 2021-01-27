import numpy as np

# Discrete Fourier Transform functions used by YMAEDA_TOOLS

# frequency step size used by YMAEDA_TOOLS
df = 0.002441406 
# frequency half space used by YMAEDA_TOOLS
f = np.arange(0, df*2049, df) 
# frequency full space used by YMAEDA_TOOLS
F = np.arange(0, df*4096, df) 

def idft(X):
    """
    This DFT was constructed to match that used in YMAEDA_TOOLS and is
    different from the version used in MATLAB, SCIPY etc.
    This uses the traditional DFT algorithm which is very slow due to having
    2 for loops, and can definitely be made faster by using symmetry...
    """
    N = len(X)
    x = np.zeros(N, 'complex')
    for n in range(0, N, 1):
        for k in range(0, N, 1):
            x[n] = x[n] + X[k] * np.exp(-1j * 2 * np.pi * k * n / N)
    return x / N

def dft(X):
    """
    This DFT was constructed to match that used in YMAEDA_TOOLS and is
    different from the version used in MATLAB, SCIPY etc.
    This uses the traditional DFT algorithm which is very slow due to having
    2 for loops, and can definitely be made faster by using symmetry...
    """
    N = len(X)
    x = np.zeros(N, 'complex')
    for n in range(0, N, 1):
        for k in range(0, N, 1):
            x[n] = x[n] + X[k] * np.exp(1j * 2 * np.pi * k * n / N)
    return x

def timeshift(mest, D):
    """
    Time shift a Fourier transform by some integer value D
    for use on the non-complex arrays output by YMAEDA_TOOLS
    warning! Use this only on full range Fourier transforms...
    """
    N = len(mest)
    for k in range(N):
        W = -2 * np.pi * k * (-D) / N
        a = mest[k, 0]
        b = mest[k, 1]
        mest[k, 0] = a * np.cos(W) - b * np.sin(W)
        mest[k, 1] = a * np.sin(W) + b * np.cos(W)
    return mest
    
def timeshift_cplx(mest, D):
    """
    Time shift a Fourier transform by some integer value D
    for use on complex python arrays
    warning! Use this only on full range Fourier transforms...
    """
    N = len(mest)
    for k in range(N):
        W = -1j * 2 * np.pi * k * (-D) / N
        mest[k] = mest[k] * np.exp(W)
    return mest  

def exifft(M, DATLEN = 2049):
    """
    Extends the data to full frequency range and perform ifft
    for use on non-complex arrays output by YMAEDA_TOOLS
    """
    M = M[:, 0] + M[:, 1] * 1j #combine real and imag parts together
    #for 2**12=4096 frequency points, [1:2047]=conj([4095:2049]) for 2047 partners
    #points 0 and 2048 are the 2 points without complex conjugate partners.
    M2 = M[1:DATLEN - 1]
    M2 = np.flipud(M2) #flip the order due to [1:2047]=conj([4095:2049])
    M2 = np.conj(M2) #remember to take the complex conjugate!!!!!!
    #M2=conj(M2[::-1]) #[1:2047]=conj([4095:2049])
    M2 = np.hstack([M, M2])
    return idft(M2)

def exifft_cplx(M, DATLEN = 2049):
    """
    Extends the data to full frequency range and perform ifft
    for use on complex python arrays
    """
    M2 = M[1:DATLEN - 1]
    M2 = np.flipud(M2) #flip the order due to [1:2047]=conj([4095:2049])
    M2 = np.conj(M2) #remember to take the complex conjugate!!!!!!
    #M2=conj(M2[::-1]) #[1:2047]=conj([4095:2049])
    M2 = np.hstack([M, M2])
    return idft(M2)

def exifft_timeshift(M, D, DATLEN = 2049):
    """
    Extend and inverse Fourier transform for non-complex matrices
    as the output by winv from YMAEDA_TOOLS only contains the first half
    of the frequency range up to Nyquist frequency, the data has to be
    extended into the complex conjugate half before ifft is performed.
    """
    M = M[:, 0] + M[:, 1] * 1j #combine real and imag parts together to complex array
    M2 = M[1:DATLEN - 1]
    M2 = np.flipud(M2) #flip the order due to [1:2047]=conj([4095:2049])
    M2 = np.conj(M2) #remember to take the complex conjugate!!!!!!
    M2 = np.hstack([M, M2])
    for k in range(len(M2)):
        W = -1j * 2 * np.pi * k * (-D) / len(M2)
        M2[k] = M2[k] * np.exp(W)
    return idft(M2)

def exifft_cplx_timeshift(M, D, DATLEN = 2049):
    """
    Extend and inverse Fourier transform for complex arrays.
    """
    M2 = M[1:DATLEN - 1]
    M2 = np.flipud(M2) #flip the order due to [1:2047]=conj([4095:2049])
    M2 = np.conj(M2) #remember to take the complex conjugate!!!!!!
    M2 = np.hstack([M, M2])
    for k in range(len(M2)):
        W = -1j * 2 * np.pi * k * (-D) / len(M2)
        M2[k] = M2[k] * np.exp(W)
    return idft(M2)