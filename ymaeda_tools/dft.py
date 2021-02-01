import numpy as np
from scipy.fft import fft, ifft

# Discrete Fourier Transform functions used by YMAEDA_TOOLS

# frequency step size used by YMAEDA_TOOLS
df = 0.002441406 
# frequency half space used by YMAEDA_TOOLS
f = np.arange(0, df * 2049, df) 
# frequency full space used by YMAEDA_TOOLS
F = np.arange(0, df * 4096, df) 

def dft(X, fast = True):
    """
    This Discrete Fourier Transform was constructed to match that used in YMAEDA_TOOLS.
    This uses the traditional DFT algorithm which is very slow due to having
    2 for loops, and can definitely be made faster by using symmetry...
    """        
    N = len(X)
    x = np.zeros(N, 'complex')
    
    if fast == False:
        # original code using full mathematical formula:
        for n in range(0, N, 1):
            for k in range(0, N, 1):
                x[n] = x[n] + X[k] * np.exp(1j * 2 * np.pi * k * n / N)
    elif fast == True:
        # new code exploiting symmetry and vectorization:
        K = np.arange(0, N, 1)
        for n in range(0, N, 1):
            x[n] = np.dot(X, np.exp(1j * 2 * np.pi * K * n / N))
    return x

def idft(X, fast = True):
    """
    This inverse Discrete Fourier Transform was constructed to match that used in YMAEDA_TOOLS.
    This uses the traditional DFT algorithm which is very slow due to having
    2 for loops, and can definitely be made faster by using symmetry...
    """
    N = len(X)
    x = np.zeros(N, 'complex')
       
    if fast == False:
        # original code using full mathematical formula: 
        for n in range(0, N, 1):
            for k in range(0, N, 1):
                x[n] = x[n] + X[k] * np.exp(-1j * 2 * np.pi * k * n / N)
    elif fast == True:        
        # new code exploiting symmetry and vectorization:
        K = np.arange(0, N, 1)
        for n in range(0, N, 1):
            x[n] = np.dot(X, np.exp(-1j * 2 * np.pi * K * n / N))
    return x / N

def timeshift(mest, D):
    """
    Time shift a Fourier transform by some integer shift value D.
    For use on the non-complex arrays output by YMAEDA_TOOLS.
    Version for complex python arrays: timeshift_cplx().
    Warning! Use this only on full range Fourier transforms...
    """
    N = len(mest)
    for k in range(N):
        W = -2 * np.pi * k * (-D) / N
        a = mest[k, 0]
        b = mest[k, 1]
        mest[k, 0] = a * np.cos(W) - b * np.sin(W) # time shift the real part
        mest[k, 1] = a * np.sin(W) + b * np.cos(W) # time shift the imag part
    return mest
    
def timeshift_cplx(mest, D):
    """
    Time shift a Fourier transform by some integer value D.
    For use on complex python arrays equivalent to the output by YMAEDA_TOOLS.
    Version for output by YMAEDA_TOOLS: timeshift().
    Warning! Use this only on full range Fourier transforms...
    """
    N = len(mest)
    for k in range(N):
        W = -1j * 2 * np.pi * k * (-D) / N
        mest[k] = mest[k] * np.exp(W)
    return mest  

def exidft(M, DATLEN = 2049):
    """
    Extend and inverse discrete Fourier transform.
    For output by YMAEDA_TOOLS.
    Extends the data from half frequency space to full frequency range and perform idft.
    For use on non-complex arrays output by YMAEDA_TOOLS.
    Version for complex python arrays: exifft_cplx().
    """
    assert len(M) == DATLEN
    M = M[:, 0] + M[:, 1] * 1j # combine real and imag parts together
    # for 2**12=4096 frequency points, [1:2047]=conj([4095:2049]) for 2047 partners
    # points 0 and 2048 are the 2 points without complex conjugate partners.
    M2 = M[1:DATLEN - 1]
    M2 = np.flipud(M2) # flip the order due to [1:2047] = np.conj([4095:2049])
    M2 = np.conj(M2) # remember to take the complex conjugate!!!!!!
    #M2 = np.conj(M2[::-1]) # note: [1:2047] = np.conj([4095:2049])
    M2 = np.hstack([M, M2]) # extend from half space to full space
    return idft(M2)

def exidft_cplx(M, DATLEN = 2049):
    """
    Extend and inverse discrete Fourier transform.
    For complex Python arrays.
    Extends the data from half frequency space to full frequency range and perform idft.
    For use on complex python arrays.
    Version for output from YMAEDA_TOOLS: exifft().
    """
    assert len(M) == DATLEN
    M2 = M[1:DATLEN - 1]
    M2 = np.flipud(M2) # flip the order due to [1:2047] = np.conj([4095:2049])
    M2 = np.conj(M2) # remember to take the complex conjugate!!!!!!
    #M2 = np.conj(M2[::-1]) # Note that: [1:2047] = np.conj([4095:2049])
    M2 = np.hstack([M, M2]) # extend from half space to full space
    return idft(M2)

def halfdft_timeshift(M, D):
    """
    Perform DFT, time shift, and then cut the frequency space from full to half.
    This is the inverse operation of exidft_timeshift.
    """
    M = dft(M) # DFT
    k = np.array(range(len(M)))
    M = M * np.exp(-1j * 2 * np.pi * k * D / len(M)) # time shift
    M = M[:1+int(len(M)/2)] # cut the frequency space into half
    return M

def exidft_timeshift(M, D, DATLEN = 2049):
    """
    Extend and inverse discrete Fourier transform with time shift.
    For output by YMAEDA_TOOLS.
    As the output by winv from YMAEDA_TOOLS only contains the first half
    of the frequency range up to Nyquist frequency, the data has to be
    extended into the complex conjugate half before ifft is performed.
    Version for complex Python arrays: exifft_cplx_timeshift().
    """
    assert len(M) == DATLEN
    M = M[:, 0] + M[:, 1] * 1j # combine real and imag parts together to complex array
    M2 = M[1:DATLEN - 1]
    M2 = np.flipud(M2) # flip the order due to [1:2047] = np.conj([4095:2049])
    M2 = np.conj(M2) # remember to take the complex conjugate!!!!!!
    M2 = np.hstack([M, M2]) # extend from half space to full space
    #for k in range(len(M2)):
    #    W = -1j * 2 * np.pi * k * (-D) / len(M2)
    #    M2[k] = M2[k] * np.exp(W)
    k = np.array(range(len(M2)))
    W = np.exp(-1j * 2 * np.pi * k * (-D) / len(M2))
    M2 = M2 * W
    return idft(M2)

def exidft_cplx_timeshift(M, D, DATLEN = 2049):
    """
    Extend and inverse discrete Fourier transform with time shift.
    For complex python arrays.
    Version for YMAEDA_TOOLS output: exifft_timeshift().
    """
    assert len(M) == DATLEN
    M2 = M[1:DATLEN - 1]
    M2 = np.flipud(M2) # flip the order due to [1:2047] = np.conj([4095:2049])
    M2 = np.conj(M2) # remember to take the complex conjugate!!!!!!
    M2 = np.hstack([M, M2]) # extend from half space to full space
    for k in range(len(M2)):
        W = -1j * 2 * np.pi * k * (-D) / len(M2)
        M2[k] = M2[k] * np.exp(W)
    return idft(M2)

def exifft_timeshift(M, D, DATLEN = 2049):
    """
    Extend and inverse discrete Fourier transform with time shift.
    Effectively the same function as exidft_timeshift.
    For output by YMAEDA_TOOLS.
    """
    assert len(M) == DATLEN
    M = M[:, 0] + M[:, 1] * 1j # combine real and imag parts together to complex array
    M2 = M[1:DATLEN - 1]
    M2 = np.flipud(M2) # flip the order due to [1:2047] = np.conj([4095:2049])
    M2 = np.conj(M2) # remember to take the complex conjugate!!!!!!
    M2 = np.hstack([M, M2]) # extend from half space to full space
    for k in range(len(M2)):
        W = -1j * 2 * np.pi * k * (-D-1) / len(M2)
        M2[k] = M2[k] * np.exp(W)
    return np.flipud(ifft(M2))

def exifft_cplx_timeshift(M, D, DATLEN = 2049):
    """
    Extend and inverse fast Fourier transform with time shift.
    Effectively the same function as exidft_cplx_timeshift.
    For complex Python arrays.
    """
    assert len(M) == DATLEN
    M = M[:, 0] + M[:, 1] * 1j # combine real and imag parts together to complex array
    M2 = M[1:DATLEN - 1]
    M2 = np.flipud(M2) # flip the order due to [1:2047] = np.conj([4095:2049])
    M2 = np.conj(M2) # remember to take the complex conjugate!!!!!!
    M2 = np.hstack([M, M2]) # extend from half space to full space
    for k in range(len(M2)):
        W = -1j * 2 * np.pi * k * (-D-1) / len(M2)
        M2[k] = M2[k] * np.exp(W)
    return np.flipud(ifft(M2))