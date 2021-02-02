import numpy as np

# Functions for loading and creating source time functions output by YMAEDA_TOOLS

# frequency step size used by YMAEDA_TOOLS
df = 0.002441406 
# frequency half space used by YMAEDA_TOOLS
f = np.arange(0, df * 2049, df) 
# frequency full space used by YMAEDA_TOOLS
F = np.arange(0, df * 4096, df) 

def read_stfunseq2(stfun_dir):
    """
    Reads the zero padded source time function output by YMAEDA_TOOLS. 
    Some how, making fourier transform of this time series does not result
    in the fourier transform obtained through read_stfunimseq2(stfun_dir)...
    
    Usage example:
    t, x = read_stfunseq2(stfun_dir)
    
    Inputs
    ------
    stfun_dir: str
        Directory containing the stfun.seq2 file to be loaded.
        
    Returns
    -------
    t: np.array
        Time values of the loaded stfun.seq2 file.
    x: np.array
        Loaded time domain values of the stfun.seq2 file.
    """
    filename = stfun_dir + "stfun.seq2"
    f = open(filename, 'r')
    B = f.read().splitlines()
    f.close()    
    Size = len(B) - 4
    #t0 = float(B[1][3:]) #initial time
    #dt = float(B[2][3:]) #time step
    B = B[4:]
    t = np.zeros(Size)
    x = np.zeros(Size)
    for i in range(Size):
        B[i] = B[i].split('\t')
        t[i] = float(B[i][0])
        x[i] = float(B[i][1])
    return t, x

def read_stfunimseq2(stfun_dir):
    """
    Reads the fourier spectrum of the source time function output by YMAEDA_TOOLS.
    
    Usage example:
    f, S = read_stfunimseq2(stfun_dir)
    
    Inputs
    ------
    stfun_dir: str
        Directory containing the stfun_spectrum.imseq2 file to be loaded.
        
    Returns
    -------
    f: np.array
        Frequency values of the loaded stfun_spectrum.imseq2 file.
    S: np.array
        Loaded Fourier spectrum values of the stfun_spectrum.imseq2 file.
    """
    filename = stfun_dir + "stfun_spectrum.imseq2"
    f = open(filename, 'r')
    B = f.read().splitlines()
    f.close()    
    Size = len(B) - 4
    #t0 = float(B[1][3:]) #initial time
    #dt = float(B[2][3:]) #time step
    B = B[4:]
    t = np.zeros(Size)
    x = np.zeros(Size)
    y = np.zeros(Size)
    for i in range(Size):
        B[i] = B[i].split('\t')
        t[i] = float(B[i][0])
        x[i] = float(B[i][1])
        y[i] = float(B[i][2])
    return t, x + y * 1j

def create_timefunc_pow5(tp = 1.0, size = 121, t0 = 0.0, dt = 0.1, Nintegral = 0):
    t = np.array([t0 + dt * i for i in range(size)])
    ft = 10 * (t/tp)**3 - 15 * (t/tp)**4 + 6 * (t/tp)**5
    ft[t > tp] = 1
    return t, ft    
    
def create_timefunc_pow33(tp = 1.0, size = 121, t0 = 0.0, dt = 0.1, Nintegral = 0):
    t = np.array([t0 + dt * i for i in range(size)])
    ft = -64.0 * tp** -6 * (t**5 * (t - tp)**3)
    ft[t > tp] = 0
    return t, ft

def create_timefunc_pow34(tp = 1.0, size = 121, t0 = 0.0, dt = 0.1, Nintegral = 0):
    t = np.array([t0 + dt * i for i in range(size)])
    ft = (7.0/3.0)**3 * (7.0/4.0)**4 * (t/tp)**3 * (1 - t/tp)**4
    ft[t > tp] = 0
    return t, ft 

def create_timefunc_cos(tp = 1.0, size = 121, t0 = 0.0, dt = 0.1, Nintegral = 0):
    t = np.array([t0 + dt * i for i in range(size)])
    ft = 0.5 * (1 - np.cos(2.0 * np.pi * t / tp))
    ft[t > tp] = 0
    return t, ft