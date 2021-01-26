import numpy as np

# Pole zero and seismometer response functions used by YMAEDA_TOOLS

# frequency step size used by YMAEDA_TOOLS
df = 0.002441406 
# frequency half space used by YMAEDA_TOOLS
f = np.arange(0, df*2049, df) 
# frequency full space used by YMAEDA_TOOLS
F = np.arange(0, df*4096, df) 

def read_pzfile(pzfilename = "/Volumes/MAC Backup/ymaeda_tools_mac/winv/share/polezero/tri120p"):
    f = open(pzfilename,'r')
    X = f.readlines()
    f.close()
    NPOLES = int(X[0][6])
    POLES = np.zeros(NPOLES,'complex')
    for i in range(NPOLES):
        buf = X[i+1]
        buf = buf.strip().split()
        POLES[i] = float(buf[0]) + float(buf[1])*1j
    NZEROS = int(X[NPOLES+1][6])
    ZEROS = np.zeros(NZEROS, 'complex')
    for i in range(NZEROS):
        buf = X[i+NPOLES+2]
        buf = buf.strip().split()
        if buf[0] == 'CONSTANT':
            break
        else:
            ZEROS[i] = float(buf[0]) + float(buf[1])*1j
    CONSTANT = float(X[-1].strip().split()[1])
    return POLES, ZEROS, CONSTANT    

def plot_seismometer_resp(POLES, ZEROS, CONSTANT, w0 = 0, w = 100, dw = 0.01):  
    s = np.arange(w0,w,dw)*1j # this is in rad/s and not 1/s
    H = np.ones(len(s))
    for i in range(len(ZEROS)):
        H = H * (s - ZEROS[i])
    # multiple the poles:
    for i in range(len(POLES)):
        H = H / (s - POLES[i])
    H = H * CONSTANT
    pha = np.angle(H)
    amp = abs(H)
    mag = 20 * np.log10(amp)
    s = s / (2 * np.pi) # convert from rad/s to 1/s 
    plt.subplot(2, 1, 1)
    plt.semilogx(np.imag(s), mag)
    plt.ylabel('Amplitude')
    plt.grid('on')
    plt.subplot(2, 1, 2)
    plt.semilogx(np.imag(s), pha / np.pi * 180)
    plt.ylabel('Phase')
    plt.grid('on')
    plt.xlabel('Frequency (Hz)')
    plt.show()
    
def seismometer_resp(POLES, ZEROS, CONSTANT, w0 = 0, w1 = 100, dw = 0.01):  
    s = np.arange(w0, w1, dw) * 1j # this is in rad/s and not 1/s
    H = np.ones(len(s))
    for i in range(len(ZEROS)):
        H = H * (s - ZEROS[i])
    for i in range(len(POLES)):
        H = H / (s - POLES[i])
    H = H * CONSTANT
    # np.imag(s) is returned in rad/s and not Hz!!!!!
    return np.imag(s), H

def remove_resp(POLES, ZEROS, CONSTANT, f, X, NO_DC = True):
    """
    input the frequency f in Hz! The program will convert to rad/s
    This script removes the seismometer response from a Fourier-transformed
    time series. The first element X[0] is assumed to be zero such that the
    time series has no Direct Current (f=0) component.
    Note that POLES and ZEROS are in rad/s. Ensure that calculations do not 
    mix values given in Hz and those given in rad/s!!!!!!!!!!!!!!!!!!!!!!!
    """
    s = f * 2 * np.pi * 1j #convert frequency to rad/s
    H = np.ones(len(s))
    for i in range(len(ZEROS)):
        H = H * (s - ZEROS[i])
    for i in range(len(POLES)):
        H = H / (s - POLES[i])
    H = H * CONSTANT
    Y = np.zeros(len(s), complex)
    Y[1:] = X[1:] / H[1:]
    if NO_DC == False:
        Y[0] = X[0]
    return Y