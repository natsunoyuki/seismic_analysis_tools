import numpy as np
import struct #for dealing with binary data

# Functions for loading d, G and m files output by YMAEDA_TOOLS.

# frequency step size used by YMAEDA_TOOLS
df = 0.002441406 
# frequency half space used by YMAEDA_TOOLS
f = np.arange(0, df*2049, df) 
# frequency full space used by YMAEDA_TOOLS
F = np.arange(0, df*4096, df) 

# Matrix output from YMAEDA_TOOLS winv
# ------------------------------------
# d_obs.cv, m_est.cv and G.bdm are output as complex data files.
#
# In linear inversion we assume that the following relation holds:
# d = G.m <==> m = inv(G).d
# 
# If N = 6 (6 seismogram traces) and M = 1 (1 moment tensor element), 
# then d = [6 x 1], m = [1 x 1] and G = [6 x 1] in shape.
# 
# The inversion is conducted in complex frequency space with the following forms:
# d_i = d_i_real + d_i_imag * 1j
# m = m_real + m_imag * 1j
# G_i = G_i_real + G_i_imag * 1j

# Therefore, d_i_real + d_i_imag * 1j = (G_i_real + G_i_imag * 1j) * (m_real + m_imag * 1j) = 
# G_i_real * m_real + G_i_real * m_imag * 1j + G_i_imag * m_real * 1j - G_i_imag * m_imag = 
# (G_i_real * m_real - G_i_imag * m_imag) + (G_i_real * m_imag + G_i_imag * m_real) * 1j.
# This can be written in matrix form as: 
# d_i_real          G_i_real       -G_i_imag         m_real
#                =                                .  
# d_i_imag * 1j     G_i_imag * 1j  G_i_real * 1j     m_imag
#  
# From the above matrix forms, the corresponding files have the contents: 
#           d_1_real        G_1_real -G_1_imag
#           d_2_real        G_2_real -G_2_imag
#           d_3_real        G_3_real -G_3_imag
#           d_4_real        G_4_real -G_4_imag
#           d_5_real        G_5_real -G_5_imag           m_real
# d_obs.cv: d_6_real G.bdm: G_6_real -G_6_imag m_est.cv: m_imag
#           d_1_imag        G_1_imag  G_1_real
#           d_2_imag        G_2_imag  G_2_real
#           d_3_imag        G_3_imag  G_3_real
#           d_4_imag        G_4_imag  G_4_real
#           d_5_imag        G_5_imag  G_5_real
#           d_6_imag        G_6_imag  G_6_real
#           [12 x 1]            [12 x 2]                 [2 x 1]
#
# There will be 2 singular values in Lambda, one for the real part of m and one for the imag part.
# Both singular values are treated as essentially real by splitting x + jy into x and y seperately.
# 
# To convert from YMAEDA_TOOLS format to something more familiar in python, 
# the following conversion equations are used:
# d_py = d_obs[:len(d_obs)/2] + d_obs[len(d_obs)/2:] * 1j
# G_py = G.bdm[:len(d_obs)/2, 0] - G.bdm[len(d_obs)/2:, 1] * 1j
# m_py = m[0] + m[1] * 1j

def read_Mseq1(main_dir):
    """
    Reads the output M.seq1 results from YMAEDA_TOOLS in time domain.
    This file is tailored specifically to read from the model/ directory
    for convenience's sake, and cannot be used for other data!
    """
    synm = read_dseq1(main_dir, 'model/M.seq1')
    return synm

def read_dseq1(data_obs_dir, file_name):
    """
    Reads the output d.seq1 observed seismograms in time domain.
    Although originally created to read d.seq1 files, this function
    should be able to read all .seq1 time series files created by YMAEDA_TOOLs.
    
    d = read_dseq1(data_obs_dir, 'EV.SMN.E.sac.seq1')
    """
    f = open(data_obs_dir + file_name, 'r')
    B = f.read().splitlines()
    f.close()
    Size = len(B) - 4
    #t0 = float(B[1][3:]) #initial time
    #dt = float(B[2][3:]) #time step
    synwave_buffer = np.array([])
    for i in range(Size):
        synwave_buffer = np.hstack([synwave_buffer, float(B[i + 4])])
    synm = synwave_buffer[0:Size] # bug may make synwave_buffer Size+1, remove extra element if needed
    assert len(synm) == Size
    return synm    

def read_dimseq2(tf_dir, file_name):
    """
    Reads the output d.imseq2 observed seismograms in frequency domain.
    Although originally created to read d.imseq2 files, this function
    should be able to read all .imseq2 fourier spectra files created by YMAEDA_TOOLs.    

    a, b = read_dimseq2(data_obs_spectrum_dir, 'EV.SMN.E.sac.imseq2')
    """
    f = open(tf_dir + file_name, 'r')
    B = f.read().splitlines()
    f.close()   
    Size = len(B) - 4
    #f0 = float(B[1][3:]) # initial frequency
    #df = float(B[2][3:]) # frequency step
    B = B[4:]
    f = np.zeros(Size)
    C = np.zeros(Size,complex)
    for i in range(Size):
        B[i] = B[i].split('\t')
        f[i] = float(B[i][0])
        C[i] = float(B[i][1]) + 1j * float(B[i][2])
    return f, C

def dcv_to_d(d_obs):
    """
    Converts the loaded dX.cv data files in frequency domain output by YMAEDA_TOOLS,
    to complex form a + jb.
    """
    d_py = d_obs[:int(len(d_obs)/2)] + d_obs[int(len(d_obs)/2):] * 1j
    return d_py
    
def read_dobs(main_dir, i):
    """
    Reads the dX.cv data files in frequency domain, output by YMAEDA_TOOLS.
    """
    d_dir = main_dir + 'd_obs/d'
    d_file = d_dir + str(i) + '.cv'
    ddata = np.loadtxt(d_file)
    n_data_in_file = ddata[0]
    d = ddata[1:]
    assert len(d) == n_data_in_file
    return d

def readall_dobs(main_dir, DATLEN = 2049, ROW = 0):
    """
    Reads the dX.cv data files in frequency domain, output by YMAEDA_TOOLS
    for one particular station defined by ROW
    
    D = readall_dobs(main_dir, ROW = 0)
    """
    d_dir = main_dir + 'd_obs/d'
    DOBS = np.zeros(DATLEN, complex)
    for i in range(DATLEN):
        d_file = d_dir + str(i) + '.cv'
        ddata = np.loadtxt(d_file)
        n_data_in_file = ddata[0]
        d = ddata[1:]
        assert len(d) == n_data_in_file
        d = d[0:int(len(d) / 2)] + d[int(len(d) / 2):] * 1j
        DOBS[i] = d[ROW]
    return DOBS

def mcv_to_m(m):
    """
    Converts the loaded mX.cv data files in frequency domain output by YMAEDA_TOOLS,
    to complex form a + jb.
    """
    m_py = m[0] + m[1] * 1j
    return m_py

def read_mest(main_dir, i):
    """
    Reads the mX.cv data files in frequency domain output by YMAEDA_TOOLS.
    """
    m_dir = main_dir + 'm_est/m'
    m_file = m_dir + str(i) + '.cv'
    mdata = np.loadtxt(m_file)
    m = mdata[1:]
    return m

def readall_mest(main_dir, DATLEN = 2049, NM = 2):
    """
    Reads ALL the mX.cv data files in frequency domain output by YMAEDA_TOOLS.
    
    NM: number of M components to read. Note that real and imag portions form 2 components.
        [:, 0] is the real component, [:, 1] is the imaginary part.
    """
    M = np.zeros([DATLEN, NM])
    for i in range(DATLEN):
        m = read_mest(main_dir, i)
        M[i, :] = m
    return M

def Gdbm_to_G(G):
    """
    Converts the loaded G.dbm data files in frequency domain output by YMAEDA_TOOLS,
    to complex form a + jb.
    """
    nrows, ncols = np.shape(G)
    GG = G[0:int(nrows/2)]
    GG = GG[:, 0:int(ncols/2)] - GG[:, int(ncols/2):] * 1j
    return GG

def read_G(main_dir, i):
    """
    Reads the GX.bdm binary data files output by YMAEDA_TOOLS.
    """
    G_dir = main_dir + 'G/G'
    G_file = G_dir + str(i) + '.bdm'
    with open(G_file, mode='rb') as file:
        fileContent = file.read()
        file.close()
        nrows = struct.unpack("i", fileContent[:4]) # no of rows
        nrows = nrows[0]
        ncols = struct.unpack("i", fileContent[4:8]) # no of cols
        ncols = ncols[0]
        dataSize = len(fileContent)-8 # first 8 bytes are the nrows and ncols
        data = np.zeros(nrows * ncols) # array to hold the data
        count = 0
        for ii in range(0, dataSize, 8):
            B = fileContent[ii+8:ii+8*2]
            b = struct.unpack("d", B)
            b = b[0]
            #print b
            data[count] = b
            count = count + 1
        G = np.zeros([nrows, ncols])
        count = 0
        for iii in range(nrows):
            for jjj in range(ncols):
                G[iii, jjj] = data[count]
                count = count + 1
    return G

def readall_G(main_dir, DATLEN = 2049):
    """
    Returns a list of arrays which needs to be unfolded.
    """
    Gstack = []
    for i in range(DATLEN):
        G = read_G(main_dir, i)
        Gstack.append(G)
    return Gstack

def unfold_G(Gstack, INDEX = 0, DATLEN = 2049):
    """
    This function is to be used with readall_G to unfold the individual green functions.
    """
    g = np.zeros(DATLEN, complex)
    for i in range(DATLEN):
        g[i] = Gstack[i][INDEX][0] - Gstack[i][INDEX][1] * 1j
    return g
