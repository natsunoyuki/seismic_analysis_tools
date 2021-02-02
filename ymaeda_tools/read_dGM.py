import numpy as np
import struct # for dealing with binary data

# Functions for loading d_obs.cv, G.dbm and m_est.cv files output by YMAEDA_TOOLS.

# frequency step size used by YMAEDA_TOOLS
df = 0.002441406 
# frequency half space used by YMAEDA_TOOLS
f = np.arange(0, df * 2049, df) 
# frequency full space used by YMAEDA_TOOLS
F = np.arange(0, df * 4096, df) 

# Output files from YMAEDA_TOOLS winv linear inversion algorithm
# --------------------------------------------------------------
# d_obs.cv, m_est.cv and G.dbm are output as "complex number" data files.
#
# In linear inversion we assume that the following relation holds:
# d = G.m <==> m = inv(G).d
# 
# We first look at the case where we only have 1 moment tensor element
# which is common in the case of geometrically constrained linear inversions.
#
# Take for example N = 6 (6 seismogram traces) and M = 1 (1 moment tensor element), 
# then d = [6 x 1], m = [1 x 1] and G = [6 x 1] in shape.
# 
# The inversion is conducted in complex frequency space with the following forms:
# d_i = d_i_real + d_i_imag * 1j
# m = m_real + m_imag * 1j
# G_i = G_i_real + G_i_imag * 1j

# Therefore, the complex vector d_i can be written in terms of G_i and m as :
# d_i_real + d_i_imag * 1j = (G_i_real + G_i_imag * 1j) * (m_real + m_imag * 1j) = 
# G_i_real * m_real + G_i_real * m_imag * 1j + G_i_imag * m_real * 1j - G_i_imag * m_imag = 
# (G_i_real * m_real - G_i_imag * m_imag) + (G_i_real * m_imag + G_i_imag * m_real) * 1j.
#
# The equation above can be written in matrix form as: 
# d_i_real          G_i_real       -G_i_imag         m_real
#                =                                .  
# d_i_imag * 1j     G_i_imag * 1j  G_i_real * 1j     m_imag
#  
# From the above matrix forms, the corresponding files have the contents: 
#           d_1_real         G_1_real, -G_1_imag
#           d_2_real         G_2_real, -G_2_imag
#           d_3_real         G_3_real, -G_3_imag
#           d_4_real         G_4_real, -G_4_imag
#           d_5_real         G_5_real, -G_5_imag           
# d_obs.cv: d_6_real  G.bdm: G_6_real, -G_6_imag  m_est.cv: m_real
#           d_1_imag         G_1_imag,  G_1_real            m_imag
#           d_2_imag         G_2_imag,  G_2_real
#           d_3_imag         G_3_imag,  G_3_real
#           d_4_imag         G_4_imag,  G_4_real
#           d_5_imag         G_5_imag,  G_5_real
#           d_6_imag         G_6_imag,  G_6_real
#           [12 x 1]              [12 x 2]                  [2 x 1]
#
# There will be 2 singular values in the SVD eigenvalue matrix, one for the real part of m and one for the imag part.
# Both singular values are treated as essentially real by splitting x + jy into x and y seperately.
# 
# To convert from YMAEDA_TOOLS format to something more familiar in python, the following conversion equations are used:
# d_py = d_obs[:len(d_obs)/2] + d_obs[len(d_obs)/2:] * 1j
# G_py = G.bdm[:len(d_obs)/2, 0] - G.bdm[len(d_obs)/2:, 1] * 1j
# m_py = m[0] + m[1] * 1j
#
# Now that we have solved the simple case where M = 1, we can extend the formalism to the full case
# where M = 6 (there are 6 independent components in the seismic moment tensor: Mxx, Myy, Mzz, Mxy, Myz, Mxz).
#
# For the full case, the complex d_i values can be written in terms of G_ij and m_j as:
# d_i_real = G_i1_real * m_1_real + G_i2_real * m_2_real + G_i3_real * m_3_real + 
#            G_i4_real * m_4_real + G_i5_real * m_5_real + G_i6_real * m_6_real + 
#           -G_i1_imag * m_1_imag - G_i2_imag * m_2_imag - G_i2_imag * m_2_imag +
#           -G_i4_imag * m_4_imag - G_i5_imag * m_5_imag - G_i6_imag * m_6_imag
#
# d_i_imag = G_i1_imag * m_1_real + G_i2_imag * m_2_real + G_i3_imag * m_3_real + 
#            G_i4_imag * m_4_real + G_i5_imag * m_5_real + G_i6_imag * m_6_real + 
#            G_i1_real * m_1_imag + G_i2_real * m_2_imag + G_i2_real * m_2_imag +
#            G_i4_real * m_4_imag + G_i5_real * m_5_imag + G_i6_real * m_6_imag
#
# Using the equation above, the corresponding files have the contents: 
#           d_1_real         G_11_real, ... G_16_real, -G_11_imag, ... -G_16_imag            m_1_real
#           d_2_real         G_21_real, ... G_26_real, -G_21_imag, ... -G_26_imag            m_2_real
#           ...              ...                                                             ...
# d_obs.cv: d_N_real  G.bdm: G_N1_real, ... G_N6_real, -G_N1_imag, ... -G_N6_imag  m_est.cv: m_6_real
#           d_1_imag         G_11_imag, ... G_16_imag,  G_11_real, ...  G_16_real            m_1_imag
#           d_2_imag         G_21_imag, ... G_26_imag,  G_21_real, ...  G_26_real            m_2_imag
#           ...              ...                                                             ...
#           d_N_imag         G_N1_imag, ... G_N6_imag,  G_N1_real, ...  G_N6_real            m_6_imag
#           [2N x 1]                              [2N x 12]                                  [12 x 1]
# 
# There will be 12 singular values in the SVD eigenvalue matrix, one for the real part  and one for the imag part
# of all 6 independent values of m. All singular values are treated as essentially real by splitting x + jy into 
# x and y seperately.
# 
# To convert from YMAEDA_TOOLS format to something more familiar in python, the following conversion equations are used:
# d_py = d_obs[:len(d_obs)/2] + d_obs[len(d_obs)/2:] * 1j
# G_py = G.bdm[:len(d_obs)/2, :len(m)/2] - G.bdm[:len(d_obs)/2, len(m)/2:] * 1j
# m_py = m[:len(m)/2] + m[len(m)/2:] * 1j

def read_Mseq1(main_dir):
    """
    Reads the output M.seq1 results from YMAEDA_TOOLS in time domain.
    This file is tailored specifically to read from the model/ directory
    for convenience's sake, and cannot be used for other data!
    
    Inputs
    ------
    main_dir: str
        Directory path to the parent folder where model/ containing M.seq1 is located.
    
    Returns
    -------
    synm: np.array
        Loaded moment tensor from M.seq1.
    """
    synm = read_dseq1(main_dir, 'model/M.seq1')
    return synm

def read_dseq1(data_obs_dir, file_name):
    """
    Reads the output d.seq1 observed seismograms in time domain.
    Although originally created to read d.seq1 files, this function
    should be able to read all .seq1 time series files created by YMAEDA_TOOLs.
    
    Usage example:
    d = read_dseq1(data_obs_dir, 'EV.SMN.E.sac.seq1')
    
    Inputs
    ------
    data_obs_dir: str
        Directory path containing the .seq1 file to be loaded.
    file_name:str
        File name of the .seq1 file to be loaded.
    
    Returns
    -------
    synm: np.array
        Contents of the loaded .seq1 file.
    """
    f = open(data_obs_dir + file_name, 'r')
    B = f.read().splitlines()
    f.close()
    Size = len(B) - 4
    #t0 = float(B[1][3:]) # initial time
    #dt = float(B[2][3:]) # time step
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

    Usage example:
    a, b = read_dimseq2(data_obs_spectrum_dir, 'EV.SMN.E.sac.imseq2')
    
    Inputs
    ------
    tf_dir: str
        Directory path containing the .imseq2 file to be loaded.
    file_name: str
        File name of the .imseq2 file to be loaded.
        
    Returns
    -------
    f: np.array 
        Frequency steps of the loaded .imseq2 file.
    C: np.array
        Contents of the loaded .imseq2 file.
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
    
    The conversion equation used is (see notes above):
    d_py = d_obs[:len(d_obs)/2] + d_obs[len(d_obs)/2:] * 1j.
    
    Inputs
    ------
    d_obs: np.array
        Contents of the loaded dX.cv data.
    
    Returns
    -------
    d_py: np.array
        Contents of the loaded dX.cv data converted to complex form for use in Python.
    """
    d_py = d_obs[:int(len(d_obs)/2)] + d_obs[int(len(d_obs)/2):] * 1j
    return d_py
    
def read_dobs(main_dir, i):
    """
    Reads the dX.cv data files in frequency domain, output by YMAEDA_TOOLS.
    
    Inputs
    ------
    main_dir: str
        Parent directory containing the d_obs/ folder which contains the dX.cv files.
    i: int
        Index of the dX.cv file to load.
        
    Returns
    -------
    d: np.array
        Contents of the loaded dX.cv file.
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
    for one particular station defined by ROW.
    
    The output is converted into a python complex array.
    
    Usage example:
    D = readall_dobs(main_dir, ROW = 0)
    
    Inputs
    ------
    main_dir: str
        Parent directory containing the d_obs/ folder which contains the dX.cv files.
    DATLEN: int
        Data length (number of dX.cv files to load). Set to 2049 for compatibility with YMAEDA_TOOLS.
    ROW: int
        Which row of data within dX.cv to load.
        
    Returns
    -------
    DOBS: np.array
        Loaded dX.cv data from all 2049 files in d_obs/.
    """
    DOBS = np.zeros(DATLEN, complex)
    for i in range(DATLEN):
        d = read_dobs(main_dir, i)
        d = d[:int(len(d) / 2)] + d[int(len(d) / 2):] * 1j
        DOBS[i] = d[ROW]
    return DOBS

def mcv_to_m(m):
    """
    Converts the loaded mX.cv data files in frequency domain output by YMAEDA_TOOLS,
    to complex form a + jb.
    
    The conversion equation used is (see notes above):
    m_py = m[:len(m)/2] + m[len(m)/2:] * 1j.
    
    Inputs
    ------
    m: np.array
        Loaded mX.cv data.
    
    Returns
    -------
    m_py: np.array
        Loaded mX.cv data converted to complex form for use in Python.
    """
    #m_py = m[0] + m[1] * 1j
    m_py = m[:int(len(m)/2)] + m[int(len(m)/2):] * 1j
    return m_py

def read_mest(main_dir, i):
    """
    Reads the mX.cv data files in frequency domain output by YMAEDA_TOOLS.
    
    Inputs
    ------
    main_dir: str
        Parent directory containing the m_est/ folder which contains the mX.cv files.
    i: int
        Index of the mX.cv file to load.
        
    Returns
    -------
    m: np.array
        Loaded mX.cv file data.
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
        
    Inputs
    ------
    main_dir: str
        Parent directory containing the m_est/ folder which contains the mX.cv files.
    DATLEN: int
        Data length (number of mX.cv files to load). Set to 2049 for compatibility with YMAEDA_TOOLS.
    NM: int
        Number of columns in YMAEDA_TOOLS output for the estimated moment tensor. Minimum is 2 for one real and one imag part.
        
    Returns
    -------
    M: np.array
        Loaded data from all 2049 mX.cv files located 
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
    
    The conversion equation used is (see notes above):
    G_py = G.bdm[:len(d_obs)/2, :len(m)/2] - G.bdm[:len(d_obs)/2, len(m)/2:] * 1j.
    
    Inputs
    ------
    G: np.array
        Loaded G.dbm data from YMAEDA_TOOLS.
    
    Returns
    -------
    GG: np.array
        Loaded G.dbm data converted to complex form for use in Python.
    """
    nrows, ncols = np.shape(G)
    GG = G[:int(nrows/2), :int(ncols/2)] - G[:int(nrows/2), int(ncols/2):] * 1j
    return GG

def read_G(main_dir, i):
    """
    Reads the G.bdm binary data files output by YMAEDA_TOOLS.
    
    Inputs
    ------
    main_dir: str
        Parent directory containing the G/ folder which contains the G.dbm files.
    i: int
        Index of the G.dbm file to be loaded.
        
    Returns
    -------
    G: np.array
        Loaded G.dbm data.
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
    
    Inputs
    ------
    main_dir: str
        Parent directory containing the G/ folder which contains the G.dbm files.
    DATLEN: int
        Number of G.dbm files in G/. Set to 2049 for compatibility with YMAEDA_TOOLS.
        
    Returns
    -------
    Gstack: np.array
        List of arrays which needs to be unfolded.
    """
    Gstack = []
    for i in range(DATLEN):
        G = read_G(main_dir, i)
        Gstack.append(G)
    return Gstack

def unfold_G(Gstack, INDEX = 0, DATLEN = 2049):
    """
    This function is to be used with readall_G to unfold the individual green functions.
    
    Inputs
    ------
    Gstack: list
        List of arrays to be unfolded.
    INDEX: int
        Index of the array to be unfolded.
    DATLEN: int
        Number of G.dbm files in G/. Set to 2049 for compatibility with YMAEDA_TOOLS.
    
    Returns
    -------
    g: np.array
        Unfolded arrays of G.
    """
    g = np.zeros(DATLEN, complex)
    for i in range(DATLEN):
        g[i] = Gstack[i][INDEX][0] - Gstack[i][INDEX][1] * 1j
    return g
