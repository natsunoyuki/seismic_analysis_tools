import numpy as np
import struct #for dealing with binary data

# Functions for loading d, G and m files output by YMAEDA_TOOLS

# frequency step size used by YMAEDA_TOOLS
df = 0.002441406 
# frequency half space used by YMAEDA_TOOLS
f = np.arange(0, df*2049, df) 
# frequency full space used by YMAEDA_TOOLS
F = np.arange(0, df*4096, df) 

def read_Mseq1(main_dir):
    """
    reads the output M.seq1 results from YMAEDA_TOOLS in time domain
    """
    main_dir = main_dir + 'model/M.seq1'
    synm = np.array([])

    f = open(main_dir, 'r')
    B = f.read().splitlines() #remove newline characters
    f.close()
    Size = len(B) - 4
    #t0 = float(B[1][3:]) #initial time
    #dt = float(B[2][3:]) #time step
    
    synwave_buffer = np.array([])
    for i in range(Size):
        synwave_buffer = np.hstack([synwave_buffer, float(B[i + 4])])
    synm = synwave_buffer[0:Size] # bug makes synwave_buffer Size+1, remove extra line
    return synm

def read_dseq1(data_obs_dir, file_name):
    """
    reads the output d.seq1 observed seismograms in time domain
    d=read_dseq1(data_obs_dir,'EV.SMN.E.sac.seq1')
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
    synm=synwave_buffer[0:Size] #bug makes synwave_buffer Size+1, remove extra line
    return synm    

def read_dimseq2(tf_dir, file_name):
    """
    reads the output d.imseq2 observed seismograms in frequency domain
    a,b=read_dimseq2(data_obs_spectrum_dir,'EV.SMN.E.sac.imseq2')
    """
    f = open(tf_dir + file_name, 'r')
    B = f.read().splitlines()
    f.close()   
    Size = len(B) - 4
    #f0 = float(B[1][3:]) #initial frequency
    #df = float(B[2][3:]) #frequency step
    B = B[4:]
    f = np.zeros(Size)
    C = np.zeros(Size,complex)
    for i in range(Size):
        B[i] = B[i].split('\t')
        f[i] = float(B[i][0])
        C[i] = float(B[i][1]) + 1j * float(B[i][2])
    return f, C

def read_dobs(main_dir, i):
    """
    reads the dX.cv data files in frequency domain, output by YMAEDA_TOOLS
    """
    d_dir = main_dir + 'd_obs/d'
    d_file = d_dir + str(i) + '.cv'
    ddata = np.loadtxt(d_file)
    d = ddata[1:]
    return d

def readall_dobs(main_dir, DATLEN = 2049, ROW = 0):
    """
    reads the dX.cv data files in frequency domain, output by YMAEDA_TOOLS
    for one particular station
    D=readall_dobs(main_dir,ROW=0)
    """
    d_dir = main_dir + 'd_obs/d'
    DOBS = np.zeros(DATLEN,complex)
    for i in range(DATLEN):
        d_file = d_dir + str(i) + '.cv'
        ddata = np.loadtxt(d_file)
        d = ddata[1:]
        d = d[0:int(len(d) / 2)] + d[int(len(d) / 2):] * 1j
        DOBS[i] = d[ROW]
    return DOBS

def read_mest(main_dir, i):
    """
    reads the mX.cv data files in frequency domain output by YMAEDA_TOOLS
    """
    m_dir = main_dir + 'm_est/m'
    m_file = m_dir + str(i) + '.cv'
    mdata = np.loadtxt(m_file)
    m = mdata[1:]
    return m

def readall_mest(main_dir, DATLEN=2049, NM=2):
    """
    reads ALL the mX.cv data files in frequency domain output by YMAEDA_TOOLS
    """
    M = np.zeros([DATLEN,NM], complex)
    for i in range(DATLEN):
        m = read_mest(main_dir, i)
        M[i,:] = m
    return M

def read_G(main_dir, i):
    """
    reads the GX.bdm binary data files output by YMAEDA_TOOLS
    """
    G_dir = main_dir + 'G/G'
    G_file = G_dir + str(i) + '.bdm'
    with open(G_file, mode='rb') as file:
        fileContent = file.read()
        file.close()
        nrows = struct.unpack("i", fileContent[:4]) #no of rows
        nrows = nrows[0]
        ncols = struct.unpack("i", fileContent[4:8]) #no of cols
        ncols = ncols[0]
        dataSize = len(fileContent)-8 #first 8 bytes are the nrows and ncols
        data = np.zeros(nrows * ncols) #array to hold the data
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
                G[iii,jjj] = data[count]
                count = count + 1
    return G

def readall_G(main_dir, DATLEN=2049):
    """
    returns a list of arrays which needs to be unfolded
    """
    Gstack = []
    for i in range(DATLEN):
        G = read_G(main_dir, i)
        Gstack.append(G)
    return Gstack

def unfold_G(Gstack, INDEX=0, DATLEN=2049):
    """
    to be used with readall_G to unfold the individual green functions
    """
    g = np.zeros(DATLEN, complex)
    for i in range(DATLEN):
        g[i] = Gstack[i][INDEX][0] - Gstack[i][INDEX][1] * 1j
    return g
