import numpy as np
import struct # for dealing with binary data

# Functions for loading the snapshot files output by YMAEDA_TOOLS runwaterPML command.

# The snapshot files are stored in the parent directory: PML/snapshot/, and the snapshots
# themselves are stored in single file for each time step such as: source.Fx.t3.0000.3db
# which stores the snapshots for the time step t = 3.0000. The Fx in the file name indicates
# that the original impulse was applied in the x direction.

# frequency step size used by YMAEDA_TOOLS
df = 0.002441406 
# frequency half space used by YMAEDA_TOOLS
f = np.arange(0, df * 2049, df) 
# frequency full space used by YMAEDA_TOOLS
F = np.arange(0, df * 4096, df) 

def extract_snapshot(snapshot_dir, X = 0, Y = 0, Z = 0, t0 = 0.0, t1 = 12.0, dt = 0.1, return_params = False):
    """
    This is the most updated function to read snapshot data output by YMAEDA_TOOLS runwaterPML.
    
    This function should be a huge improvement over read_snapshot_loc3D_fast!!!
    
    Inputs
    ------
    snapshot_dir: str
        directory where the snapshot .3db files are located
    X: float
        x-location of the point to measure the Green's function (in actual coordinates)
    Y: float
        y-location of the point to measure the Green's function (in actual coordinates)
    Z: float
        z-location of the point to measure the Green's function (in actual coordinates)
    t0: float
        starting time
    t1: float
        ending time
    dt: float
        time step
    
    Returns
    -------
    t: np.array
        array of time steps
    g: np.array
        array of 3D Green's functions
    """
    n_steps = int((t1 - t0) / dt) + 1 # number of time steps
    g = np.zeros((n_steps, 3)) # final output green functions
    t = np.arange(t0, t1 + dt, dt) # final output times 
    j = ['x', 'y', 'z'] # 3 axes to read
    
    Ns = []
    x0s = []
    dxs = []
    
    BYTELEN = 8 # .3db file byte length
    for i, k in enumerate(j): # loop over all 3 axes
        gt = np.zeros(n_steps)
        for n in range(n_steps): # loop over all time steps. One file for one time step...
            TIME_ZEROS = format(t0 + dt * n, "0.4f")
            snapshot_file = snapshot_dir + "/source.F" + k + ".t" + TIME_ZEROS + ".3db"
            #print("> > > Loading: {}".format("/source.F" + k + ".t" + TIME_ZEROS + ".3db"))
            with open(snapshot_file, mode = "rb") as File:
                fileContent = File.read()
                N = struct.unpack("iii", fileContent[0:12]) # number of elements per axis
                x0 = struct.unpack("ddd", fileContent[12:36]) # starting point of each axis
                dx = struct.unpack("ddd", fileContent[36:60]) # distance step size per axis
                data_length = int((len(fileContent) - 60) / BYTELEN)
                if n == 0: # record the params only for the first time step. They should remain the same for the rest!
                    Ns.append(N)  
                    x0s.append(x0)
                    dxs.append(dx)
                # At the end of the data extraction process, Ns, x0s and dxs should have the form:
                # np.array([[Fx params], [Fy params], [Fz params]])
                assert data_length == N[0] * N[1] * N[2]
                
                idx, idy, idz = snapshot_stnloc(N, x0, dx, X, Y, Z)
                counter = index_convert31(idx, idy, idz, N[0], N[1], N[2])
                L = 60 + BYTELEN * counter
                R = L + BYTELEN
                gt[n] = struct.unpack('d', fileContent[L:R])[0]
        g[:, i] = gt.copy()
    if return_params == True:
        return t, g, Ns, x0s, dxs
    else:
        return t, g

def read_snapshot_params(snapshot_file = 'source.Fx.t3.0000.3db'):
    """
    Returns only the various parameters of a snapshot.3db file without outputting any data.
    
    N, x0, dx = read_snapshot_params(snapshot_dir = '/media/yumi/INVERSION/SMN_EW_SMALL/PML/snapshot/source.Fx.t3.0000.3db')
    """
    with open(snapshot_file, mode = 'rb') as File:
        fileContent = File.read()
        File.close()
        BYTELEN = 8
        N = struct.unpack('iii', fileContent[0:12])
        x0 = struct.unpack('ddd', fileContent[12:36])
        dx = struct.unpack('ddd', fileContent[36:60])
        data_length = int((len(fileContent) - 60) / BYTELEN)
        assert data_length == N[0] * N[1] * N[2]
    return N, x0, dx

def snapshot_XYZ(N, x0, dx):
    """
    Creates the X, Y and Z axis arrays from the output parameters from read_snapshot_params().
    
    X = arange(x0[0], x0[0]+dx[0]*N[0], dx[0])
    Y = arange(x0[1], x0[1]+dx[1]*N[1], dx[1])
    Z = arange(x0[2], x0[2]+dx[2]*N[2], dx[2])
    """
    X = np.array([x0[0] + dx[0] * i for i in range(N[0])])
    Y = np.array([x0[1] + dx[1] * i for i in range(N[1])])
    Z = np.array([x0[2] + dx[2] * i for i in range(N[2])])
    return X, Y, Z   

def snapshot_stnloc(N, x0, dx, X_STN, Y_STN, Z_STN):
    """
    From the parameters of the snapshot.3db file calculate the nearest grid location for a specified station.
    If the specified location is outside the grid, the grid point nearest 
    to the specified location is returned and a warning is given.
    SMN: -11175, -119878, 1317
    SMW: -12295, -120893, 1110
    LP:  -10900, -121100, 1000
    """
    X, Y, Z = snapshot_XYZ(N, x0, dx)
    idx = (abs(X - X_STN)).argmin()
    idy = (abs(Y - Y_STN)).argmin()
    idz = (abs(Z - Z_STN)).argmin()
    if not(X.min() <= X_STN <= X.max()):
        print('Warning! X out of range! Returning nearest grid value...')
    if not(Y.min() <= Y_STN <= Y.max()):
        print('Warning! Y out of range! Returning nearest grid value...')
    if not(Z.min() <= Z_STN <= Z.max()):
        print('Warning! Z out of range! Returning nearest grid value...')    
    return idx, idy, idz

def index_convert13(r, Nx, Ny, Nz):
    """
    Converts the 1D index of GT to the 3D index of GT3D
    1. the outer most loop: x
    2. next inner loop: y
    3. inner most loop: z
    4. therefore z is the most folded coordinate followed by y, x!
    """
    idz = np.mod(r, Nz) # most folded coordinate
    idy = np.mod(r // Nz, Ny) # next folded coordinate
    idx = r // Nz // Ny # least folded coordinate
    # output is in python index format of z:y:x
    return idz, idy, idx

def index_convert31(idx, idy, idz, Nx, Ny, Nz):
    """
    Converts the 3D index of GT3D to the 1D index of GT.
    """
    r = idx * Ny * Nz + idy * Nz + idz
    return r

def read_snapshot_fast(snapshot_dir = 'source.Fx.t3.0000.3db'):
    """
    Reads the PML snapshots generated by the FDM code in YMAEDA_TOOLS.
    File name (eg.): source.Fx.t0.0000.3db.
    First 12 bytes: Binary expressions for N[0], N[1], N[2].
    Next 24 bytes: Binary expressions for x0[0], x0[1], x0[2].
    Next 24 bytes: Binary expressions for dx[0], dx[1], dx[2].
    Following each 8 bytes: Binary expressions for member value[index[i][j][k]].
    The most out loop is that with regard to "x",
    the intermediate is "y" and the most inner loop is "z".
    Separators such as tab do not appear.
    
    Unlike the previous function this one outputs the data as a 1D array
    to speed up load and output times. This function should be used instead 
    of read_snapshot for most applications.
    """
    with open(snapshot_dir, mode = 'rb') as File:
        fileContent = File.read()
        File.close()
        BYTELEN = 8
        N = struct.unpack('iii', fileContent[0:12])
        #x0=struct.unpack('ddd',fileContent[12:36])
        #dx=struct.unpack('ddd',fileContent[36:60])
        data_length = int((len(fileContent) - 60) / BYTELEN)
        assert data_length == N[0] * N[1] * N[2]
        GT = np.zeros(data_length)
        for counter in range(data_length):
            L = 60 + BYTELEN * counter
            R = 60 + BYTELEN * counter + BYTELEN
            GT[counter] = struct.unpack('d', fileContent[L:R])[0]
    return GT, N

def read_snapshot_loc_fast(snapshot_dir, direction = 'z', X = 0, Y = 0, Z = 0, t0 = 0.0, t1 = 12.0, dt = 0.1):
    """
    Reads the snapshot over time for a particular location given by location indices 
    X, Y, Z over a time range of t0:t1 with dt.
    Faster code that deals with only 1D array structures.
    This code should be used instead of read_snapshot_loc.
    
    t,gx=read_snapshot_loc_fast(snapshot_dir="/media/yumi/INVERSION/SMN_EW_SMALL/PML/snapshot",direction='x',X=idx,Y=idy,Z=idz)
    t,gy=read_snapshot_loc_fast(snapshot_dir="/media/yumi/INVERSION/SMN_EW_SMALL/PML/snapshot",direction='y',X=idx,Y=idy,Z=idz)
    t,gz=read_snapshot_loc_fast(snapshot_dir="/media/yumi/INVERSION/SMN_EW_SMALL/PML/snapshot",direction='z',X=idx,Y=idy,Z=idz)
    """
    t = np.arange(t0, t1 + dt, dt)
    N = int((t1 - t0) / dt) + 1
    gt = np.zeros(N)
    for i in range(N): # loop over all time steps
        TIME_ZEROS = format(t0 + dt * i, '0.4f')
        snapshot_file = snapshot_dir + "/source.F" + direction + ".t" + TIME_ZEROS + ".3db"
        #print("> > > Loading: {}".format("/source.F" + direction + ".t" + TIME_ZEROS + ".3db"))
        GT, N0 = read_snapshot_fast(snapshot_file)
        gt[i] = GT[index_convert31(X, Y, Z, N0[0], N0[1], N0[2])]
    return t, gt

def read_snapshot_all_fast(snapshot_dir, direction = 'z', t0 = 0.0, t1 = 12.0, dt = 0.1):
    """
    This function reads and loads the entire snapshot over the specified time range for a particular direction. 
    THIS WILL CONSUME PLENTY OF MEMORY and should not be used on slower computers!!!
    """
    #t = np.arange(t0, t1 + dt, dt)
    N = int((t1 - t0) / dt) + 1
    G_ALL = []
    N_ALL = []
    for i in range(N):
        TIME_ZEROS = format(t0 + dt * i, '0.4f')
        snapshot_file = snapshot_dir + "/source.F" + direction + ".t" + TIME_ZEROS + ".3db"
        GT, N0 = read_snapshot_fast(snapshot_file)
        G_ALL.append(GT)
        N_ALL.append(N0)
    return G_ALL, N_ALL

def read_snapshot_loc3D_fast(snapshot_dir, X = 0, Y = 0, Z = 0, t0 = 0.0, t1 = 12.0, dt = 0.1): 
    """
    This faster function should be used instead of read_snapshot_loc3D.
    """
    N = int((t1 - t0) / dt) + 1
    g = np.zeros((N, 3))
    i = 0
    j = ['x', 'y', 'z']
    for k in j: # loop over all 3 axes
        t, g[:, i] = read_snapshot_loc_fast(snapshot_dir, k, X, Y, Z, t0, t1, dt)
        i = i + 1
    return t, g
