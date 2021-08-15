import numpy as np
import struct # for dealing with binary data
import os

# Functions for loading the snapshot .3db files output by YMAEDA_TOOLS runwaterPML command.

# The snapshot files are stored in the parent directory: PML/snapshot/<station>, and the snapshots
# themselves are stored in a single file for each time step such as: source.Fx.t3.0000.3db
# which stores the snapshots for the time step t = 3.0000. The Fx in the file name indicates
# that the original impulse was applied in the x direction.

# frequency step size used by YMAEDA_TOOLS
df = 0.002441406 
# frequency half space used by YMAEDA_TOOLS
f = np.arange(0, df * 2049, df) 
# frequency full space used by YMAEDA_TOOLS
F = np.arange(0, df * 4096, df) 

def extract_greens_functions():
    """
    This function is used to extract the snapshots for processing into Green's functions.

    """
    return

def extract_snapshot(snapshot_dir, X = 0, Y = 0, Z = 0, t0 = 0.0, t1 = 12.0, dt = 0.1, 
                     exact = False, return_params = False):
    """
    This is the most updated function to read snapshot data output by YMAEDA_TOOLS runwaterPML.
    This function should be a huge improvement over read_snapshot_loc3D_fast!!!
    This function extracts the 3D snapshot for a given X, Y, Z candidate geographical location.
    
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
    exact: bool
        Flag to indicate whether the exact location should be used.
        If False, the snapshot at the grid nearest to the wanted
        location is returned. If True, the snapshot at the exact
        grid point is returned if possible.
    return_params: bool
        Flag to indicate whether the snapshot parameters should be returned.
    
    Returns
    -------
    t: np.array
        array of time steps
    g: np.array
        array of 3D Green's functions
    """
    n_steps = int((t1 - t0) / dt) + 1 # number of time steps
    g = np.zeros([n_steps, 3]) # final output green functions in t; x, y, z
    t = np.arange(t0, t1 + dt, dt) # final output times 
    j = ['x', 'y', 'z'] # 3 axes to read
    
    Ns = [] # number of grid cells for each axis
    x0s = [] # grid cell initial point for each axis.
    dxs = [] # grid step size for each axis.
    
    BYTELEN = 8 # snapshot .3db file byte length
    
    for i, k in enumerate(j): # loop over [x, y, z]
        gt = np.zeros(n_steps)
        for n in range(n_steps): # loop over all time steps. One file for one time step...
            TIME_ZEROS = format(t0 + dt * n, "0.4f")
            snapshot_file = os.path.join(snapshot_dir, "source.F" + k + ".t" + TIME_ZEROS + ".3db")
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
                
                # From the 3D snapshot data, extract only the data for the wanted point [X, Y Z].
                # Convert the geographical coordinates to indices of the binary data:
                idx, idy, idz = snapshot_stnloc(N, x0, dx, X, Y, Z, exact)
                counter = index_convert31(idx, idy, idz, N[0], N[1], N[2])
                L = 60 + BYTELEN * counter # Left most index of the wanted data.
                R = L + BYTELEN            # Right most index of the wanted data.
                gt[n] = struct.unpack("d", fileContent[L:R])[0]
        g[:, i] = gt.copy()
    if return_params == True:
        return t, g, Ns, x0s, dxs
    else:
        return t, g

def extract_snapshot_1d(snapshot_dir, k = "x", X = 0, Y = 0, Z = 0, t0 = 0.0, t1 = 12.0, dt = 0.1, 
                        exact = False, return_params = False):
    """
    This is the 1D version of extract_snapshot.
    """
    n_steps = int((t1 - t0) / dt) + 1 # number of time steps
    g = np.zeros(n_steps) # final output green functions in t; x, y, z
    t = np.arange(t0, t1 + dt, dt) # final output times 
    
    BYTELEN = 8 # snapshot .3db file byte length
    
    for n in range(n_steps): # loop over all time steps. One file for one time step...
        TIME_ZEROS = format(t0 + dt * n, "0.4f")
        snapshot_file = os.path.join(snapshot_dir, "source.F" + k + ".t" + TIME_ZEROS + ".3db")
            
        with open(snapshot_file, mode = "rb") as File:
            fileContent = File.read()
            if n == 0:
                N = struct.unpack("iii", fileContent[0:12]) # number of elements per axis
                x0 = struct.unpack("ddd", fileContent[12:36]) # starting point of each axis
                dx = struct.unpack("ddd", fileContent[36:60]) # distance step size per axis
                data_length = int((len(fileContent) - 60) / BYTELEN)
                    
            # At the end of the data extraction process, Ns, x0s and dxs should have the form:
            # np.array([[Fx params], [Fy params], [Fz params]])
            #assert data_length == N[0] * N[1] * N[2]
                
            # From the 3D snapshot data, extract only the data for the wanted point [X, Y Z].
            # Convert the geographical coordinates to indices of the binary data:
            idx, idy, idz = snapshot_stnloc(N, x0, dx, X, Y, Z, exact)
            counter = index_convert31(idx, idy, idz, N[0], N[1], N[2])
            L = 60 + BYTELEN * counter # Left most index of the wanted data.
            R = L + BYTELEN            # Right most index of the wanted data.
            g[n] = struct.unpack("d", fileContent[L:R])[0]
    if return_params == True:
        return t, g, N, x0, dx
    else:
        return t, g
    
def extract_4pt_snapshot(snapshot_dir, X = 0, Y = 0, Z = 0, t0 = 0.0, t1 = 12.0, dt = 0.1, 
                         exact = False, return_params = False):
    """
    Due to the staggered grid used by YMAEDA_TOOLS PML simulation program, extract_snapshot
    and extract_snapshot_1d do not work as originally intended. 
    
    The x, y and z grids are slightly shifted by about dx / 2 (assuming dx == dy == dz) from 
    each other. Therefore, instead of extracting the snapshot at a given geographical location,
    we need to extract the snapshot at 4 grid points around the candidate grid point.
    
    E.g. for candidate point [-10900, -121100, 1000], we need to load snapshots at:
    .Fx snapshot: 
    [-10900, -121105,  995] >>> [X, Y - dy/2, Z - dz/2]
    [-10900, -121105, 1005] >>> [X, Y - dy/2, Z + dz/2]
    [-10900, -121095,  995] >>> [X, Y + dy/2, Z - dz/2]
    [-10900, -121095, 1005] >>> [X, Y + dy/2, Z + dz/2]
    .Fy snapshot:
    [-10905, -121100,  995] >>> [X - dx/2, Y, Z - dz/2]
    [-10905, -121100, 1005] >>> [X - dx/2, Y, Z + dz/2]
    [-10895, -121100,  995] >>> [X + dx/2, Y, Z - dz/2]
    [-10895, -121100, 1005] >>> [X + dx/2, Y, Z + dz/2]
    .Fz snapshot:
    [-10905, -121105, 1000] >>> [X - dx/2, Y - dy/2, Z]
    [-10905, -121095, 1000] >>> [X - dx/2, Y + dy/2, Z]
    [-10895, -121105, 1000] >>> [X + dx/2, Y - dy/2, Z]
    [-10895, -121095, 1000] >>> [X + dx/2, Y + dy/2, Z]
    """
    
    return
    
def read_snapshot_params(snapshot_file = 'source.Fx.t3.0000.3db'):
    """
    Returns only the parameters of a snapshot .3db file without outputting any data.
    
    Inputs
    ------
    snapshot_file: str
        Path to the .3db snapshot file.
    
    Returns
    -------
    N: tuple
        (Nx, Ny, Nz) number of elements along each axis.
    x0: tuple
        (x0, y0, z0) starting point of each axis.
    dx: tuple
        (dx, dy, dz) step size for each axis.
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
    
    Inputs
    ------
    N: tuple, np.array
        (Nx, Ny, Nz) number of elements along each axis.
    x0: tuple, np.array
        (x0, y0, z0) starting point of each axis.
    dx: tuple, np.array
        (dx, dy, dz) step size for each axis.
        
    Returns
    -------
    X: np.array
        X axis array.
    Y: np.array
        Y axis array.
    Z: np.array
        Z axis array.
    """
    X = np.array([x0[0] + dx[0] * i for i in range(N[0])])
    Y = np.array([x0[1] + dx[1] * i for i in range(N[1])])
    Z = np.array([x0[2] + dx[2] * i for i in range(N[2])])
    return X, Y, Z   

def snapshot_stnloc(N, x0, dx, X_STN, Y_STN, Z_STN, exact = False):
    """
    From the parameters of the snapshot .3db file calculate the nearest 
    grid location for a specified station.
    If the specified location is outside the grid, the grid point nearest 
    to the specified location is returned and a warning is given.
    SMN: -11175, -119878, 1317
    SMW: -12295, -120893, 1110
    LP:  -10900, -121100, 1000
    
    Inputs
    ------
    N: tuple, np.array
        (Nx, Ny, Nz) number of elements along each axis.
    x0: tuple, np.array
        (x0, y0, z0) starting point of each axis.
    dx: tuple, np.array
        (dx, dy, dz) step size for each axis.
    X_STN: float
        Station geographical location (X-coordinate).
    Y_STN: float
        Station geographical location (Y-coordinate).
    Z_STN: float
        Station geographical location (Z-coordinate).
    exact: bool
        Flag to indicate whether the exact location should be used.
        If False, the snapshot at the grid nearest to the wanted
        location is returned. If True, the snapshot at the exact
        grid point is returned if possible.
        
    Returns
    -------
    idx: int
        Index along the X axis corresponding to X_STN.
    idy: int
        Index along the Y axis corresponding to Y_STN.
    idz: int
        Index along the Z axis corresponding to Z_STN.
    """
    X, Y, Z = snapshot_XYZ(N, x0, dx)
    if not(X.min() <= X_STN <= X.max()):
        print('Warning! X out of range! Returning nearest grid value...')
    if not(Y.min() <= Y_STN <= Y.max()):
        print('Warning! Y out of range! Returning nearest grid value...')
    if not(Z.min() <= Z_STN <= Z.max()):
        print('Warning! Z out of range! Returning nearest grid value...')  
    if exact == True:
        idx = np.where(X == X_STN)[0]
        idy = np.where(Y == Y_STN)[0]
        idz = np.where(Z == Z_STN)[0]
        if len(idx) == 0:
            print("Warning! Exact X_STN does not exist! Returning nearest grid value...")
            idx = (abs(X - X_STN)).argmin()
        else:
            idx = idx[0]
        if len(idy) == 0:
            print("Warning! Exact Y_STN does not exist! Returning nearest grid value...")
            idy = (abs(Y - Y_STN)).argmin()
        else:
            idy = idy[0]
        if len(idz) == 0:
            print("Warning! Exact Z_STN does not exist! Returning nearest grid value...")
            idz = (abs(Z - Z_STN)).argmin()  
        else:
            idz = idz[0]
    elif exact == False:
        # Use .argmin() to find the closest index to the wanted point!
        idx = (abs(X - X_STN)).argmin()
        idy = (abs(Y - Y_STN)).argmin()
        idz = (abs(Z - Z_STN)).argmin()  
    return idx, idy, idz

def index_convert13(r, Nx, Ny, Nz):
    """
    Converts the 1D index of GT to the 3D index of GT3D.
    1. the outer most loop: x,
    2. next inner loop: y,
    3. inner most loop: z,
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

################################################################################
# Obsolete functions

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
    
    This function has been superseded by read_snapshot_loc3D_fast.
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
    This faster function should be used instead of read_snapshot_fast.
    
    This function has been superseded by extract_snapshot.
    """
    N = int((t1 - t0) / dt) + 1
    g = np.zeros((N, 3))
    i = 0
    j = ['x', 'y', 'z']
    for k in j: # loop over all 3 axes
        t, g[:, i] = read_snapshot_loc_fast(snapshot_dir, k, X, Y, Z, t0, t1, dt)
        i = i + 1
    return t, g