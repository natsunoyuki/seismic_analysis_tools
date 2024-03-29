import numpy as np
from .read_dGM import read_dobs, read_G, Gdbm_to_G, dcv_to_d

# Linear inversion functions used by YMAEDA_TOOLS.

# The inputs are d_obs/dX.cv and G/G.dbm observations and Green function files.
# The outputs are m_est/mX.cv moment tensor files.

# frequency step size used by YMAEDA_TOOLS
df = 0.002441406 
# frequency half space used by YMAEDA_TOOLS
f = np.arange(0, df * 2049, df) 
# frequency full space used by YMAEDA_TOOLS
F = np.arange(0, df * 4096, df) 

def svdinv(G, d, l = 0.01):
    """
    Linear inversion using SVD to get the Penrose inverse.
    For a linear system given by: |d> = G |m>, the model parameters is: |m_est> = Gg |d>.
    The SVD of some matrix G is given by: G = U S V.
    Take only the largest eigenvalues of S, and limit the number of columns of U and rows of V.
    We can then obtain the Penrose inverse using the limited V, S and U.
    
    Inputs
    ------
    G: np.array
        Green's functions matrix.
    d: np.array
        Observed seismogram matrix.
    l: float
        Tolerance value for non-zero SVD eigenvalues.
    
    Returns
    -------
    m: np.array
        Inverted moment tensor.
    """
    u, s, vh = np.linalg.svd(G)
    cond = s > (np.max(s) * l) 
    s = s[:sum(cond)]
    u = u[:,:sum(cond)]
    vh = vh[:sum(cond),:]
    Gg = np.dot(vh.T, np.linalg.inv(np.diag(s)))
    Gg = np.dot(Gg, u.T)
    m = np.dot(Gg, d)
    return m

def lstsqinv(G, d, w = 0):
    """
    Calculate the moment tensor using least squares inversion, with 
    water level regularization, for each frequency step. The inversion
    should be repeated for all frequency steps until the end.
    This function was made to deal with the output files from YMAEDA_TOOLS,
    and will probably not work with "normal" Python style arrays.
    
    Inputs
    ------
    G: np.array
        Green's functions matrix output by YMAEDA_TOOLS.
    d: np.array
        Observed seismogram matrix output by YMAEDA_TOOLS..
    w = float
        Water level regularization parameter.
        
    Returns
    -------
    m: np.array
        Inverted moment tensor, in the YMAEDA_TOOLS output style.
    """
    # as d and G is in the output format of YMAEDA_TOOLS, we need to first
    # convert G to the standard complex number format a + jb:
    GG = Gdbm_to_G(G)
    dd = dcv_to_d(d)
    # since we are dealing with an overdetermined problem here, use least squares:
    GINV = np.dot(np.conj(GG.T), GG)
    # use water level regularization?
    if w > 0:
        if 0 < abs(GINV) <= w:
            GINV = w * GINV / abs(GINV)
        elif GINV == 0:
            GINV = w
    GINV = GINV**-1 * np.conj(GG.T)
    mestlstsq = np.dot(GINV, dd)
    return np.real(mestlstsq)[0], np.imag(mestlstsq)[0]

def gtg_magnitude(main_dir, w = 0, DATLEN = 2049):
    """
    Obtains the magnitude of GINV = inv(G.T G) during least squares inversion.
    ginv = gtg_magnitude(main_dir,0,2049)
    
    Inputs
    ------
    main_dir: str
        Main directory containing the Green's functions files output by YMAEDA_TOOLS.
    w: float
        Water level regularization parameter.
    DATLEN: int
        Data length. Set to 2049 to comply with YMAEDA_TOOLS.
        
    Returns
    -------
    ginv: np.array
        Magnitude of inv(G.T G) during least squares inversion.
    """
    ginv = np.zeros(DATLEN)
    for i in range(DATLEN):
        G = read_G(main_dir, i)
        nrows, ncols = np.shape(G)
        # convert G into standard complex number format: a + ib
        GG = G[0:int(nrows/2)]
        GG = GG[:,0:int(ncols/2)] - GG[:,int(ncols/2):]*1j
        GINV = np.dot(np.conj(GG.T), GG)
        if w > 0:
            if 0 < abs(GINV) <= w:
                GINV = w * GINV / abs(GINV)
            elif GINV == 0:
                GINV = w
        ginv[i] = np.real(GINV) # this is completely real, remove the 0j part
    return ginv

def winv_lstsq(main_dir, w = 0, DATLEN = 2049, dt = 0.1, N_DATA = 6, N_MODEL = 2):
    """
    Read and load data, conduct inversion using least squares for the outputs by YMAEDA_TOOLS.
    Wrapper function for lstsqinv(G, d, w).

    Inputs
    ------
    main_dir: str
        Main directory containing the Green's functions and seismogram files output by YMAEDA_TOOLS.
    w: float
        Water level regularization parameter.
    DATLEN: int
        Data length. Set to 2049 to comply with YMAEDA_TOOLS.
    dt: float
        Time step size (s). Set to 0.1 s to comply with YMAEDA_TOOLS.
    N_DATA: 6
        Number of seismogram traces.
    N_MODEL: 2
        Number of model parameters. 2 is the minimum for 1 real and 1 imaginary part.

    Returns
    -------
    M: np.array
        Inverted seismic moment tensor.
    """
    M = np.zeros([DATLEN, N_MODEL]) # YMAEDA_TOOLS calculated m style
    for i in range(DATLEN):
        # the inversion is carried out for each frequency step and must be
        # repeated for all values. Unfortunately this means for each frequency
        # step new data must be loaded and processed which might slow the
        # inversion process down.
        d = read_dobs(main_dir, i)
        G = read_G(main_dir, i)
        M[i, :] = lstsqinv(G, d, w)
    return M
    
def winv_svd(main_dir, DATLEN = 2049, dt = 0.1, N_DATA = 6, N_MODEL = 2):
    """
    Read and load data, conduct inversion using SVD for the outputs by YMAEDA_TOOLS.
    Wrapper function for svdinv(G, d)
    
    Inputs
    ------
    main_dir: str
        Main directory containing the Green's functions and seismogram files output by YMAEDA_TOOLS.
    DATLEN: int
        Data length. Set to 2049 to comply with YMAEDA_TOOLS.
    dt: float
        Time step size (s). Set to 0.1 s to comply with YMAEDA_TOOLS.
    N_DATA: 6
        Number of seismogram traces.
    N_MODEL: 2
        Number of model parameters. 2 is the minimum for 1 real and 1 imaginary part.

    Returns
    -------
    M: np.array
        Inverted seismic moment tensor.
    """
    M = np.zeros([DATLEN, N_MODEL]) # YMAEDA_TOOLS calculated m style
    for i in range(DATLEN):
        d = read_dobs(main_dir, i)
        G = read_G(main_dir, i)
        M[i, :] = svdinv(G, d)
    return M