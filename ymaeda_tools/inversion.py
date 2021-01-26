import numpy as np

from .read_dGM import read_dobs, read_G

# Linear inversion functions used by YMAEDA_TOOLS

# frequency step size used by YMAEDA_TOOLS
df = 0.002441406 
# frequency half space used by YMAEDA_TOOLS
f = np.arange(0, df*2049, df) 
# frequency full space used by YMAEDA_TOOLS
F = np.arange(0, df*4096, df) 

def svdinv(G, d, l = 0.01):
    """
    Linear inversion using SVD to get the Penrose inverse
    For a linear system given by: |d> = G |m>
    |m_est> = Gg |d>
    The SVD of some matrix G is given by: G = U S V
    Take only the largest eigenvalues of S, and limit the number of columns
    of U and rows of V 
    We can then obtain the Penrose inverse using the limited V, S and U.
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
    w = water level regularization parameter
    """
    nrows, ncols = np.shape(G)
    # as G is in the output format of YMAEDA_TOOLS, we need to first
    # convert G to the standard complex number format a + ib:
    GG = G[0:int(nrows/2)]
    GG = GG[:,0:int(ncols/2)] - GG[:,int(ncols/2):]*1j
    # since we are dealing with an overdetermined problem here, use least squares:
    GINV = np.dot(np.conj(GG.T), GG)
    # .T is faster than transpose()
    # print(GINV)
    # use water level regularization?
    if w > 0:
        if 0 < abs(GINV) <= w:
            GINV = w * GINV / abs(GINV)
        elif GINV == 0:
            GINV = w
    GINV = GINV**-1 * np.conj(GG.T)
    mestlstsq = np.dot(GINV, d[0:int(len(d)/2)] + d[int(len(d)/2):]*1j)
    return np.real(mestlstsq)[0], np.imag(mestlstsq)[0]

def gtg_magnitude(main_dir, w = 0, DATLEN = 2049):
    """
    ginv = gtg_magnitude(main_dir,0,2049)
    obtains the magnitude of GINV=inv(G.T G) during least squares inversion
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
    Read and load data, conduct inversion using least squares
    w = water level regularization parameter
    """
    M = np.zeros([DATLEN, N_MODEL]) # YMAEDA_TOOLS calculated m
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
    Read and load data, conduct inversion using SVD.
    """
    M = np.zeros([DATLEN, N_MODEL]) # YMAEDA_TOOLS calculated m
    for i in range(DATLEN):
        d = read_dobs(main_dir, i)
        G = read_G(main_dir, i)
        M[i, :] = svdinv(G, d)
    return M