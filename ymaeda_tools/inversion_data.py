import numpy as np
import struct

# Functions for making the inversion data used by YMAEDA_TOOLS.
# The inputs for the Green functions are the .3db files such as "source.Fx.t3.0000.3db".
# The inputs for the observations are the d.seq1 files.
# The output should equal to the data in d_obs.cv, G.dbm files output by YMAEDA_TOOLS.

# The functions for loading d_obs.cv, G.dbm and m_est.cv files output by YMAEDA_TOOLS
# are found in read_dGM.py.

# Refer to winv_sub/make_green_seq.h for Green functions.
# Refer to winv_sub/make_data_seq.h for seismograms.

from .read_snapshot import extract_snapshot

# frequency step size used by YMAEDA_TOOLS
df = 0.002441406 
# frequency half space used by YMAEDA_TOOLS
f = np.arange(0, df * 2049, df) 
# frequency full space used by YMAEDA_TOOLS
F = np.arange(0, df * 4096, df) 

# The ith component of the ground displacement u_i(x, t) is given by the relation:
#
#       d G_ix        d G_iy        d G_iz        d G_ix        d G_iy        d G_iz    
# u_i = ------ M_xx + ------ M_yy + ------ M_zz + ------ M_xy + ------ M_yz + ------ M_zx.
#        d x           d y           d z           d y           d z           d x
# 
# The 6 independent moment tensor components can be cast into the following 1D np.array:
#
#     M_xx
#     M_yy
# m = M_zz
#     M_xy
#     M_yz
#     M_zx
# 
# which allows us to cast dG/dx in the form of the following 2D np.array with the ith row having the 6 components:
#
#       d G_ix  d G_iy  d G_iz  d G_ix  d G_iy  d G_iz    
# G_i = ------, ------, ------, ------, ------, ------.
#        d x     d y     d z     d y     d z     d x
#
# The spatial derivatives are approximated as follows:
#
# d G_ij    1   [ d G_ij   d G_ik ]
# ------ := - . [ ------ + ------ ].
#  d k      2   [  d k      d j   ]
#
# E.g. for Shinmoedake, with 2 stations SMN and SMW with 3 components, we have 6 of these equations:
#
#         d SM$_ix        d SM$_iy        d SM$_iz        d SM$_xx        d SM$_iy        d SM$_iz
# sm$_i = -------- M_xx + -------- M_yy + -------- M_zz + -------- M_xy + -------- M_yz + -------- M_zx,
#           d x             d y             d z             d y             d z             d x
#
# where $ = {N, W}, i = {x, y, z}. sm$_i is the observed long period event time series, and SM$_ij is the
# ith component of the Green's functions due to an impulse in the jth direction at station SM$.
#
