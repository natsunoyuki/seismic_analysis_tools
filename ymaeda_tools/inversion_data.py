import numpy as np
import struct #for dealing with binary data

# Functions for making the inversion data used by YMAEDA_TOOLS.
# The output should equal to the data in d_obs.cv, G.dbm and m_est.cv files output by YMAEDA_TOOLS.

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
# 