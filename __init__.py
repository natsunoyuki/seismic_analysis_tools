# seismic_analysis_tools

# Directories
# -----------
# 1. sac_tools
#    A set of python codes which uses the obspy library to deal with .sac data. 
#    In particular, the codes in this directory allow for quick conversion to pd.DataFrames for a more pythonic seismic data analysis.
# 2. ymaeda_tools
#    A set of python codes associated with the linear inversion C code (YMAEDA_TOOLS) written originally by Dr. Yuta Maeda of Nagoya University.

# Jupyter Notebooks
# -----------------
# 1. Sac_test.ipynb
#    Jupyter notebook containing tests of the Sac.py codes in sac_tools.
# 2. Shinmoedake.ipynb
#    Jupyter notebook containing tests of the Shinmoedake.py codes in sac_tools.
# 3. ymaeda_tools_gf_checks.ipynb
#    Jupyter notebook containing checks of the Green's functions calculated by runwaterPML.
# 4. ymaeda_tools_inv_checks.ipynb
#    Jupyter notebook containing checks of the linear inversion results calculated by winv.
# 5. ymaeda_tools_dGM_check.ipynb
#    Jupyter notebook containing checks of the d_obs.cv, m_est.cv and G.dbm files created by winv.