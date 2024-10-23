import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import matplotlib
matplotlib.matplotlib_fname()
import os


_bd=f"/leonardo_work/IscrB_REACSIM/METALS/NPTX4_FAST"
NREPLICAS = 5
tresh=0.4
FOLDER="TRESH_0.4"


for NREPLICA in range(5,NREPLICAS+1):
	_d = f"{_bd}/replica_{NREPLICA}/LENS"
	lens_f=f"{_d}/lens_scnemd.npz"

	#leggo
	# NATOMI x NFRAMES
	LENS = np.load(lens_f)['arr_0']
	#TEMPO
	n_frames = LENS.shape[1]
	n_atoms = LENS.shape[0]
	######
	# ATOMO PER ATOMO
	#####
	l_max = np.max(LENS)
	l_min = np.min(LENS)
	LENS_N  = (LENS - l_min)/(l_max-l_min)
	_td=f"{_d}/{FOLDER}"
	os.makedirs(_td,exist_ok=True)
	for frame in range(0,n_frames):
	    mask = LENS_N[:, frame] > tresh
	    indices = np.where(mask)[0] +1
	    np.savetxt(f"{_td}/IDs_frame{frame}.dat",indices,fmt='%d')
