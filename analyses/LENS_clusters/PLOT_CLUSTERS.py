import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import matplotlib
matplotlib.matplotlib_fname()
import os
import sys
from matplotlib.ticker import MaxNLocator

TRESH = 1
NREPLICA= sys.argv[1]
NFRAMES = 5001
#INPUTSDIRS
DIR = "TRESH_0.4/CLUSTERS_CUTOFF6"
BD = "/leonardo_work/IscrB_REACSIM/METALS/NPTX4_FAST"
OUTDIR = f"{BD}/SCRIPTS/plots/CLUSTERS_CUTOFF6"
######################################

DATAD = f"{BD}/replica_{NREPLICA}/LENS/{DIR}"

os.makedirs(f"{OUTDIR}",exist_ok=True)
COLL = []
SINGOLI = []

factor = 500
xs = np.linspace(0,(NFRAMES-1)/factor,NFRAMES)



for NFRAME in range(NFRAMES):
    coll_count = 0
    singoli_count = 0
    file_name = f"{DATAD}/clusters.{NFRAME}.dat"

    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            for line in file:
                elements = line.split()
                if len(elements) > TRESH:
                    coll_count += 1
                else:
                    singoli_count += 1
    COLL.append(coll_count)
    SINGOLI.append(singoli_count)
    
    
fig, ax = plt.subplots(2,1,dpi=300, figsize=(3,5), facecolor="w", tight_layout=True,sharex=True)
ax[0].plot(xs,COLL,'r')
ax[1].plot(xs,SINGOLI,)


ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))

ax[0].set_ylabel("#eventi collettivi")
ax[1].set_ylabel("#eventi singoli")
ax[1].set_xlabel("Time [ns]")

ax[0].set_title(f"Replica #{NREPLICA}")
fig.savefig(f"{OUTDIR}/replica_{NREPLICA}.png",format='png',bbox_inches='tight')
