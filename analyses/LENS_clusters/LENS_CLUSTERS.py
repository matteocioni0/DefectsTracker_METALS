from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from ovito.pipeline import *
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
from time import time
import warnings


NREPLICA= sys.argv[1]
DIR = "TRESH_0.4"
BD = "/leonardo_work/IscrB_REACSIM/METALS/NPTX4_FAST"

_td = f"{BD}/replica_{NREPLICA}/LENS/{DIR}/TRJ"
OUTD = f"{BD}/replica_{NREPLICA}/LENS/{DIR}/CLUSTERS_CUTOFF6"
os.makedirs(OUTD,exist_ok=True)


pipeline = import_file(f"{_td}/frame.*.lammpsdump", multiple_frames = True)
# Cluster analysis:
pipeline.modifiers.append(ClusterAnalysisModifier(
    cutoff = 6.0, 
    sort_by_size = True, 
    unwrap_particles = True, 
    cluster_coloring = True))


for NFRAME in range(pipeline.source.num_frames):
    data = pipeline.compute(NFRAME)
    with open(f"{OUTD}/clusters.{NFRAME}.dat", 'w') as file:
        NCLUSTERS = data.attributes['ClusterAnalysis.cluster_count']
        if NCLUSTERS == 0:
            pass
        else:
            for NCLUSTER in range(1,NCLUSTERS+1):
                a = data.particles.identifiers[:]
                a = a[data.particles.cluster==NCLUSTER]
                file.write(' '.join(map(str, a)))
                file.write("\n")
