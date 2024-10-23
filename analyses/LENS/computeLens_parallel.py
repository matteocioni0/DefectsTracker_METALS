#%%
import SOAPify.HDF5er as HDF5er
import SOAPify
import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
from seaborn import kdeplot
import os
import sys
import glob
import re
from multiprocessing import Pool as pollo
import numpy
from numpy import ndarray
from MDAnalysis import Universe, AtomGroup
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch
import h5py

cutoff = 8

# %%
def filter_done_atoms(directory, NATOMS):
    atoms_to_do = list(range(NATOMS))  # Convert to a list
    dat_files = glob.glob(os.path.join(directory, '*.dat'))
    pattern = re.compile(r'ROW\.(\d+)\.dat')
    for dat_file in dat_files:
        file_name = os.path.basename(dat_file)
        match = pattern.match(file_name)
        if match:
            number = int(match.group(1))
            if number in atoms_to_do:
                atoms_to_do.remove(number)
    return atoms_to_do

def check_lens_file(directory,NAME):
    # Use glob to check if any file starts with slipMatrix in the directory
    file_path = f"{directory}/lens_{NAME}.npz"
    if os.path.exists(file_path):
        print(f"HAI GIA FATTO LENS")
        sys.exit()

def lens_per_atom(args):
    inputUniverse, cutOff, atom_idx, trajSlice = args
    return listNeighboursAlongTrajectory(inputUniverse, cutOff, atom_idx, trajSlice)


def listNeighboursAlongTrajectory(
    inputUniverse: Universe, cutOff: float, atom_idx: int, trajSlice: slice = slice(None)
) -> "list[list[AtomGroup]]":

    nnListPerFrame = []
    for ts in inputUniverse.universe.trajectory[trajSlice]:
        nnListPerAtom = []
        nnSearch = AtomNeighborSearch(inputUniverse.atoms, box=inputUniverse.dimensions)
        nnListPerAtom = nnSearch.search(inputUniverse.atoms[atom_idx], cutOff)
        #print(nnListPerAtom)
        nnListPerFrame.append(nnListPerAtom.ix)
    

    nFrames = numpy.asarray(nnListPerFrame, dtype=object).shape[0]
    # this is the number of common NN between frames
    lensArray = numpy.zeros((nFrames))
    # this is the number of NN at that frame
    numberOfNeighs = numpy.zeros((nFrames))
    # this is the numerator of LENS
    lensNumerators = numpy.zeros((nFrames))
    # this is the denominator of lens
    lensDenominators = numpy.zeros((nFrames))
    # each nnlist contains also the atom that generates them,
    # so 0 nn is a 1 element list


    numberOfNeighs[0] = len(nnListPerFrame[0]) - 1
        # let's calculate the numerators and the denominators
    for frame in range(1, nFrames):
        numberOfNeighs[frame] = len(nnListPerFrame[frame]) - 1
        lensDenominators[frame] = (
            len(nnListPerFrame[frame])
            + len(nnListPerFrame[frame - 1])
            - 2
        )
        lensNumerators[frame] = numpy.setxor1d(
            nnListPerFrame[frame], nnListPerFrame[frame - 1]
        ).shape[0]

    denIsNot0 = lensDenominators != 0
    # lens
    lensArray[denIsNot0] = lensNumerators[denIsNot0] / lensDenominators[denIsNot0]
    np.savetxt(f"ROWS_{NAME}/ROW.{atom_idx}.dat",lensArray)
    # return lensArray, numberOfNeighs, lensNumerators, lensDenominators


NAME=sys.argv[1]
#%%
wantedTrajectory = slice(0, None, 1)
trajFileName = f"../{NAME}.hdf5"
trajAddress = f"/Trajectories/{NAME}"
with h5py.File(trajFileName, "r") as trajFile:
    tgroup = trajFile[trajAddress]
    universe = HDF5er.createUniverseFromSlice(tgroup, wantedTrajectory)


# %%
os.makedirs(f"ROWS_{NAME}",exist_ok=True)
_dir = os.getcwd()
rowsdir = f"{_dir}/ROWS_{NAME}"
check_lens_file(_dir,NAME)


# %%
nAtoms = len(universe.atoms)
atoms_to_do = filter_done_atoms(rowsdir, nAtoms)
trajSlice = slice(None)  
args_list = [(universe, cutoff, atom_index, trajSlice) for atom_index in atoms_to_do]

with pollo(processes=31) as pool:
    pool.map(lens_per_atom, args_list)

#ASSEMBLA MATRICE
lens_matrix = np.zeros((nAtoms, universe.trajectory.n_frames))
for i in range(nAtoms):
    a = np.loadtxt(f"ROWS_{NAME}/ROW.{i}.dat")
    lens_matrix[i, :] = a
np.savez(f"lens_{NAME}.npz",lens_matrix)

# %%

