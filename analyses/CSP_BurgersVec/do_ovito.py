# Set it to false to switch back to conventional for-loop processing (included for comparison).
use_multiprocessing = True

if use_multiprocessing:
    # Disable internal parallelization of OVITO's pipeline system, which otherwise would perform certain
    # computations using all available processor cores. Setting the environment variable OVITO_THREAD_COUNT=1
    # must be done BEFORE importing the ovito module and will restrict OVITO to a single CPU core per process.
    import os
    os.environ["OVITO_THREAD_COUNT"] = "1"

# Boilerplate code generated by OVITO Pro 3.8.5
from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from ovito.pipeline import *
import sys
import os
import numpy as np
from time import time

# Set up the OVITO data pipeline for performing some heavy computations for each trajectory frame.
# Keep in mind that this initialization code gets executed by EVERY worker process in the multiprocessing pool.
# So we should only prepare but not carry out any expensive operations here.
pipeline = import_file(f"scnemd.lammpsdump", multiple_frames = True)

# Dislocation analysis (DXA):
pipeline.modifiers.append(DislocationAnalysisModifier())

# Centrosymmetry parameter:
pipeline.modifiers.append(CentroSymmetryModifier())

def process_frame(frame):
    data = pipeline.compute(frame)
    DXA = [data.attributes["SourceFrame"],
           data.attributes["DislocationAnalysis.counts.BCC"],
            data.attributes["DislocationAnalysis.counts.CubicDiamond"],
            data.attributes["DislocationAnalysis.counts.FCC"],
            data.attributes["DislocationAnalysis.counts.HCP"],
            data.attributes["DislocationAnalysis.counts.HexagonalDiamond"],
            data.attributes["DislocationAnalysis.counts.OTHER"],
            data.attributes["DislocationAnalysis.length.1/2<110>"],
            data.attributes["DislocationAnalysis.length.1/3<100>"],
            data.attributes["DislocationAnalysis.length.1/3<111>"],
            data.attributes["DislocationAnalysis.length.1/6<110>"],
            data.attributes["DislocationAnalysis.length.1/6<112>"],
            data.attributes["DislocationAnalysis.length.other"],
            data.attributes["DislocationAnalysis.total_line_length"],]
    CS_SUM = np.sum(data.particles.centrosymmetry[:])
    CS = data.particles.centrosymmetry[:]
    return DXA, CS_SUM, CS

# Main program entry point:
if __name__ == '__main__':

    # Measure time for benchmarking purposes.
    from time import time
    t_start = time()

    if use_multiprocessing:

        # Force "spawn" start method on all platforms, because OVITO is not compatible with "fork" method.
        import multiprocessing as mp
        mp.set_start_method('spawn')

        # Create a pool of processes which will process the trajectory frames in parallel.
        with mp.Pool(None) as pool:
            results = list(pool.imap(process_frame, range(pipeline.source.num_frames))) #range(pipeline.source.num_frames)

    else:

        # Conventional for-loop iterating over all frames of the trajectory and processing one by one.
        results = [process_frame(frame) for frame in range(30)]


    DXA, CS_SUM, CS = zip(*results)
    os.makedirs('CS',exist_ok=True)
    os.makedirs('DXA',exist_ok=True)
    np.savetxt("CS/CS_SUM.dat",CS_SUM,fmt='%.4f')
    np.savetxt("CS/CS.dat",CS,fmt='%.4f')

    dxa_fmt = ('%d','%d','%d','%d','%d','%d','%d','%.2f','%.2f','%.2f','%.2f','%.2f','%.2f','%.2f',)
    np.savetxt("DXA/DXA.dat",DXA,fmt=dxa_fmt)

    t_end = time()

    print(f"Computation took {t_end - t_start} seconds")


