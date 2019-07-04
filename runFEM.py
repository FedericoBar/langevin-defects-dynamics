import os
import subprocess
import sys
from EOM_integrate import Ncores_FE
# import time

"""
This script runs all the study of Langevin dynamics for the dislocation loops.
It start by initializing the dislocation loops, then computes the force acting
on the loop resulting from the presence of a surface. This force is added to
the interaction term in the overdamped coupled Langevin equations.
One time step after that, the process is reiterated.
"""
# t0 = time.time()

# Get the current working directory
dir = os.getcwd()

# Go into the folder where the Cast3m scripts are
os.chdir(dir + '/Cast3m')

# run the meshing
subprocess.call('castem19 meshing.dgibi', shell=True)
# Go in the python directory and back to compute the stress on the file_mesh_surface
#
os.chdir(dir)
# os.system('python stressBC.py ' + sys.argv[1] + ' ' + sys.argv[2] + ' ' + sys.argv[3])
subprocess.call('python stressBC.py {}'.format(sys.argv[2]), shell=True)
os.chdir(dir + '/Cast3m')
#

# Run the program
# os.system('castem19 solution.dgibi')
subprocess.call('castem19 solution_mod.dgibi', shell=True)

# Run the processing
if Ncores_FE > 0:
    os.environ['MPI_RUNCMD']='mpirun -np {}'.format(Ncores_FE)
subprocess.call('castem19 processing.dgibi', shell=True)
os.environ['MPI_RUNCMD']=''


# Go back in the python scripts directory
if (len(sys.argv) > 2):
    subprocess.run('cp DISPCORR ' + dir + '/Cast3m/OUTPUT/DISPCORR' + sys.argv[1], shell = True)
else:
    subprocess.run('cp DISPCORR ' + dir + '/Cast3m/OUTPUT/DISPCORR', shell = True)

os.chdir(dir)

# print('Total time : ', time.time() - t0)
