import os
import sys
import numpy as np
import torch
import pickle

import useful_functions as uf
# import time
from EOM_integrate import Np, Nz, nu, mu, Ncores_BC

# t0 = time.time()

dtype = torch.double

def surf_node(ind):
    """
    This function returns the surface a node is on
    Input : index of the node (position of surface nodes...)
    Output : surface the node is on (if it is in more than one surface
    returns all of them)
    """
    surf = []
    node_coor = pos_surf_final[ind]
    if node_coor[0] == 0:
        surf += ['lb']
    if node_coor[0] == Np:
        surf += ['rf']
    if node_coor[1] == 0:
        surf += ['rb']
    if node_coor[1] == Np:
        surf += ['lf']
    if node_coor[2] == 0:
        surf += ['b']
    if node_coor[2] == Nz:
        surf += ['u']
    return surf

# Correspondance between surface name and normal vector
corr_name_norm = {'lb': torch.tensor([[-1],[0],[0]], dtype = dtype), 'rf': torch.tensor([[1],[0],[0]], dtype = dtype), \
                'rb': torch.tensor([[0],[-1],[0]], dtype = dtype), 'lf': torch.tensor([[0],[1],[0]], dtype = dtype), \
                'b': torch.tensor([[0],[0],[-1]], dtype = dtype), 'u': torch.tensor([[0],[0],[1]], dtype = dtype)}

# We go and fetch the coordinates of each node on the surface in 'surf.txt'
dir = os.getcwd()
file_mesh_surface = open(dir + '/Cast3m/surf.txt', 'r')
pos_surf_init = file_mesh_surface.read().split('\n')

pos_surf_final = [[float(x) for x in pos_surf_init[5].split()][1:]]
for line in pos_surf_init[7:-3]:
    pos_surf_final += [[float(x) for x in line.split()][1:]]
file_mesh_surface.close()

pos_surf_final = torch.tensor(pos_surf_final, dtype = dtype)

# We fetch the coordinates of all the defects and their dipole tensor components
dis_position_final = []
dis_dtensor_final = []

os.chdir(dir + '/Cast3m/simu')
list_files = os.listdir(dir + '/Cast3m/simu/')
for file in list_files:
    if ('pos.txt' in file):
        file_dis_position = open(file, 'r')
        dis_position_init = [float(x) for x in file_dis_position.read().split()]
        dis_position_final += [[dis_position_init[0], dis_position_init[1], dis_position_init[2]]]
        # Close the open files
        file_dis_position.close()

    if ('dt.txt' in file):
        file_dis_dtensor = open(file, 'r')
        dis_dtensor_init = [float(x) for x in file_dis_dtensor.read().split()]
        dis_dtensor_final += [[[dis_dtensor_init[0],dis_dtensor_init[1] ,dis_dtensor_init[2]]\
        ,[dis_dtensor_init[3], dis_dtensor_init[4], dis_dtensor_init[5]]\
        ,[dis_dtensor_init[6], dis_dtensor_init[7], dis_dtensor_init[8]]]]
        # Close the open files
        file_dis_dtensor.close()

dis_dtensor_final = torch.tensor(dis_dtensor_final, dtype = dtype)
dis_position_final = torch.tensor(dis_position_final, dtype = dtype)

# We import the useful_function script so that we can compute the
# analytical sigma associated

# Use the functions defined previously to compute the sigma Components
# on the cell boundary.

# os.chdir(dir + '/Cast3m/sigmaSurf')
os.chdir(dir + '/Cast3m')

filename_castem_sigma = dir + '/Cast3m/solution.dgibi'
file_castem_sigma = open(filename_castem_sigma, 'r')

texte = file_castem_sigma.readlines()

file_castem_sigma.close()

ind = -1
for line in texte:
    if ('HERE' in line):
        ind_int = ind + 1
    ind += 1

ind_act = ind_int + 1
texte.insert(ind_act, 'Fload = VIDE CHPOINT/DISC;\n')
ind_act += 1

# Let's compute the elastic compliance tensor
elast_compl = uf.define_C(nu,mu)

# Let's pickle the input data (dtensor, loop position and node position)
f = open('init.pckl', 'wb')
pickle.dump([pos_surf_final, dis_position_final, dis_dtensor_final], f)
f.close()

if (sys.argv[1] == 'True'):
    # Let's compute the sigma with mpiexec (cpu parallelisation)
    os.system('mpiexec -n {} python ../mpi_sigma.py'.format(Ncores_BC))

    # Let's fetch the sigma components computed above
    f = open('sigma.pckl', 'rb')
    stress_node = pickle.load(f)
    f.close()

for ind_position in range(len(pos_surf_final)):
    surfaces_name = surf_node(ind_position)
    surfaces_normv = [corr_name_norm[x] for x in surfaces_name]

    if (sys.argv[1] == 'False'):
        stress_node = torch.zeros((3,3), dtype = dtype)
        for ind_loops in range(len(dis_position_final)):
            r = torch.tensor([pos_surf_final[ind_position][x] - dis_position_final[ind_loops][x] for x in range(3)])
            stress_node -= uf.sigma(dis_dtensor_final[ind_loops], r, nu, mu, elast_compl)

        traction = sum([torch.tensordot(stress_node, surfaces_normv[x], 1).div(2**(len(surfaces_name)-1)) for x in range(len(surfaces_name))])

    else:
        traction = sum([torch.tensordot(stress_node[ind_position], surfaces_normv[x], 1).div(2**(len(surfaces_name)-1)) for x in range(len(surfaces_name))])

    Tx = str(traction[0].item())
    Ty = str(traction[1].item())
    Tz = str(traction[2].item())

    texte.insert(ind_act, 'Fload = Fload ET (FORCE FX (' + Tx + \
                                                ') FY (' + Ty + \
                                                ') FZ (' + Tz + \
                                                ') (surfPoi POIN ' + str(ind_position + 1)+ '));\n')

    ind_act += 1
    nx = str(pos_surf_final[ind_position][0].item() - Np/2.0)
    ny = str(pos_surf_final[ind_position][1].item() - Np/2.0)
    nz = str(pos_surf_final[ind_position][2].item() - Nz/2.0)


    texte.insert(ind_act, 'Mtotx = Mtotx + ('+ny+'*'+Tz+') - ('+nz+'*'+Ty+');\n')
    ind_act += 1
    texte.insert(ind_act, 'Mtoty = Mtoty + ('+nz+'*'+Tx+') - ('+nx+'*'+Tz+');\n')
    ind_act += 1
    texte.insert(ind_act, 'Mtotz = Mtotz + ('+nx+'*'+Ty+') - ('+ny+'*'+Tx+');\n')
    ind_act += 1

    # file_sigma = open(dir + '/Cast3m/sigmaSurf/sigma' + str(ind_position) + '.txt', 'w')
filename_castem_sigma = dir + '/Cast3m/solution_mod.dgibi'
file_castem_sigma = open(filename_castem_sigma, 'w')
for line in texte:
    file_castem_sigma.write(line)

file_castem_sigma.close()
# print('Total time : ', time.time() - t0, ' s' )
