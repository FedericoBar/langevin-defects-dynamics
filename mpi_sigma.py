from mpi4py import MPI
import numpy as np
# import time
import torch
import useful_functions as uf
import pickle
import os


C = uf.define_C(0.278, 1.0065)
dtype = torch.double

def sigma_par(pos_surf_final, dis_position_final, dis_dtensor_final, Nnodes, Nloops):
    """
    This function computes the sigma condition on the surface nodes of the mesh
    and pickles them in 'sigma.pckl'
    Input : nodes pos, loops pos, loops dtensor
    Output : sigma pickled
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    perrank = Nnodes // size
    comm.Barrier()

    temp = torch.zeros((Nnodes, 3, 3), dtype = dtype)
    for ind_position in range(rank*perrank, (rank + 1)*perrank):
        for ind_loop in range(Nloops):
            r = pos_surf_final[ind_position] - dis_position_final[ind_loop]
            temp[ind_position, :, :] -= uf.sigma(dis_dtensor_final[ind_loop], r, 0.278, 1.0065, C)

    comm.Barrier()
    traction = comm.gather(temp, root = 0)
    if rank == 0:
        traction = sum(traction)
        f = open('sigma.pckl', 'wb')
        pickle.dump(traction, f)
        f.close()
        return 0

if __name__ == '__main__':
    dir = os.getcwd()
    f = open(dir + '/init.pckl', 'rb')
    config = pickle.load(f)
    f.close()

    Nnodes = len(config[0])
    Nloops = len(config[1])

    sigma_par(config[0], config[1], config[2], Nnodes, Nloops)
