from mpi4py import MPI
import pickle
import numpy as np
import copy
import time

from EOM_integrate import dim_Zllp, dim_Zllz, DzU_ll, Nloops
import linked
import EOM_struct as EOMs
import useful_functions as uf


cell_to_ind = np.array([0 for i in range(dim_Zllp**2*dim_Zllz)], dtype = list)
ind = 0
for i in range(dim_Zllp):
    for j in range(dim_Zllp):
        for k in range(dim_Zllz):
            cell_to_ind[ind] = [i,j,k]
            ind += 1

def DzU():
    """
    This function computes the elastic interaction potential.
    with multiple CPU's
    Input : Cut_off radius
    Output : Upot
    """

    f = open('zll.pckl', 'rb')
    Zll = pickle.load(f)
    f.close()
    f = open('rzll.pckl', 'rb')
    reduced_Zll = pickle.load(f)
    f.close()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    perrank = dim_Zllp**2*dim_Zllz//size
    comm.Barrier()

    temp = np.zeros(Nloops)
    for ind_loop in range(rank*perrank, (rank+1)*perrank):
        ind = 0

        i = cell_to_ind[ind_loop][0]
        j = cell_to_ind[ind_loop][1]
        k = cell_to_ind[ind_loop][2]

        curr_list = Zll[i,j,k]
        curr_node = curr_list.head

        while (curr_node != None):
            temp[curr_node.id] = DzU_ll(Zll, reduced_Zll, curr_node, [i,j,k], ind)
            ind += 1
            curr_node = copy.deepcopy(curr_node.next)


    comm.Barrier()
    DzU = comm.gather(temp, root = 0)
    if rank == 0:
        DzU = sum(DzU)
        f = open('dzu.pckl', 'wb')
        pickle.dump(DzU, f)
        f.close()
        return DzU

if __name__ == '__main__':
    # Cut off radius 1
    DzU()
