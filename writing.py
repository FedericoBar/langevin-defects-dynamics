import os
import time
import copy
import numpy as np


def control(curr_node, time_iter, cell_id, force_boundary, force_interaction, xi, t1, kB, T, step):
    """
    This function prints some control data. It is called during the iteration process.
    """
    # Coorinates of the cell which id is cell_id
    i = cell_id[0]
    j = cell_id[1]
    k = cell_id[2]

    # Control : printing values for diagnostic and for keeping track.
    if (time_iter%step == 0):
        print('Iteration number : ', time_iter)
        print('cell number : ', i, j, k)
        print('node number : ', curr_node.id)
        print('node tensor : ', curr_node.val.dtensor)
        print('node burger : ', curr_node.val.burger)
        print('node size : ', curr_node.val.size)
        print('force_boundary :')
        print(-curr_node.val.diffusivity(T) / (kB * T) * force_boundary)
        print('\n')
        print('force_interaction :')
        print(-curr_node.val.diffusivity(T) / (kB * T) * force_interaction)
        print('\n')
        # print(curr_node.val.size)
        # print(curr_node.val.dtensor)
        # print('\n')
        print('force_random :')
        print(np.sqrt(2*curr_node.val.diffusivity(T)) * xi)
        print('\n')
        # print('Position :')
        # print(curr_node.val.position)
        # print('\n')
        print('TIME FOR 1 LOOP : ', time.time() - t1)
        print('\n')
    return 0


"""""""""""""""""""""

\\\\\\\\\\\\\\\\\\\\\
  FUNCTIONS for Zll
\\\\\\\\\\\\\\\\\\\\\

"""""""""""""""""""""

def init_write(example, Np, Nl, T, FEM, dt):
    """
    This function initialise the output file containingthe position of the loops
    """
    dir = os.getcwd()
    filename = dir + '/OUTPUT/out{}'.format(example) + '_N' + str(Np) + '_Nl' + str(Nl) + '_T' + str(T) + '_FEM' + str(FEM) + '.txt'
    file = open(filename, 'w')
    file.write('Iter' + '\t' + 'Loop' + '\t' + 'px' + '\t' + 'py' + '\t' + 'pz' + \
    '\t' + 'bx' + '\t' + 'by' + '\t' + 'bz' + \
    '\t' + 'Fint' + \
    '\t' + 'FBC' + \
    '\t' + 'OmeF' + \
    '\t' + 'size' + \
    '\t' + 'nature' +\
    '\t' + 'time' +\
    '\t' + 'dt' + '\n')
    file.close()
    return 0

def write_param(Np, Nz, Nm, Nloops, nu, E):
    """
    This function writes the essential parameters for Cast3m's input
    """
    dir = os.getcwd()
    filename = dir + '/Cast3m/param.txt'
    file = open(filename, 'w')
    file.write(str(Np) + '\t' + str(Nz) + '\t' + str(Nm) + '\t' + str(Nloops) + '\t' + str(nu) + '\t' + str(E) + '\n')
    file.close()
    return 0

def write_Zll(Zll, iter):
    """
    This function writes down all what is essential for Cast3m (stressBC.py and processing.dgibi)
    which is : the dipole tensor coordinates, the burger vectors and position of all the loops
    """

    dir = os.getcwd()

    os.chdir(dir + '/Cast3m')
    foldername = 'simu'

    # Clear previous loops
    os.system('rm -R ' + foldername)
    os.mkdir(foldername)
    os.chdir(dir + '/Cast3m/' + foldername)


    # number of cell per side
    # print('\n next \n')
    dim_Zllp = Zll.shape[0]
    dim_Zllz = Zll.shape[2]
    for i in range(dim_Zllp):
        for j in range(dim_Zllp):
            for k in range(dim_Zllz):
                # curr_list is the linked list system associated with the cell i,j,k
                curr_list = Zll[i,j,k]
                # curr_node will run through all the nodes of curr_list
                curr_node = curr_list.head


                # # Run through all the nodes of curr_list
                while(curr_node != None):
                    # print('current node index : ', curr_node.id, '\t')

                    # File for the position coordinates
                    filename1 = 'loop' + str(curr_node.id) + 'pos.txt'
                    file1 = open(filename1,'w')
                    # File for the dipole tensor coordinates
                    filename2 = 'loop' + str(curr_node.id) + 'dt.txt'
                    file2 = open(filename2, 'w')
                    # File for the burger vector coordinates
                    filename3 = 'loop' + str(curr_node.id) + 'bur.txt'
                    file3 = open(filename3, 'w')
                    # File for the size of the loops
                    filename4 = 'loop' + str(curr_node.id) + 'size.txt'
                    file4 = open(filename4, 'w')

                    file4.write(str(curr_node.val.size.item()))

                    for ii in range(3):
                        file1.write(str(curr_node.val.position[ii].item()) + '\t')
                        file3.write(str(curr_node.val.burger[ii].item()) + '\t')
                    for ii in range(3):
                        for jj in range(3):
                            dipole_tensor_comp = curr_node.val.dtensor[ii,jj]
                            file2.write(str(dipole_tensor_comp.item()) + '\t')

                    # Go to the next node
                    curr_node = curr_node.next

                    file1.close()
                    file2.close()
                    file3.close()
                    file4.close()

    os.chdir(dir)
    return 0

def write_pos_ll(Zll, iter, example, Np, Nl, T, FEM, dt, F_int, F_BC, duration, F_volume):
    """
    This function saves in a text file all the data we need for analysis
    Input : Zll
    Output : text file with position of loops during the simulation
    """
    Nloops_inter = 0
    Nloops_sessile = 0
    dir = os.getcwd()
    filename = dir + '/OUTPUT/out{}'.format(example) + '_N' + str(Np) + '_Nl' + str(Nl) + '_T' + str(T) + '_FEM' + str(FEM) + '.txt'
    file = open(filename, 'r')

    data_tmp = file.readlines()
    file.close()

    dim_Zllp = Zll.shape[0]
    dim_Zllz = Zll.shape[2]
    for i in range(dim_Zllp):
        for j in range(dim_Zllp):
            for k in range(dim_Zllz):
                curr_list = Zll[i,j,k]
                curr_node = curr_list.head
                while(curr_node != None):
                    Nloops_inter += 1
                    if curr_node.val.diffusivity(T) == 0:
                        Nloops_sessile += 1
                    line = str(iter) + '\t' + str(curr_node.id) + '\t'
                    for ii in range(3):
                        line = line + str(round(curr_node.val.position[ii].item(), 1)) + '\t'
                    for ii in range(3):
                        line = line + str(curr_node.val.burger[ii].item()) + '\t'

                    line = line + str(F_int[curr_node.id].item()) + '\t'
                    line = line + str(F_BC[curr_node.id].item()) + '\t'

                    line = line + str(F_volume) + '\t'

                    line = line + str(curr_node.val.size.item()) + '\t'

                    line = line + str(curr_node.val.kind(T)) + '\t'

                    line = line + str(duration) + '\t'

                    line = line + str(dt) + '\n'

                    data_tmp.append(line)

                    curr_node = copy.deepcopy(curr_node.next)

    file = open(filename, 'w')
    for text in data_tmp:
        file.write(text)
    file.close()

    return Nloops_inter, Nloops_sessile


"""""""""""""""""""""

\\\\\\\\\\\\\\\\\\\\\
 Obsolete functions
\\\\\\\\\\\\\\\\\\\\\

"""""""""""""""""""""

"""""""""""""""""""""

\\\\\\\\\\\\\\\\\\\\\
   FUNCTIONS for Z
\\\\\\\\\\\\\\\\\\\\\

"""""""""""""""""""""


def write_loops(Z, iter):
    """
    This function write down, in a text file loop_n situated in /Documents/Cast3m/simu_date
    his position, burger vector and dipole tensore
    Input : The dislocation loops at any given moment Z and the time iteration iter
    Output : The file in the directory mentionned above containing all the loop properties
    """

    dir = os.getcwd()

    os.chdir(dir + '/Cast3m')
    # File with the current time step
    filename0 = 'Nstep.txt'
    file0 = open(filename0, 'w')
    file0.write(str(iter))
    file0.close()

    foldername = 'simu'

    if (iter == 0):
        os.system('rm -R ' + foldername)
        os.mkdir(foldername)
    os.chdir(dir + '/Cast3m/' + foldername)

    for loop in range(len(Z)):
        # File for the position
        filename1 = 'loop' + str(iter) + '_' + str(loop) + 'pos.txt'
        file1 = open(filename1,'w')
        # File for the dipole tensor coordinates
        filename2 = 'loop' + str(iter) + '_' + str(loop) + 'dt.txt'
        file2 = open(filename2, 'w')
        # File for the burger vector coordinates
        filename3 = 'loop' + str(iter) + '_' + str(loop) + 'bur.txt'
        file3 = open(filename3, 'w')
        # File for the size of the loops
        filename4 = 'loop' + str(iter) + '_' + str(loop) + 'size.txt'
        file4 = open(filename4, 'w')

        file4.write(str(Z[loop].size))


        for i in range(3):
            file1.write(str(Z[loop].position[i]) + '\t')
            file3.write(str(Z[loop].burger[i]) + '\t')
        for i in range(3):
            for j in range(3):
                # We write in the appropriate dimension (J)
                dipole_tensor_comp = Z[loop].dtensor[i,j]
                file2.write(str(dipole_tensor_comp) + '\t')
        file1.close()
        file2.close()
        file3.close()
        file4.close()

    os.chdir(dir)
    return 0

def write_Nloops(N):
    """
    This function saves the number of loops in a file,
    Cast3m will load this number in the processing
    """
    dir = os.getcwd()
    filename = dir + '/Cast3m/Nloops.txt'
    file1 = open(filename, 'w')
    file1.write(str(N))
    file1.close()
    return 0

def write_force(F):
    """
    This function writes the force during the Langevin EOM Resolution
    in a file /OUTPUT/F_???.txt
    Input : Force F (list of values)
    Output : text file with the force values
    """

    dir = os.getcwd()
    filename = dir + '/OUTPUT/' + str(F[0]) + '.txt'
    file = open(filename, 'w')
    for force in F[1:]:
        file.write(str(float(force)))
        file.write('\n')
    file.close()

    return 0

def write(Z):
    """
    This function saves in a text file all the data we need for analysis
    Input : Z
    Output : text file with position of loops during the simulation
    """

    dir = os.getcwd()
    filename = dir + '/OUTPUT/position.txt'
    file = open(filename, 'w')
    file.write('Iter' + '\t' + 'Loop' + '\t' + 'px' + '\t' + 'py' + '\t' + 'pz' + '\n')
    for time_iter in range(len(Z)):
        for loop_ind in range(len(Z[time_iter])):
            file.write(str(time_iter) + '\t' + str(loop_ind) + '\t')
            for i in range(3):
                pos_i = str(round(Z[time_iter][loop_ind].position[i].item(), 1))
                file.write(pos_i + '\t')
            file.write('\n')

    file.close()
    return 0
