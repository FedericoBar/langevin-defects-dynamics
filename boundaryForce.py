import os
import sys
import torch

def boundaryForce(loop):
    dir = os.getcwd()

    os.chdir(dir + '/Cast3m')

    filename = 'Dstrain' + str(loop) + '.txt'

    file_diff = open(filename, 'r')

    raw_diff = file_diff.read().split('\n')

    # Find the 3 index for the 3 line we know contain the gradient coordinates
    line_inds = []
    for line_ind, line in enumerate(raw_diff):
        if (len(line.split()) > 0 and line.split()[0] == 'LISTREEL'):
            line_inds += [line_ind + 1]


    diff_x = [float(x) for x in raw_diff[line_inds[0]].split()]
    diff_y = [float(y) for y in raw_diff[line_inds[1]].split()]
    diff_z = [float(z) for z in raw_diff[line_inds[2]].split()]

    file_diff.close()

    # The data is of the form
    # [epsxx,x  epsyy,x  epszz,x  epsxy,x  epsxz,x  epsyz,x]
    # We wan't to have it as epsij_(x,y,z)


    # We fetch the coordinates of the defect and the dipole tensor components
    # of the dislocation loop

    file_dis_dtensor = open(dir + '/Cast3m/simu/loop' + str(loop) + 'dt.txt', 'r')

    dis_dtensor_init = [float(x) for x in file_dis_dtensor.read().split()]

    dis_dtensor_final = torch.tensor([dis_dtensor_init[0], dis_dtensor_init[4] \
    , dis_dtensor_init[8], dis_dtensor_init[1], dis_dtensor_init[2], dis_dtensor_init[5]])

    # Close the open files
    file_dis_dtensor.close()

    # We fetch the burger vector component for this loop

    file_dis_burger = open(dir + '/Cast3m/simu/loop' + str(loop) + 'bur.txt', 'r')

    dis_burger = torch.tensor([float(x) for x in file_dis_burger.read().split()])

    dis_burger_n = dis_burger.mul(1.0/torch.norm(dis_burger))

    # Now we can compute the force: F = P_ij grad(eps_ij).b/norm(b)      (in N)

    F = 0
    for i in range(6):
        grad = torch.tensor([diff_x[i],diff_y[i],diff_z[i]])
        if (i < 3):
            F += dis_dtensor_final[i] * torch.dot(grad, dis_burger_n)
        else:
            F += 2 * dis_dtensor_final[i] * torch.dot(grad, dis_burger_n)

    os.chdir(dir)
    return F

def boundaryForce3d(loop):
    dir = os.getcwd()

    os.chdir(dir + '/Cast3m')

    filename = 'Dstrain' + str(loop) + '.txt'

    file_diff = open(filename, 'r')

    raw_diff = file_diff.read().split('\n')

    if (loop == 0):
        diff_x = [float(x) for x in raw_diff[3].split()]
        diff_y = [float(y) for y in raw_diff[6].split()]
        diff_z = [float(z) for z in raw_diff[9].split()]
    else:
        diff_x = [float(x) for x in raw_diff[1].split()]
        diff_y = [float(y) for y in raw_diff[3].split()]
        diff_z = [float(z) for z in raw_diff[5].split()]
    file_diff.close()

    # The data is of the form
    # [epsxx,x  epsyy,x  epszz,x  epsxy,x  epsxz,x  epsyz,x]
    # We wan't to have it as epsij_(x,y,z)


    # We fetch the coordinates of the defect and the dipole tensor components
    # of the dislocation loop

    file_dis_dtensor = open(dir + '/Cast3m/simu/loop' + str(loop) + 'dt.txt', 'r')

    dis_dtensor_init = [float(x) for x in file_dis_dtensor.read().split()]

    dis_dtensor_final = torch.tensor([dis_dtensor_init[0], dis_dtensor_init[4] \
    , dis_dtensor_init[8], dis_dtensor_init[1], dis_dtensor_init[2], dis_dtensor_init[5]])

    # Close the open files
    file_dis_dtensor.close()

    # We fetch the burger vector component for this loop

    file_dis_burger = open(dir + '/Cast3m/simu/loop' + str(loop) + 'bur.txt', 'r')

    dis_burger = torch.tensor([float(x) for x in file_dis_burger.read().split()])

    dis_burger_n = dis_burger.mul(1.0/torch.norm(dis_burger))

    # Now we can compute the force: F = P_ij grad(eps_ij)      (in N)

    F = torch.zeros(3)
    for i in range(6):
        grad = torch.tensor([diff_x[i],diff_y[i],diff_z[i]])
        if (i < 3):
            F += dis_dtensor_final[i] * grad
        else:
            F += 2 * dis_dtensor_final[i] * grad

    os.chdir(dir)

    return F

def energy_density(loop):
    dir = os.getcwd()

    os.chdir(dir + '/Cast3m')

    filename = 'Dstrain' + str(loop) + '.txt'

    file_diff = open(filename, 'r')

    raw_diff = file_diff.read().split('\n')

    dE = float(raw_diff[-4].split()[-1])

    os.chdir(dir)

    return dE
