import numpy as np
import scipy.sparse as sp
import LaplaceSpheresFunctionsEnvBasic1 as lsf

def electro_linear_operator_FF(sigma_e, sigma_i, c_m, tau, r, L):
    num = (L + 1) ** 2
    A_0, A_1, X_diag_up, X_diag_down = lsf.laplace_MTF_parts_all_basis_1_sphere_FF(sigma_e, sigma_i, r, L)

    M_01_cuartos = - 0.5 * X_diag_down[0:num]
    M_10_medios = X_diag_up[num:(2 * num)]
    M_11 = np.ones((num))

    return 0.


def electro_matrix_operator_FF(sigma_e, sigma_i, c_m, tau, r, L):
    num = (L + 1) ** 2
    A_0, A_1, X_diag_up, X_diag_down = lsf.laplace_MTF_parts_all_basis_1_sphere_FF(sigma_e, sigma_i, r, L)

    MTF_FF_matrix = lsf.MTF_1_sphere_full_matrix(A_0, A_1, X_diag_up, X_diag_down, num)

    M_01_cuartos = - 0.5 * X_diag_down[0:num]
    M_10_medios = X_diag_up[num:(2*num)]
    M_11 = np.ones((num))

    #sp.dia_matrix((M_01_medios*0.5*sigma_e/sigma_i, np.array([0])), shape=(num,num))
    zeros = np.zeros((num, num))
    #sp.dia_matrix((M_01_medios * -0.5, np.array([0])), shape=(num, num))

    costado = np.concatenate((np.concatenate(
        (sp.dia_matrix((M_01_cuartos * sigma_e / sigma_i, np.array([0])), shape=(num, num)).toarray(), zeros), axis=0),
                              np.concatenate(
                                  (sp.dia_matrix((-M_01_cuartos, np.array([0])), shape=(num, num)).toarray(), zeros),
                                  axis=0)), axis=0)
    MTF_FF_matrix = np.concatenate((MTF_FF_matrix, costado), axis=1)
    del costado
    zeros = np.zeros((num, 3*num))
    #sp.dia_matrix(((c_m / tau) * M_11, np.array([0])), shape=(num, num))
    #sp.dia_matrix((M_10_medios*sigma_i, np.array([0])), shape=(num, num))
    abajo = np.concatenate((zeros, np.concatenate((sp.dia_matrix(((c_m / tau) * M_11, np.array([0])), shape=(num, num)).toarray(),
                                                   sp.dia_matrix((M_10_medios * sigma_i, np.array([0])),
                                                                 shape=(num, num)).toarray()),
                                                  axis=1)), axis=1)
    MTF_FF_matrix = np.concatenate((MTF_FF_matrix, abajo), axis=0)
    del abajo

    return MTF_FF_matrix, M_01_cuartos, M_11


def phi_E_dot_x(E):

    def E_dot_x(x):
        return np.dot(E, x)

    return E_dot_x


def i_ion_linear_version(r_m):

    def i_ion(v):
        return v/r_m

    return i_ion

def b_update_constant_step(i_ion, num, sigma_e_i, M_01_cuartos, M_11, c_m, tau):

    def b_function(v_n, v_gorro_n, phi_d, phi_n):
        b = np.empty((5 * num))
        b[0:num] = - 0.5 * sigma_e_i * phi_d - sigma_e_i * M_01_cuartos * v_n
        b[num:2 * num] = -0.5 * sigma_e_i * phi_n
        b[2 * num:3 * num] = 0.5 * phi_d + M_01_cuartos * v_n
        b[3 * num:4 * num] = b[num:2 * num]
        b[4 * num:5 * num] = M_11 * (v_n * c_m / tau - i_ion(v_gorro_n))
        return b

    return b_function
