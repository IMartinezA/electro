import numpy as np
import scipy.sparse as sp       # Library for sparse matrix formats.


def laplace_bios_pre_computations(L):
    eles = np.arange(0, L + 1)
    l2_1 = 2*eles+1
    eles_1 = eles + 1
    pre_V = 1./l2_1
    K_out_times_minus_1 = eles/l2_1 - 0.5
    K_ast_out = eles_1/l2_1 - 0.5
    pre_W = np.multiply(eles, eles+1)/l2_1

    return pre_V, K_out_times_minus_1, K_ast_out, pre_W


def coefficients_pre_computations_2_mediums(sigma_e, sigma_i):
    sigma_e_i = sigma_e/sigma_i
    sigma_i_e = sigma_i/sigma_e
    return sigma_e_i, sigma_i_e


def laplace_MTF_1_sphere_linear_operator_FF(sigma_e, sigma_i, r, L):
    sigma_e_i, sigma_i_e = coefficients_pre_computations_2_mediums(sigma_e, sigma_i)
    pre_V, K_out_times_minus_1, K_ast_out, pre_W = laplace_bios_pre_computations(L)

    pre_V = pre_V*r
    pre_W = pre_W/r
    num = L + 1

    def MTF_block_matrix_times_vector(v):
        x = np.empty(np.shape(v))

        x[0:num] = sigma_e_i * (np.multiply(K_out_times_minus_1,v[0:num]) + np.multiply(pre_V, v[num:2 * num])) - v[2 * num:3 * num] * 0.5
        x[num:2 * num] = sigma_e_i * (np.multiply(pre_W , v[0:num]) + np.multiply(K_ast_out , v[num:2 * num])) + 0.5 * v[3 * num:4 * num]
        x[2 * num:3 * num] = -0.5 * v[0:num] + np.multiply(K_ast_out , v[2 * num:3 * num]) + np.multiply(pre_V , v[3 * num:4 * num])
        x[3 * num:4 * num] = 0.5 * sigma_e_i * v[num:2 * num] + np.multiply(pre_W , v[2 * num:3 * num]) + np.multiply(K_out_times_minus_1 , v[3 * num:4 * num])
        return x

    def MTF_block_matrix_conjugate_transpose_times_vector(v):
        x = np.empty(np.shape(v))
        x[0:num] = sigma_e_i * (np.multiply(K_out_times_minus_1 * v[0:num]) + np.multiply(pre_W , v[num:2 * num])) - v[2 * num:3 * num] * 0.5
        x[num:2 * num] = sigma_e_i * (np.multiply(pre_V , v[0:num]) + np.multiply(K_ast_out, v[num:2 * num])) + 0.5 * v[3 * num:4 * num]
        x[2 * num:3 * num] = -0.5 * sigma_e_i * v[0:num] + np.multiply(K_ast_out , v[2 * num:3 * num]) + np.multiply(pre_W , v[3 * num:4 * num])
        x[3 * num:4 * num] = 0.5 * v[num:2 * num] + np.multiply(pre_V , v[2 * num:3 * num]) + np.multiply(K_out_times_minus_1 , v[3 * num:4 * num])
        return x

    MTF_linear_operator = sp.linalg.LinearOperator((4 * num, 4 * num),
                                                   matvec=MTF_block_matrix_times_vector,
                                                   matmat=MTF_block_matrix_times_vector,
                                                   rmatvec=MTF_block_matrix_conjugate_transpose_times_vector,
                                                   rmatmat=MTF_block_matrix_conjugate_transpose_times_vector)
    return MTF_linear_operator


def laplace_MTF_1_sphere_full_matrix_FF(sigma_e, sigma_i, r, L):
    sigma_e_i, sigma_i_e = coefficients_pre_computations_2_mediums(sigma_e, sigma_i)
    pre_V, K_out_times_minus_1, K_ast_out, pre_W = laplace_bios_pre_computations(L)

    pre_V = pre_V * r
    pre_W = pre_W / r
    num = L + 1

    MTF_1_sphere_full_matrix = np.zeros((4 * num, 4 * num))
    for j in np.arange(0, num):
        MTF_1_sphere_full_matrix[j, j] = sigma_e_i * K_out_times_minus_1[j]
        MTF_1_sphere_full_matrix[j, num + j] = sigma_e_i * pre_V[j]
        MTF_1_sphere_full_matrix[j, 2 * num + j] = - sigma_e_i * 0.5
        MTF_1_sphere_full_matrix[num + j, j] = sigma_e_i * pre_W[j]
        MTF_1_sphere_full_matrix[num + j, num + j] = sigma_e_i * K_ast_out[j]
        MTF_1_sphere_full_matrix[num + j, 3*num + j] = 0.5
        MTF_1_sphere_full_matrix[2 * num+j, j] = -0.5
        MTF_1_sphere_full_matrix[2 * num+j, 2 * num+j] = K_ast_out[j]
        MTF_1_sphere_full_matrix[2 * num+j, 3 * num+j] = pre_V[j]
        MTF_1_sphere_full_matrix[3 * num+j, num + j] = 0.5 * sigma_e_i
        MTF_1_sphere_full_matrix[3 * num+j, 2 * num+j] = pre_W[j]
        MTF_1_sphere_full_matrix[3 * num+j, 3 * num+j] = K_out_times_minus_1[j]

    return MTF_1_sphere_full_matrix


def rows_columns_A_sparse_1_sphere(num):
    # num = (L + 1) ** 2
    num_A = 4 * num

    rows_A_sparse = np.empty((num_A), dtype=int)
    columns_A_sparse = np.empty((num_A), dtype=int)

    rango = np.arange(0, num)

    number = 0

    s = 1
    s_minus_1_times_2 = (s - 1) * 2

    #V
    rows_A_sparse[number:(number + num)] = rango + num * (s_minus_1_times_2)
    columns_A_sparse[number:(number + num)] = num * (1 + s_minus_1_times_2) + rango
    number = number + num

    #K
    rows_A_sparse[number:(number + num)] = rango + num * (s_minus_1_times_2)
    columns_A_sparse[number:(number + num)] = num * (s_minus_1_times_2) + rango
    number = number + num

    #Kast
    rows_A_sparse[number:(number + num)] = rango + num * (s_minus_1_times_2 + 1)
    columns_A_sparse[number:(number + num)] = num * (s_minus_1_times_2 + 1) + rango
    number = number + num

    #W
    rows_A_sparse[number:(number + num)] = rango + num * (s_minus_1_times_2 + 1)
    columns_A_sparse[number:(number + num)] = num * (s_minus_1_times_2) + rango
    number = number + num

    return rows_A_sparse, columns_A_sparse


def laplace_As_parse(sigma_e_i, r, L):
    pre_V, K_out_times_minus_1, K_ast_out, pre_W = laplace_bios_pre_computations(L)
    pre_V = pre_V * r
    pre_W = pre_W / r

    num = (L + 1) ** 2
    num_A = 4 * num

    A_0 = np.empty((num_A))
    A_1 = np.empty((num_A))

    super_L = np.arange(0, L + 1)

    number = 0

    # V
    A_0[number:(number + num)] = np.repeat(sigma_e_i * pre_V, (super_L * 2 + 1))
    A_1[number:(number + num)] = np.repeat(pre_V, (super_L * 2 + 1))
    number = number + num

    # K
    A_0[number:(number + num)] = np.repeat(sigma_e_i * K_out_times_minus_1, (super_L * 2 + 1))
    A_1[number:(number + num)] = np.repeat(K_ast_out, (super_L * 2 + 1))
    number = number + num

    # Kast
    A_0[number:(number + num)] = np.repeat(sigma_e_i * K_ast_out, (super_L * 2 + 1))
    A_1[number:(number + num)] = np.repeat(K_out_times_minus_1, (super_L * 2 + 1))
    number = number + num

    # W
    A_0[number:(number + num)] = np.repeat(sigma_e_i * pre_W, (super_L * 2 + 1))
    A_1[number:(number + num)] = np.repeat(pre_W, (super_L * 2 + 1))
    #number = number + num
    return A_0, A_1


def laplace_MTF_parts_all_basis_1_sphere_FF(sigma_e, sigma_i, r, L):
    sigma_e_i, sigma_i_e = coefficients_pre_computations_2_mediums(sigma_e, sigma_i)

    num = (L + 1)**2

    rows_A, columns_A = rows_columns_A_sparse_1_sphere(num)
    A_0, A_1 = laplace_As_parse(sigma_e_i, r, L)

    A_0 = sp.coo_matrix((A_0, (rows_A, columns_A)), shape=(2 * num, 2 * num))
    A_1 = sp.coo_matrix((A_1, (rows_A, columns_A)), shape=(2 * num, 2 * num))

    X_diag_up = np.empty((2 * num))
    X_diag_down = np.empty((2 * num))

    s = 1
    s_minus_1_times_2 = (s - 1)*2

    X_diag_up[(s_minus_1_times_2 * num):((s_minus_1_times_2 + 1) * num)] = - sigma_e_i * 0.5
    X_diag_up[((s_minus_1_times_2 + 1) * num):(s * 2 * num)] = 0.5
    X_diag_down[(s_minus_1_times_2 * num):((s_minus_1_times_2 + 1) * num)] = - 0.5
    X_diag_down[((s_minus_1_times_2 + 1) * num):(s * 2 * num)] = sigma_e_i * 0.5

    return A_0, A_1, X_diag_up, X_diag_down


def MTF_1_sphere_full_matrix(A_0, A_1, X_diag_up, X_diag_down, num):

    MTF_FF_matrix = np.concatenate((np.concatenate((A_0.toarray(), sp.dia_matrix((X_diag_up, np.array([0])), shape=(2*num, 2*num)).toarray()), axis=1), np.concatenate((sp.dia_matrix((X_diag_down, np.array([0])), shape=(2*num, 2*num)).toarray(), A_1.toarray()), axis=1)), axis=0)

    return MTF_FF_matrix


def MTF_1_sphere_linear_operator(A_0, A_1, X_diag_up, X_diag_down, num):

    num = 2 * num
    A_0 = A_0.tocsr()
    A_1 = A_1.tocsr()

    def MTF_block_matrix_times_vector(v):
        x = np.empty(np.shape(v))
        v_up = v[0:num]
        v_down = v[num:(2*num)]
        x[0:num] = A_0.dot(v_up) + np.multiply(X_diag_up, v_down)
        x[num:(2 * num)] = np.multiply(X_diag_down, v_up) + A_1.dot(v_down)
        return x

    MTF_linear_operator = sp.linalg.LinearOperator((2 * num, 2 * num),
                                                   matvec=MTF_block_matrix_times_vector,
                                                   matmat=MTF_block_matrix_times_vector)#, rmatvec=MTF_block_matrix_conjugate_transpose_times_vector, rmatmat=MTF_block_matrix_conjugate_transpose_times_vector)
    return MTF_linear_operator