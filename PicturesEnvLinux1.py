import numpy as np
import matplotlib.pyplot as plt

import pyshtools as psh

import LaplaceSpheresFunctionsEnvBasic1 as lsfb1
import HelmholtzSpheresFunctionsEnvLinux1 as hsf
import GraficosFigurasEnvLinux1 as pic

def zerozerozero(x):
    return 0.

# Constant external
one_constant = 1.
def function_E_phi(x):
    return one_constant

L = 0

[minus_cos_theta, weights] = np.polynomial.legendre.leggauss(L + 1)
cos_theta = - minus_cos_theta
del minus_cos_theta

phi = np.linspace(0, 2 * np.pi, num=(2 * L + 1), endpoint=False)
quantity_theta_points = len(cos_theta)
quantity_phi_points = len(phi)

cos_phi = np.cos(phi)
sen_phi = np.sin(phi)
del phi

sen_theta = np.sqrt(1. - np.square(cos_theta))

pre_vector = np.empty((3, quantity_theta_points, quantity_phi_points))
for i in np.arange(0, quantity_theta_points):
    pre_vector[0, i, :] = np.multiply(sen_theta[i], cos_phi)
    pre_vector[1, i, :] = np.multiply(sen_theta[i], sen_phi)
    pre_vector[2, i, :] = cos_theta[i]
del sen_theta
del cos_phi
del sen_phi

mapeo_dirichlet_phi_E0 = np.empty((quantity_theta_points, quantity_phi_points))
for i in np.arange(0,quantity_theta_points):
    for j in np.arange(0,quantity_phi_points):
        mapeo_dirichlet_phi_E0[i, j] = function_E_phi(pre_vector[:, i, j])

temp_phi_d = psh.expand.SHExpandGLQ(mapeo_dirichlet_phi_E0, weights, cos_theta, norm=4, csphase=-1)

num = (L+1)**2

l2_1 = 0
phi_d = np.empty((num))
for l in np.arange(0,L+1):
    phi_d[(l2_1 + l)] = temp_phi_d[0, l, 0]
    for m in np.arange(1, l+1):
        phi_d[(l2_1 + l + m)] = temp_phi_d[0, l, m]
        phi_d[(l2_1 + l - m)] = temp_phi_d[1, l, m]
    l2_1 = l2_1 + 2*l+1

phi_n = -phi_d * 0.

sigma_e_i = 0.5
r = 1.
b = np.empty((4*num,1))
b[0:num,0] = - 0.5 * sigma_e_i * phi_d
b[num:2*num,0] = -0.5 * sigma_e_i * phi_n
b[2*num:3*num,0] = 0.5 * phi_d
b[3*num:4*num,0] = - 0.5 * sigma_e_i * phi_n
#print(b)

A_0, A_1, X_diag_up, X_diag_down = lsfb1.laplace_MTF_parts_all_basis_1_sphere_FF(1., 2., r, L)
full_matrix = lsfb1.MTF_1_sphere_full_matrix(A_0, A_1, X_diag_up, X_diag_down, num)

solutionL = np.linalg.solve(full_matrix, b)

L_c = 1
N = 1
omega = 0.0000001
k = np.array([1., 1.]) * omega
r = np.array([1.])
p = [np.array([0., 0., 0.])]

a = np.ones((1))
a[0] = sigma_e_i

num = (L + 1) ** 2
j_l, j_lp, h_l, h_lp = hsf.helmholtz_pre_computations(N, k, r, L)

almost_A_0 = hsf.cross_interactions_version_1_fast_general(L_c, L, N, k[0], r, p, num, j_l[:, :, 0], j_lp[:, :, 0])
matrix, pp, solucion = hsf.direct_solution_MTF(r, p, a, L, num, N, k, j_l, j_lp, h_l, h_lp, almost_A_0)
#full_matrix[0:2, :] = full_matrix[0:2, :]/sigma_e_i
#print('Diferencias entre las matrices '+str(np.linalg.norm(matrix/2.-full_matrix)))

'''
plt.figure()
plt.imshow(np.real(matrix/2.-full_matrix))
plt.title('Diferencia de las matrices')
plt.colorbar()
'''

print('constant ' + str(np.linalg.norm(solutionL - solucion)))

dd = 3
centro = p[0]
ancho = 5.
alto = 5.
interancho = 10*10*4
interlargo = 10*10*4

potential_u = pic.drawing_cut_with_real_spherical_harmonics_one_sphere_laplace_kernel(dd, centro, ancho, alto, interancho, interlargo, solutionL[:, 0], r, L, function_E_phi)

extent = [-ancho/2., ancho/2.,-ancho/2., ancho/2.]
plt.figure()
maximo = np.max(np.abs(potential_u))
plt.imshow(potential_u, extent=extent, interpolation='None', cmap='seismic', vmin=-maximo, vmax=maximo)
plt.title('Total field.')
plt.colorbar()

potential_u = pic.drawing_cut_with_real_spherical_harmonics_one_sphere_laplace_kernel(dd, centro, ancho, alto, interancho, interlargo, solutionL[:, 0], r, L, zerozerozero)
plt.figure()
maximo = np.max(np.abs(potential_u))
plt.imshow(potential_u, extent=extent, interpolation='None', cmap='seismic', vmin=-maximo, vmax=maximo)
plt.title('Field without the external potential.')
plt.colorbar()


# E constant dot x
E_phi = np.asarray([0.,0.,3.])
def function_E_phi(x):
    return np.dot(E_phi, x)

L = 1

[minus_cos_theta, weights] = np.polynomial.legendre.leggauss(L + 1)
cos_theta = - minus_cos_theta
del minus_cos_theta

phi = np.linspace(0, 2 * np.pi, num=(2 * L + 1), endpoint=False)
quantity_theta_points = len(cos_theta)
quantity_phi_points = len(phi)

cos_phi = np.cos(phi)
sen_phi = np.sin(phi)
del phi

sen_theta = np.sqrt(1. - np.square(cos_theta))

pre_vector = np.empty((3, quantity_theta_points, quantity_phi_points))
for i in np.arange(0, quantity_theta_points):
    pre_vector[0, i, :] = np.multiply(sen_theta[i], cos_phi)
    pre_vector[1, i, :] = np.multiply(sen_theta[i], sen_phi)
    pre_vector[2, i, :] = cos_theta[i]
del sen_theta
del cos_phi
del sen_phi

mapeo_dirichlet_phi_E0 = np.empty((quantity_theta_points, quantity_phi_points))
for i in np.arange(0,quantity_theta_points):
    for j in np.arange(0,quantity_phi_points):
        mapeo_dirichlet_phi_E0[i, j] = function_E_phi(pre_vector[:, i, j])

temp_phi_d = psh.expand.SHExpandGLQ(mapeo_dirichlet_phi_E0, weights, cos_theta, norm=4, csphase=-1)

num = (L+1)**2

l2_1 = 0
phi_d = np.empty((num))
for l in np.arange(0,L+1):
    phi_d[(l2_1 + l)] = temp_phi_d[0, l, 0]
    for m in np.arange(1, l+1):
        phi_d[(l2_1 + l + m)] = temp_phi_d[0, l, m]
        phi_d[(l2_1 + l - m)] = temp_phi_d[1, l, m]
    l2_1 = l2_1 + 2*l+1

phi_n = - phi_d

sigma_e_i = 0.5
r = 1.
b = np.empty((4*num,1))
b[0:num,0] = - 0.5 * sigma_e_i * phi_d
b[num:2*num,0] = -0.5 * sigma_e_i * phi_n
b[2*num:3*num,0] = 0.5 * phi_d
b[3*num:4*num,0] = - 0.5 * sigma_e_i * phi_n
#print('b ' +str(b))

A_0, A_1, X_diag_up, X_diag_down = lsfb1.laplace_MTF_parts_all_basis_1_sphere_FF(1., 2., r, L)
full_matrix = lsfb1.MTF_1_sphere_full_matrix(A_0, A_1, X_diag_up, X_diag_down, num)

solutionL = np.linalg.solve(full_matrix, b)

L_c = 1
r = np.array([1.])

num = (L + 1) ** 2
j_l, j_lp, h_l, h_lp = hsf.helmholtz_pre_computations(N, k, r, L)

almost_A_0 = hsf.cross_interactions_version_1_fast_general(L_c, L, N, k[0], r, p, num, j_l[:, :, 0], j_lp[:, :, 0])
matrix, pp, solucion = hsf.direct_solution_MTF(r, p, a, L, num, N, k, j_l, j_lp, h_l, h_lp, almost_A_0)
#full_matrix[0:2, :] = full_matrix[0:2, :]/sigma_e_i
#print('Diferencias entre las matrices '+str(np.linalg.norm(matrix/2.-full_matrix)))

'''
plt.figure()
plt.imshow(np.real(matrix/2.-full_matrix))
plt.title('Diferencia de las matrices')
plt.colorbar()
'''

print('E dot x: ' + str(np.linalg.norm(solutionL - solucion)))

dd = 2
centro = p[0]
ancho = 4.
alto = 4.
interancho = 10*10*4
interlargo = 10*10*4

#print(solutionL)
#print(solucion)

potential_u = pic.drawing_cut_with_real_spherical_harmonics_one_sphere_laplace_kernel(dd, centro, ancho, alto, interancho, interlargo, solutionL[:, 0], r, L, function_E_phi)
extent = [-ancho/2., ancho/2.,-ancho/2., ancho/2.]
plt.figure()
maximo = np.max(np.abs(potential_u))
plt.imshow(potential_u, extent=extent, interpolation='None', cmap='seismic', vmin=-maximo, vmax=maximo)
plt.title('Total field')
plt.colorbar()

potential_u = pic.drawing_cut_with_real_spherical_harmonics_one_sphere_laplace_kernel(dd, centro, ancho, alto, interancho, interlargo, solutionL[:, 0], r, L, zerozerozero)
plt.figure()
maximo = np.max(np.abs(potential_u))
plt.imshow(potential_u, extent=extent, interpolation='None', cmap='seismic', vmin=-maximo, vmax=maximo)
plt.title('Field without the external potential.')
plt.colorbar()

# Caso onda plana (no fisico)

L = 14

num = (L+1)**2

sigma_e_i = 0.5
r = 1.

A_0, A_1, X_diag_up, X_diag_down = lsfb1.laplace_MTF_parts_all_basis_1_sphere_FF(1., 2., r, L)
full_matrix = lsfb1.MTF_1_sphere_full_matrix(A_0, A_1, X_diag_up, X_diag_down, num)

L_c = 1
N = 1
omega = 2.
k = np.array([1., 1.]) * omega
r = np.array([1.])
p = [np.array([0., 0., 0.])]

a = np.ones((1))
a[0] = sigma_e_i

num = (L + 1) ** 2
j_l, j_lp, h_l, h_lp = hsf.helmholtz_pre_computations(N, k, r, L)

almost_A_0 = hsf.cross_interactions_version_1_fast_general(L_c, L, N, k[0], r, p, num, j_l[:, :, 0], j_lp[:, :, 0])

matrix, pp, solucion = hsf.direct_solution_MTF(r, p, a, L, num, N, k, j_l, j_lp, h_l, h_lp, almost_A_0)

b = np.empty((4*num,1))

b[0:num,0] = 0.5 * sigma_e_i * np.real(pp[0:num, 0])
b[num:2*num,0] = 0.5 * sigma_e_i * np.real(pp[num:2*num, 0])
b[2*num:3*num,0] = 0.5 * np.real(pp[2*num:3*num, 0])
b[3*num:4*num,0] = 0.5 * sigma_e_i * np.real(pp[3*num:4*num, 0])

solutionL = np.linalg.solve(full_matrix, b)

print('plane wave (esta no va a dar cero a menos que se ponga omega muy bajo) ')
print(np.linalg.norm(solutionL - solucion))

error = np.zeros((L-2))
contador = 0
for l in np.arange(2, L):
    num = (l + 1) ** 2
    A_0, A_1, X_diag_up, X_diag_down = lsfb1.laplace_MTF_parts_all_basis_1_sphere_FF(1., 2., r, l)
    full_matrix = lsfb1.MTF_1_sphere_full_matrix(A_0, A_1, X_diag_up, X_diag_down, num)
    L_c = 1
    N = 1
    omega = 2.
    k = np.array([1., 1.]) * omega
    r = np.array([1.])
    p = [np.array([0., 0., 0.])]

    a = np.ones((1))
    a[0] = sigma_e_i


    j_l, j_lp, h_l, h_lp = hsf.helmholtz_pre_computations(N, k, r, l)

    almost_A_0 = hsf.cross_interactions_version_1_fast_general(L_c, l, N, k[0], r, p, num, j_l[:, :, 0], j_lp[:, :, 0])

    matrix, pp, solucion = hsf.direct_solution_MTF(r, p, a, l, num, N, k, j_l, j_lp, h_l, h_lp, almost_A_0)

    b = np.empty((4 * num, 1))

    b[0:num, 0] = 0.5 * sigma_e_i * np.real(pp[0:num, 0])
    b[num:2 * num, 0] = 0.5 * sigma_e_i * np.real(pp[num:2 * num, 0])
    b[2 * num:3 * num, 0] = 0.5 * np.real(pp[2 * num:3 * num, 0])
    b[3 * num:4 * num, 0] = 0.5 * sigma_e_i * np.real(pp[3 * num:4 * num, 0])

    solution = np.linalg.solve(full_matrix, b)
    error[contador] = np.linalg.norm(solution[0:num] - solutionL[0:num])**2 + np.linalg.norm(solutionL[num:(L+1)**2])**2
    error[contador] = error[contador] + np.linalg.norm(solution[num:num*2] - solutionL[(L+1)**2:(L+1)**2+num])**2 + np.linalg.norm(solutionL[(L+1)**2+num:2*(L+1)**2])**2
    error[contador] = error[contador] + np.linalg.norm(solution[num*2:num*3] - solutionL[(L+1)**2*2:(L+1)**2*2+num])**2 + np.linalg.norm(solutionL[(L+1)**2*2+num:3*(L+1)**2])**2
    error[contador] = error[contador] + np.linalg.norm(solution[num*3:num*4] - solutionL[(L+1)**2*3:(L+1)**2*3+num])**2 + np.linalg.norm(solutionL[(L+1)**2*3+num:4*(L+1)**2])**2
    contador = contador + 1

error = np.sqrt(error)

plt.figure()
plt.semilogy(error, marker='x', linestyle='--')
plt.title('Convergence in l2')
plt.xlabel('L (level) (total number of basis is (L+1)(L+1))')

dd = 1
centro = p[0]
ancho = 5.
alto = 5.
interancho = 10 * 10 * 2
interlargo = 10 * 10 * 2

potential_u = pic.drawing_cut_with_real_spherical_harmonics_one_sphere_laplace_kernel(dd, centro, ancho, alto, interancho, interlargo, solutionL[:, 0], r, L, zerozerozero)
extent = [-ancho/2., ancho/2.,-ancho/2., ancho/2.]
plt.figure()
maximo = np.max(np.abs(potential_u))
plt.imshow(potential_u, extent=extent, interpolation='None', cmap='seismic', vmin=-maximo, vmax=maximo)
plt.title('Field without the external potential.')
plt.colorbar()
plt.show()