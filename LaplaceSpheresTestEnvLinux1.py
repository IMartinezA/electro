import numpy as np
import pyshtools as psh
import LaplaceSpheresFunctionsEnvBasic1 as lsfb1

#Testing #phantom #campo 0
E_phi = np.asarray([0.,0.,3.])#np.zeros((3))
def function_E_phi(x):
    return np.dot(E_phi, x) #+3.

L = 1

zeros, w = psh.expand.SHGLQ(L)

[minus_cos_theta, weights] = np.polynomial.legendre.leggauss(L + 1)
cos_theta = - minus_cos_theta
print(np.linalg.norm(cos_theta-zeros)) #equivalentes
print(np.linalg.norm(weights-w)) # equivalentes
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

phi_d = np.empty((num))
l2_1 = 0
for l in np.arange(0,L+1):
    phi_d[(l2_1 + l)] = temp_phi_d[0, l, 0]
    for m in np.arange(1, l+1):
        phi_d[(l2_1 + l + m)] = temp_phi_d[0, l, m]
        phi_d[(l2_1 + l - m)] = temp_phi_d[1, l, m]
    l2_1 = l2_1 + 2*l+1

print(phi_d)

phi_n = -phi_d#(phi_d-3.)

sigma_e_i = 1.

b = np.empty((4*num,1))
b[0:num,0] = - 0.5 * sigma_e_i * phi_d
b[num:2*num,0] = -0.5 * sigma_e_i * phi_n
b[2*num:3*num,0] = 0.5 * phi_d
b[3*num:4*num,0] = b[num:2*num,0]
print(b)
r=1.
A_0, A_1, X_diag_up, X_diag_down = lsfb1.laplace_MTF_parts_all_basis_1_sphere_FF(1., 1., r, L)
full_matrix= lsfb1.MTF_1_sphere_full_matrix(A_0, A_1, X_diag_up, X_diag_down, num)

solution3 = np.linalg.solve(full_matrix, b)
import HelmholtzSpheresFunctionsEnvLinux1 as hsf

L_c = 1 #67
L = 1 #66
N = 1
omega = 0.0001
k = np.array([1., 0.5, 0.5, 0.5]) * omega
r = np.array([1., 1., 1.])
p = [np.array([0., 0., 0.]), np.array([3., 0., 0.]), np.array([0., 0., 3.])]

k = np.array([1., 1.]) * omega
r = np.array([1.])
p = [np.array([0., 0., 0.])]

a = np.ones((4))

L_plus_1_square = (L + 1)**2
j_l, j_lp, h_l, h_lp = hsf.helmholtz_pre_computations(N, k, r, L)

almost_A_0 = hsf.cross_interactions_version_1_fast_general(L_c, L, N, k[0], r, p, L_plus_1_square, j_l[:, :, 0], j_lp[:, :, 0])

matrix, solucion = hsf.direct_solution_MTF(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0)

hola = full_matrix-matrix/2.
print(np.linalg.norm(hola))
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(np.real(hola))
plt.title('MTF')
plt.show()


# Testing version 1
'''
E0 = np.asarray([1., 0.7, 0.4])
E1 = np.asarray([3., 2., 4.5])
sigma_i = 1.
sigma_e = 1.
E_phi = np.zeros((3))#(sigma_i*E1 -sigma_e*E0)/sigma_e
E_v = E1-E0-E_phi

def function_E_phi(x):
    return np.dot(E_phi, x)

def function_E_v(x):
    return np.dot(E_v, x)

def function_E1(x):
    return np.dot(E1, x)

# y=np.asarray([[0],[1],[2]])
# print(function_E1(y))

def function_E0(x):
    return np.dot(E0, x)

L = 0

zeros, w = psh.expand.SHGLQ(L)

[minus_cos_theta, weights] = np.polynomial.legendre.leggauss(L + 1)
cos_theta = - minus_cos_theta
print(np.linalg.norm(cos_theta-zeros)) #equivalentes
print(np.linalg.norm(weights-w)) # equivalentes
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

mapeo_dirichlet_E1 = np.empty((quantity_theta_points, quantity_phi_points))
mapeo_dirichlet_E0 = np.empty((quantity_theta_points, quantity_phi_points))
mapeo_v = np.empty((quantity_theta_points, quantity_phi_points))
mapeo_dirichlet_phi_E0 = np.empty((quantity_theta_points, quantity_phi_points))
for i in np.arange(0,quantity_theta_points):
    for j in np.arange(0,quantity_phi_points):
        mapeo_dirichlet_E1[i,j] = function_E1(pre_vector[:,i,j])
        mapeo_dirichlet_E0[i, j] = function_E0(pre_vector[:, i, j])
        mapeo_v[i,j] = function_E_v(pre_vector[:, i, j])
        mapeo_dirichlet_phi_E0[i,j] = function_E_phi(pre_vector[:, i, j])


temp_dirichletE1 = psh.expand.SHExpandGLQ(mapeo_dirichlet_E1, weights, cos_theta, norm=4, csphase=-1)
temp_dirichletE0 = psh.expand.SHExpandGLQ(mapeo_dirichlet_E0, weights, cos_theta, norm=4, csphase=-1)
temp_dirichlet_v = psh.expand.SHExpandGLQ(mapeo_v, weights, cos_theta, norm=4, csphase=-1)
temp_phi_d = psh.expand.SHExpandGLQ(mapeo_dirichlet_phi_E0, weights, cos_theta, norm=4, csphase=-1)

num = (L+1)**2

dirichletE1=np.empty((num))
dirichletE0=np.empty((num))
v_test = np.empty((num))
phi_d = np.empty((num))
l2_1 = 0
for l in np.arange(0,L+1):
    dirichletE1[(l2_1 + l)] = temp_dirichletE1[0,l,0]
    dirichletE0[(l2_1 + l)] = temp_dirichletE0[0, l, 0]
    v_test[(l2_1 + l)] = 1.#temp_dirichlet_v[0, l, 0]
    phi_d[(l2_1 + l)] = temp_phi_d[0, l, 0]
    for m in np.arange(1, l+1):
        dirichletE1[(l2_1 + l+m)] = temp_dirichletE1[0, l, m]
        dirichletE1[(l2_1 + l - m)] = temp_dirichletE1[1, l, m]
        dirichletE0[(l2_1 + l + m)] = temp_dirichletE0[0, l, m]
        dirichletE0[(l2_1 + l - m)] = temp_dirichletE0[1, l, m]
        v_test[(l2_1 + l + m)] = 1.#temp_dirichlet_v[0, l, m]
        v_test[(l2_1 + l - m)] = 1.#temp_dirichlet_v[1, l, m]
        phi_d[(l2_1 + l + m)] = temp_phi_d[0, l, m]
        phi_d[(l2_1 + l - m)] = temp_phi_d[1, l, m]
    l2_1 = l2_1 + 2*l+1

# del temp_dirichletE1
# del temp_dirichletE0
# del temp_dirichlet_v
# del temp_phi_d

neumannE1 = np.copy(dirichletE1)
neumannE0 = -dirichletE0

print(phi_d)

phi_n = -phi_d

sigma_e_i = sigma_e/sigma_i

b = np.empty((4*num))
b[0:num] = - 0.5 * sigma_e_i * phi_d - sigma_e_i * v_test
b[num:2*num] = -0.5 * sigma_e_i * phi_n
b[2*num:3*num] = 0.5 * phi_d + v_test
b[3*num:4*num] = b[num:2*num]

r=1.
A_0, A_1, X_diag_up, X_diag_down = lsfb1.laplace_MTF_1_sphere_all_basis(sigma_e, sigma_i, r, L)
full_matrix=lsfb1.MTF_1_sphere_full_matrix_FF_all_basis(A_0, A_1, X_diag_up, X_diag_down, num)

solutionL = np.linalg.solve(full_matrix, b)

solutionL = np.empty((4*num))
solutionL[0:num] = dirichletE0
solutionL[num:2*num] = neumannE0
solutionL[2*num:3*num] = dirichletE1
solutionL[3*num:4*num] = neumannE1

A=full_matrix[2:4,2:4]
g=np.zeros((2,1))
g[:,0]=solutionL[0:2]
m2 = np.matmul(A,g)

N=1
k=np.asarray([0.01,0.01])
rk = r*k
j_l_1 = np.empty((L + 1, 2))
j_lp_1 = np.empty((L + 1, 2))
h_l_1 = np.empty((L + 1, 2), dtype=np.complex128)
h_lp_1 = np.empty((L + 1, 2), dtype=np.complex128)

import scipy.special as sci
rango = np.arange(0, L + 1)
for i in np.arange(0, 2):
    aux00 = sci.spherical_jn(rango, rk[i])
    aux01 = sci.spherical_jn(rango, rk[i], derivative=True)
    j_l_1[:, i] = aux00
    j_lp_1[:, i] = aux01
    h_l_1[:, i] = aux00 + np.multiply(1j, sci.spherical_yn(rango, rk[i]))
    h_lp_1[:, i] = aux01 + np.multiply(1j, sci.spherical_yn(rango, rk[i], derivative=True))

V_out = 1j * r**2 * k[0] * j_l_1[:, 0] * h_l_1[:, 0]
V_ins = 1j * r**2 * k[1] * j_l_1[:, 1]*h_l_1[:, 1]
K_out_times_minus_1 = 1j * r**2 * k[0]**2*j_lp_1[:, 0]*h_l_1[:, 0] - 0.5
K_ins_times_minus_1 = -1j * r**2 * k[1]**2*j_l_1[:, 1]*h_lp_1[:, 1] - 0.5
K_ast_out = -1j * r**2 * k[0] ** 2 * j_l_1[:, 0] * h_lp_1[:, 0] - 0.5
K_ast_ins = 1j * r**2 * k[1] ** 2 * j_lp_1[:, 1] * h_l_1[:, 1] - 0.5
W_out = -1j * r**2 * k[0]**3 * j_lp_1[:, 0] * h_lp_1[:, 0]
W_ins = -1j * r**2 * k[1]**3 * j_lp_1[:, 1] * h_lp_1[:, 1]


mmm= np.matmul(full_matrix, solutionL)
mmm2= np.matmul(full_matrix, solutionL)
print(np.linalg.norm(solutionL-solutionL))
'''