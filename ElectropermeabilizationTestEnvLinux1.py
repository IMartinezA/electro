import numpy as np

import pyshtools as psh

import ElectropermeabilizationEnvBasic1 as elec

sigma_e = 1.
sigma_i = 0.5
c_m = 1.
r_m = 2.
r = 1.

E = np.asarray([0., 0., 3.])
phi_x = elec.phi_E_dot_x(E)
omega = 1.

i_ion_v = elec.i_ion_linear_version(r_m)

L = 1
tau = 0.1
step_number = 100

num = (L+1)**2
sigma_e_i = sigma_e / sigma_i

v_0 = np.zeros((num))
g_0 = np.zeros((num))


MTF_FF_matrix, M_01_cuartos, M_11 = elec.electro_matrix_operator_FF(sigma_e, sigma_i, c_m, tau, r, L)

b = elec.b_update_constant_step(i_ion_v, num, sigma_e_i, M_01_cuartos, M_11, c_m, tau)


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
for i in np.arange(0, quantity_theta_points):
    for j in np.arange(0, quantity_phi_points):
        mapeo_dirichlet_phi_E0[i, j] = phi_x(pre_vector[:, i, j])

temp_phi_d = psh.expand.SHExpandGLQ(mapeo_dirichlet_phi_E0, weights, cos_theta, norm=4, csphase=-1)

l2_1 = 0
phi_d_0 = np.zeros((num))
for l in np.arange(0,L+1):
    phi_d_0[(l2_1 + l)] = temp_phi_d[0, l, 0]
    for m in np.arange(1, l+1):
        phi_d_0[(l2_1 + l + m)] = temp_phi_d[0, l, m]
        phi_d_0[(l2_1 + l - m)] = temp_phi_d[1, l, m]
    l2_1 = l2_1 + 2*l+1

times = np.arange(tau, tau*(step_number), tau)
phi_d = np.zeros((num, step_number))
contador = 0
for t in times:
    phi_d[:, contador] = phi_d_0*np.cos(omega*t)
    contador = contador + 1
phi_n = -phi_d
solutions = np.zeros((num*5, step_number))

# Predictor-corrector

# (i)
solutions[:, 0] = np.linalg.solve(MTF_FF_matrix, b(v_0, v_0, phi_d[:, 0], phi_n[:, 0]))

w_1 = solutions[num*4:num*5, 0]
solutions[num*4:num*5, 0] = v_0

# (ii)
solutions[:, 1] = np.linalg.solve(MTF_FF_matrix, b(v_0, (v_0+w_1)/2, phi_d[:, 0], phi_n[:, 0]))

# The rest
for jj in np.arange(2, step_number):
    solutions[:, jj] = np.linalg.solve(MTF_FF_matrix, b(solutions[num*4:num*5, jj-1], (3 * solutions[num*4:num*5, jj-1]-solutions[num*4:num*5, jj-2])/2, phi_d[:, jj-1], phi_n[:, jj-1]))

import matplotlib.pyplot as plt
import GraficosFigurasEnvLinux1 as pic
import matplotlib.animation as animation

r = np.array([1.])
p = [np.array([0., 0., 0.])]

dd = 2
centro = p[0]
ancho = 4.
alto = 4.
interancho = 10*10#*4
interlargo = 10*10#*4

#print(solutionL)
#print(solucion)


potential = pic.drawing_cut_with_real_spherical_harmonics_one_sphere_laplace_kernel(dd, centro, ancho, alto, interancho, interlargo, solutions[0:4*num, 0], r, L, phi_x)

potential_all_times = np.zeros((len(potential[:,0]), len(potential[0,:]), len(solutions[0,:])-1))
del potential

for jj in np.arange(0, len(solutions[0, :])-2):
    potential_all_times[:,:, jj] = pic.drawing_cut_with_real_spherical_harmonics_one_sphere_laplace_kernel(dd, centro, ancho, alto, interancho, interlargo, solutions[0:4*num, jj+1], r, L, phi_x)

maximo = np.max(np.abs(potential_all_times))
extent = [-ancho/2., ancho/2., -ancho/2., ancho/2.]

fig, ax = plt.subplots()
ims = []
for jj in np.arange(0, len(solutions[0, :])-2):
    im = ax.imshow(potential_all_times[:,:, jj], animated=True, extent=extent, interpolation='None', cmap='seismic', vmin=-maximo, vmax=maximo)
    ims.append([im])

fig.colorbar(im)
ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=500)

plt.title('Total field')
#plt.colorbar()
plt.show()

def zerozerozero(x):
    return 0.

for jj in np.arange(0, len(solutions[0, :])-2):
    potential_all_times[:,:, jj] = pic.drawing_cut_with_real_spherical_harmonics_one_sphere_laplace_kernel(dd, centro, ancho, alto, interancho, interlargo, solutions[0:4*num, jj+1], r, L, zerozerozero)

maximo = np.max(np.abs(potential_all_times))
extent = [-ancho/2., ancho/2.,-ancho/2., ancho/2.]

fig, ax = plt.subplots()
ims = []
for jj in np.arange(0, len(solutions[0, :])-2):
    im = ax.imshow(potential_all_times[:,:, jj], animated=True, extent=extent, interpolation='None', cmap='seismic', vmin=-maximo, vmax=maximo)
    ims.append([im])

fig.colorbar(im)
ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=500)

plt.title('Field without the external potential')
#plt.colorbar()
plt.show()


