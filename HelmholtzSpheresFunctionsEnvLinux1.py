# Functions for solving the Helmholtz transmission problem in an unbounded domain with N spheres using
# a Galerkin scheme with boundary integral equations an spherical harmonics as basis.
# There will be two formulations.


# Some libraries used
#---
import numpy as np  # BSD
# Library for arrays, vectors and matrices in dense format, along with routines
# of linear algebra, norm computation, dot product. It also computes
# functions like sine, cosine, exponential, etc.

import scipy.special as sci
# Library for spherical Bessel and Hankel functions.

import scipy.sparse as sp
# Library for sparse matrix formats.

import scipy.sparse.linalg as splg
# Library for routines of linear algebra for sparse matrix formats from the package scipy.sparse

import time as tm
# For measuring time.

# Does not work on Windows
import pyshtools as psh  # BSD-3-clauses
# Library for spherical harmonic functions. It doesn't work on Windows.


def helmholtz_pre_computations(N, k, r, L):
    # Outputs: [jn,jnp,hn,hnp]
    # Spherical Bessel, its derivative, spherical Hankel, its derivative,
    # functions evaluated in a number (values from the package scipy.special):
    # j_l     (L+1) x N x 2, complex array,
    #                       j_l[l,j,0]=jn(r_j[j+1]*k[0])
    #                       j_l[j,j,1]=jn(r_j[j+1]*k[j+1)])
    # j_lp    (L+1) x N x 2, complex array, analogous
    # h_l     (L+1) x N x 2, complex array, analogous
    # h_lp    (L+1) x N x 2, complex array, analogous

    # Inputs:
    # N:    1    int. Number of spheres.
    # k:    N+1 array [k_0,k_1,k_2,...,k_n], k_j = omega / c_j. Wave numbers of each sphere.
    # r_j:  N+1 array floats. Spheres radius.
    # L:    1 int. Maximum order of the spherical functions.

    rk = np.zeros((N, 2), np.complex128)  # array de complex 128
    rk[:, 0] = np.multiply(r, k[0])
    rango = np.arange(0, N)
    rk[rango, 1] = np.multiply(r[rango], k[rango + 1])
    del rango

    j_l = np.zeros((L + 1, N, 2), dtype=np.complex128)
    j_lp = np.zeros((L + 1, N, 2), dtype=np.complex128)
    h_l = np.zeros((L + 1, N, 2), dtype=np.complex128)
    h_lp = np.zeros((L + 1, N, 2), dtype=np.complex128)

    rango = np.arange(0, L + 1)
    aux00 = 0.
    aux01 = 0.
    for i in np.arange(0, 2):
        for j in np.arange(0, N):
            aux00 = sci.spherical_jn(rango, rk[j, i])
            aux01 = sci.spherical_jn(rango, rk[j, i], derivative=True)
            j_l[:, j, i] = aux00
            j_lp[:, j, i] = aux01
            h_l[:, j, i] = aux00 + np.multiply(1j, sci.spherical_yn(rango, rk[j, i]))
            h_lp[:, j, i] = aux01 + np.multiply(1j, sci.spherical_yn(rango, rk[j, i], derivative=True))
    del rk
    del rango
    del aux00
    del aux01

    return j_l, j_lp, h_l, h_lp


# Pre computation for the integral quadratures.
# ---
def pre_computation_integral_quadrature(L_c, L):
    # Outputs: [quantity_quadrature_points, pre_vector, pre_integral]
    # Inputs:
    ##L:    1    int. It's the maximum degree of spherical harmonics used.
    #                   (L+1)**2 is the quantity of spherical harmonics used.
    ##L_c: 1 int. It's the parameter used to compute the points of the
    #               quadrature. Without considering errors produced by the
    #               approximation by finite numbers, the quadrature must be
    #               exact for functions consisting in polynomials of L_c degree
    #               times a exponential power to (m times i), with |m|<=L_c.

    ##Integral on theta are (L_c + 1) quadrature points.
    ##Integral on phi are (2*L_c + 1) quadrature points.

    [minus_cos_theta, weights] = np.polynomial.legendre.leggauss(L_c + 1)
    # (according to the numpy documentation it has been proved until 100 quadrature points)
    cos_theta = - minus_cos_theta

    del minus_cos_theta
    # v = cos(theta),
    # dv = - sin(theta) dtheta --> dtheta = - dv / sin(theta)
    phi = np.linspace(0, 2 * np.pi, num=(2 * L_c + 1), endpoint=False)
    quantity_theta_points = len(cos_theta)
    quantity_phi_points = len(phi)
    weights = weights*2*np.pi/quantity_phi_points

    cos_phi = np.cos(phi)
    sen_phi = np.sin(phi)

    # Legendre functions with Condon-Shortley phase with the normalization of the spherical harmonics
    Legendre_functions = np.zeros(((L + 1) * (L + 2) // 2, quantity_theta_points))  # (L+1)*((L+1)+1)/2
    for i in np.arange(0, quantity_theta_points):
        Legendre_functions[:, i] = psh.legendre.PlmON(L, cos_theta[i], csphase=-1, cnorm=1)*weights[i]

    exp_pos = np.zeros((L, quantity_phi_points), dtype=np.complex128)
    for m in np.arange(1, L + 1):
        exp_pos[m - 1, :] = np.exp(1j * m * phi)
    del phi
    sen_theta = np.sqrt(1. - np.square(cos_theta))

    # First is Tile (theta integral), then is repeat (phi integral).

    quantity_quadrature_points = quantity_theta_points * quantity_phi_points

    pre_integral = np.zeros((np.square(L + 1), quantity_quadrature_points), dtype=np.complex128)
    l_square_plus_l = 0.
    l_times_l_plus_1_divided_by_2 = 0.
    auxiliar = np.zeros((quantity_quadrature_points))
    for l in np.arange(0, L + 1):
        l_square_plus_l = (l ** 2 + l)
        l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2

        pre_integral[l_square_plus_l, :] = np.tile(Legendre_functions[l_times_l_plus_l_divided_by_2, :],
                                                   quantity_phi_points)
        for m in np.arange(1, l + 1):
            auxiliar = np.tile(Legendre_functions[l_times_l_plus_l_divided_by_2 + m, :], quantity_phi_points)
            pre_integral[(l_square_plus_l + m), :] = auxiliar * np.repeat(exp_pos[m - 1, :], quantity_theta_points)
            pre_integral[(l_square_plus_l - m), :] = (-1) ** m * np.conjugate(pre_integral[(l_square_plus_l + m), :])
    pre_integral = np.conjugate(pre_integral)
    del auxiliar
    del Legendre_functions
    del exp_pos
    del weights
    del l_square_plus_l
    del l_times_l_plus_1_divided_by_2

    cos_phi = np.repeat(cos_phi, quantity_theta_points)  # length quantity_quadrature_points
    sen_phi = np.repeat(sen_phi, quantity_theta_points)  # length quantity_quadrature_points
    sen_theta = np.tile(sen_theta, quantity_phi_points)  # length quantity_quadrature_points
    cos_theta = np.tile(cos_theta, quantity_phi_points)  # length quantity_quadrature_points

    del quantity_theta_points
    del quantity_phi_points

    pre_vector = np.zeros((3, quantity_quadrature_points))
    pre_vector[0, :] = sen_theta * cos_phi
    pre_vector[1, :] = sen_theta * sen_phi
    pre_vector[2, :] = cos_theta

    del cos_phi
    del sen_phi
    del sen_theta
    del cos_theta

    return quantity_quadrature_points, pre_vector, pre_integral


def pre_computation_integral_quadrature_fast(L_c):
    # Outputs: [quantity_theta_points, quantity_phi_points, w, pre_vector]

    # Inputs:
    ##L_c:1 int. It's the parameter used to compute the points of the
    #               quadrature. Without considering errors produced by the
    #               approximation by finite numbers, the quadrature must be
    #               exact for functions consisting in polynomials of L_c degree
    #               times a exponential power to (m times i), with |m|<=L_c.

    ##Integral on theta are (L_c + 1) quadrature points.
    ##Integral on phi are (2*L_c + 1) quadrature points.

    cos_theta, w = psh.expand.SHGLQ(L_c)
    phi = np.linspace(0, 2 * np.pi, num=(2 * L_c + 1), endpoint=False)

    quantity_theta_points = len(cos_theta)
    quantity_phi_points = len(phi)

    cos_phi = np.cos(phi)
    sen_phi = np.sin(phi)

    del phi
    sen_theta = np.sqrt(1. - np.square(cos_theta))

    pre_vector = np.zeros((3, quantity_theta_points, quantity_phi_points))
    for i in np.arange(0, quantity_theta_points):
        pre_vector[0, i, :] = np.multiply(sen_theta[i], cos_phi)
        pre_vector[1, i, :] = np.multiply(sen_theta[i], sen_phi)
        pre_vector[2, i, :] = cos_theta[i]

    del sen_theta
    del cos_phi
    del sen_phi
    del cos_theta

    return quantity_theta_points, quantity_phi_points, w, pre_vector


def PesyKus(L):
    PyQ = np.zeros(((L + 1) * L // 2, 2), dtype=int)
    contador = 0
    for l in np.arange(1, L + 1):
        for m in np.arange(1, l + 1):
            PyQ[contador, 0] = l
            PyQ[contador, 1] = m
            contador = contador + 1
    P2_plusP = PyQ[:, 0] * (PyQ[:, 0] + 1)  # l*(l+1)
    P2_plusP_plus_Q = P2_plusP + PyQ[:, 1]
    P2_plusP_minus_Q = P2_plusP - PyQ[:, 1]
    del P2_plusP

    return PyQ, P2_plusP_plus_Q, P2_plusP_minus_Q


def rows_columns_dense(L):
    dimension = (2 * (L + 1) ** 2)
    row_dense = np.zeros((dimension, dimension), dtype=int)
    column_dense = np.zeros((dimension, dimension), dtype=int)
    for i in np.arange(0, dimension):
        row_dense[i, :] = i
        column_dense[:, i] = i
    row_dense = np.concatenate(row_dense)
    column_dense = np.concatenate(column_dense)
    del dimension

    return row_dense, column_dense

# Routines for the numeric integrals.
# ---
def integral_version_1(p_j, r_j, p_s, L, k_0, quantity_quadrature_points, pre_vector, L_plus_1_square, pre_integral, j_l_0_s, j_lp_0_s):
    # Output: dataV, dataK, dataKa, dataW
    # dataV : complex matrix (np.complex128) dataV[a,b] = (V**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    #                                                   with Y_{l,m} the spherical harmonic of degree l and order m,
    #                                                   a = sum_{i=0}**{p-1}{2*p+1} + sum_{i=-p}**{q}{1}, and
    #                                                   b = sum_{i=0}**{l-1}{2*l+1} + sum_{i=-l}**{m}{1}
    # dataK : complex matrix (np.complex128) dataK[a,b] = (K**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    # dataKa : complex matrix (np.complex128) dataKa[a,b] = (K**{*0}_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    # dataW : complex matrix (np.complex128) dataW[a,b] = (W**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},

    # Input:
    # j indica desde que esfera se tom\'o la traza
    # s indica a que traza se le aplic\'o el operador
    # j_l_0_s = jl[0,s-1]
    # j_lp_0_s = jl[0,s-1]
    # p_j es la posici\'on del centro de la esfera j
    # p_s es la posici\'on del centro de la esfera s
    # r_j es el radio de la esfera j
    # k_0 es el n\'umero de onda del medio exterior.
    # r_s_square es r_s**2
    # L es un int. Es el orden m\'as alto que se utiliza de los arm\'onicos esf\'ericos.
    # (L+1)**2 es la cantidad de arm\'onicos esf\'ericos que se utilizan para discretizar cada traza.
    # num = (L+1)**2
    # quantity_quadrature_points es la cantidad de parejas de puntos que se utilizan para calcular una
    # integral de superficie.
    # pre_vector viene de pre_computation_integral_quadrature(L_c)

    # d_sj es el vector resultado de restar:
    d_sj = p_j - p_s
    # Se utiliza para escribir un vector del sistema coordenado j en el sistema s
    temp = np.multiply(r_j, pre_vector)

    # erres es un array de 3 x quantity_quadrature_points con los vectores en el sistema
    # coordenado s. Para ello se les suma d_sj (ver pdf)
    erres = np.zeros((3, quantity_quadrature_points))
    erres[0, :] = temp[0, :] + d_sj[0]
    erres[1, :] = temp[1, :] + d_sj[1]
    erres[2, :] = temp[2, :] + d_sj[2]

    del d_sj

    # norm_erres es un array de largo quantity_quadrature_points con la norma de cada vector columna de erres
    norm_erres = np.linalg.norm(erres, axis=0)

    # Se quiere obtener los valores de los \'angulos theta y phi del sistema coordenado s de los vectores:

    # Para el \'angulo theta
    cos_theta_coords = np.divide(erres[2, :], norm_erres)
    theta_coords = np.arccos(cos_theta_coords)
    # Para el \'angulo phi
    phi_coords = np.arctan2(erres[1, :], erres[0, :])
    del erres

    # Ahora se quiere calcular valores que aparecen en los operadores que
    # tienen derivadas. Los que se guardar\'an en pre_calc_1, pre_calc_2 y
    # pre_calc_3. Estos tres array son de largo quantity_quadrature_points.
    # Se recomienda mirar el pdf para entender la siguiente explicaci\'on,
    # La secci\'on del anexo que trata del gradiente de la funci\'on de Hankel
    # por un arm\'onico esf\'erico
    # pre_calc_1 es el vector unitario de la direcci\'on r punto la normal a la
    # esfera
    # pre_calc_2 es el vector unitario de la direcci\'on theta punto la normal
    # a la esfera multiplicado por las constantes que acompa\~nan a este t\'ermino
    # (ver el pdf)
    # pre_calc_3 es el vector unitario de la direcci\'on phi punto la normal a
    # la esfera multiplicado por las constantes que acompa\~nan a este t\'ermino
    # (ver el pdf)
    # Ahora se obtienen los valores del sen(tetha), cos(theta), cos(phi),
    # sen(phi) de los \'angulos en el sistema coordenado s.
    # Notar que no se est\'an sobreescribiendo las variables, si no que
    # se cre\'o una nueva con una s al final
    # para el sistema de referencia s
    sin_theta_coords = np.sin(theta_coords)

    del theta_coords

    cos_phi_coords = np.cos(phi_coords)
    sin_phi_coords = np.sin(phi_coords)

    # Vectores unitarios de la coordenada r del sistema de referencia s
    # ers est\'a escrito en el sistema de referencia s
    ers = np.zeros((3, quantity_quadrature_points))
    ers[0, :] = np.multiply(sin_theta_coords, cos_phi_coords)
    ers[1, :] = np.multiply(sin_theta_coords, sin_phi_coords)
    ers[2, :] = cos_theta_coords

    # Obtenemos los vectores normales a la esfera al dividir temp por el radio
    # de esta
    temp = np.divide(temp, r_j)

    # Producto punto entre los vectores temp y ers
    pre_calc_1 = np.sum(np.multiply(temp, ers), axis=0)

    del ers

    # Vectores unitarios de la coordenada theta del sistema de referencia s
    # etheta_coords est\'a escrito en el sistema de referencia s
    etheta_coords = np.zeros((3, quantity_quadrature_points))
    etheta_coords[0, :] = np.multiply(cos_theta_coords, cos_phi_coords)
    etheta_coords[1, :] = np.multiply(cos_theta_coords, sin_phi_coords)
    etheta_coords[2, :] = -sin_theta_coords

    del sin_theta_coords

    # Se divide por 2*norm_erres. Esto debido a que en las siguientes
    # operaciones en las que se utiliza temp se ten\'ia que dividir por este
    # valor, as\'i que para reducir las operaciones a la mitad se hizo eso.
    for i in np.arange(0, 3):
        temp[i, :] = np.divide(temp[i, :], np.multiply(2., norm_erres))

    # Producto punto entre los vectores temp y etheta_coords.
    pre_calc_2 = np.sum(np.multiply(temp, etheta_coords), axis=0)
    del etheta_coords

    # Vectores unitarios de la coordenada phi del sistema de referencia s
    # ephi_coords est\'a escrito en el sistema de referencia s
    ephi_coords = np.zeros((3, quantity_quadrature_points))
    ephi_coords[0, :] = np.multiply(-1, sin_phi_coords)
    ephi_coords[1, :] = cos_phi_coords

    del sin_phi_coords
    del cos_phi_coords

    # Producto punto entre los vectores temp y ephi_coords.
    pre_calc_3 = np.sum(np.multiply(temp, ephi_coords), axis=0)
    del ephi_coords
    del temp

    # int_h_l es un array de quantity_quadrature_points x L+1
    # En donde en la coordenada (i,l) se guarda el valor de la funci\'on de
    # Hankel de primer tipo de orden l evaluada en el valor de la coordenada i
    # del array k_0*norm_erres.
    # Primero, creamos el array y luego lo inicializamos.
    # k_0_int_h_lp es un array de quantity_quadrature_points x L+1
    # En donde en la coordenada (i,l) se guarda el valor de k_0 * la derivada de
    # la  funci\'on de Hankel de primer tipo de orden l evaluada en el valor de
    # la coordenada i del array k_0*norm_erres por el valor que se obtiene de
    # pre_calc_1.

    # Calculamos los valores de los polinomios de Legendre para calcular los valores
    # de esf\'ericos arm\'onicos que vienen de la integral anal\'itica anterior
    Legendre_functions = np.zeros(((L + 2) * (L + 3) // 2, quantity_quadrature_points))
    argumento = np.multiply(k_0, norm_erres)
    del norm_erres
    L_plus_1 = L + 1
    rango = np.arange(0, L_plus_1)
    int_h_l = np.zeros((quantity_quadrature_points, L_plus_1), dtype=np.complex128)
    k_0_int_h_lp = np.zeros((quantity_quadrature_points, L_plus_1), dtype=np.complex128)
    for i in np.arange(0, quantity_quadrature_points):
        int_h_l[i, :] = sci.spherical_jn(rango, argumento[i]) + np.multiply(1j, sci.spherical_yn(rango, argumento[i]))
        k_0_int_h_lp[i, :] = sci.spherical_jn(rango, argumento[i], derivative=True) + np.multiply(1j, sci.spherical_yn(
            rango, argumento[i], derivative=True))
        Legendre_functions[:, i] = psh.legendre.PlmON(L_plus_1, cos_theta_coords[i], csphase=-1, cnorm=1)
    del argumento
    del cos_theta_coords

    k_0_int_h_lp = k_0_int_h_lp * k_0

    # Se calculan las exponenciales conjugadas
    exp_pos = np.zeros((L_plus_1, quantity_quadrature_points), dtype=np.complex128)
    for m in np.arange(1, L_plus_1 + 1):
        exp_pos[m - 1, :] = np.exp(1j * m * phi_coords)
    del phi_coords


    some_spherical_harmonics = np.zeros((np.square(L_plus_1 + 1), quantity_quadrature_points), dtype=np.complex128)
    for l in np.arange(0, L_plus_1 + 1):
        l_square_plus_l = l * (l + 1)
        l_square_plus_l_divided_2 = l_square_plus_l // 2
        some_spherical_harmonics[l_square_plus_l, :] = Legendre_functions[l_square_plus_l_divided_2, :]
        for m in np.arange(1, l + 1):
            auxiliar = Legendre_functions[l_square_plus_l_divided_2 + m, :]
            some_spherical_harmonics[(l_square_plus_l + m), :] = auxiliar * exp_pos[m - 1, :]
            some_spherical_harmonics[(l_square_plus_l - m), :] = (-1) ** m * np.conjugate(some_spherical_harmonics[(l_square_plus_l + m), :])

    del Legendre_functions
    exp_pos = exp_pos[0, :]
    exp_neg = np.divide(1., exp_pos)


    data_V = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)
    data_K = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)
    data_Ka = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)
    data_W = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)

    temp_2 = np.zeros((quantity_quadrature_points, L_plus_1), dtype=np.complex128)
    temp_3 = np.zeros((quantity_quadrature_points, L_plus_1), dtype=np.complex128)
    pre_int_Ka_lm = np.zeros((quantity_quadrature_points), dtype=np.complex128)
    for l in rango:
        k_0_int_h_lp[:, l] = np.multiply(k_0_int_h_lp[:, l], pre_calc_1)
        temp_2 = np.multiply(int_h_l[:, l], pre_calc_2)
        temp_3 = np.multiply(int_h_l[:, l], pre_calc_3)
        l_square = np.square(l)
        l_plus_1_square = np.square(l + 1)
        for m in np.arange(-l, l + 1):
            # Arm\'onicos esf\'ericos
            l_square_plus_l_plus_m = l_square + l + m

            # Array vac\'io que guardar\'a la parte del gradiente que tienen
            # algunas integrales.
            pre_int_Ka_lm = np.zeros((quantity_quadrature_points), dtype=np.complex128)

            # Parte exclusiva del gradiente, asignaci\'on de valores dependiendo del caso
            if l > 0:
                if abs(m) < l:
                    pre_int_Ka_lm = pre_int_Ka_lm + np.multiply(temp_2,
                                                                np.sqrt((l - m) * (l + m + 1.)) * np.multiply(exp_neg,
                                                                                                              some_spherical_harmonics[
                                                                                                              (
                                                                                                                          l_square_plus_l_plus_m + 1),
                                                                                                              :]) - np.sqrt(
                                                                    (l + m) * (
                                                                                l - m + 1.)) * exp_pos * some_spherical_harmonics[
                                                                                                         (
                                                                                                                     l_square_plus_l_plus_m - 1),
                                                                                                         :])
                elif m == l:
                    pre_int_Ka_lm = pre_int_Ka_lm - temp_2 * np.sqrt(
                        (l + m) * (l - m + 1.)) * exp_pos * some_spherical_harmonics[(l_square_plus_l_plus_m - 1), :]
                else:
                    pre_int_Ka_lm = pre_int_Ka_lm + temp_2 * np.sqrt(
                        (l - m) * (l + m + 1.)) * exp_neg * some_spherical_harmonics[(l_square_plus_l_plus_m + 1), :]
            pre_int_Ka_lm = pre_int_Ka_lm + np.multiply(k_0_int_h_lp[:, l],
                                                        some_spherical_harmonics[l_square_plus_l_plus_m,
                                                        :]) - 1j * np.sqrt((2 * l + 1.) / (2 * l + 3.)) * temp_3 * (
                                        np.sqrt((l + m + 1.) * (l + m + 2.)) * exp_neg * some_spherical_harmonics[
                                                                                         (l_plus_1_square + l + m + 2),
                                                                                         :] + np.sqrt(
                                    (l - m + 1.) * (l - m + 2.)) * exp_pos * some_spherical_harmonics[
                                                                             (l_plus_1_square + l + m), :])
            # Obtenci\'on de la parte de la integral que no tiene derivada
            # Pero solo la parte que requer\'ia los cambios de variables al sistema s
            # C\'alculo de la integral sin gradiente
            int_V = np.sum(np.multiply(np.multiply(some_spherical_harmonics[l_square_plus_l_plus_m, :], int_h_l[:, l]),
                                       pre_integral), axis=1)

            # C\'alculo de la integral con gradiente
            int_Ka = np.sum(np.multiply(pre_int_Ka_lm, pre_integral), axis=1)

            # print(int_Ka)
            # Asignaci\'on previa de los valores, falta multiplicar por otras cosas
            data_V[:, l_square_plus_l_plus_m] = int_V
            data_K[:, l_square_plus_l_plus_m] = int_V
            data_Ka[:, l_square_plus_l_plus_m] = int_Ka
            data_W[:, l_square_plus_l_plus_m] = int_Ka
        # Aqu\'i se multiplica por los t\'erminos que dependen solo de l
        data_V[:, l_square:l_plus_1_square] = np.multiply(j_l_0_s[l], data_V[:, l_square:l_plus_1_square])
        data_K[:, l_square:l_plus_1_square] = np.multiply(j_lp_0_s[l], data_K[:, l_square:l_plus_1_square])
        data_Ka[:, l_square:l_plus_1_square] = np.multiply(j_l_0_s[l], data_Ka[:, l_square:l_plus_1_square])
        data_W[:, l_square:l_plus_1_square] = np.multiply(j_lp_0_s[l], data_W[:, l_square:l_plus_1_square])
    del some_spherical_harmonics
    del pre_calc_1
    del pre_calc_2
    del pre_calc_3
    del int_h_l
    del k_0_int_h_lp
    del exp_neg
    del exp_pos
    del rango

    common_factor = 1j * k_0

    return common_factor*data_V, -common_factor*k_0*data_K, -common_factor*data_Ka, -common_factor*k_0*data_W


def integral_version_1_fast(p_j, r_j, p_s, L, k_0, quantity_theta_points, quantity_phi_points, w, pre_vector,
                            L_plus_1_square, j_l_0_s, j_lp_0_s, PyQ, P2_plusP_plus_Q, P2_plusP_minus_Q):
    # Output: dataV, dataK, dataKa, dataW
    # dataV : complex matrix (np.complex128) dataV[a,b] = (V**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    #                                                   with Y_{l,m} the spherical harmonic of degree l and order m,
    #                                                   a = sum_{i=0}**{p-1}{2*p+1} + sum_{i=-p}**{q}{1}, and
    #                                                   b = sum_{i=0}**{l-1}{2*l+1} + sum_{i=-l}**{m}{1}
    # dataK : complex matrix (np.complex128) dataK[a,b] = (K**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    # dataKa : complex matrix (np.complex128) dataKa[a,b] = (K**{*0}_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    # dataW : complex matrix (np.complex128) dataW[a,b] = (W**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},

    # Input:
    # j indica desde que esfera se tom\'o la traza
    # s indica a que traza se le aplic\'o el operador
    # j_l_0_s = jn[0,s-1]
    # j_lp_0_s = jn[0,s-1]
    # p_j es la posici\'on del centro de la esfera j
    # p_s es la posici\'on del centro de la esfera s
    # r_j es el radio de la esfera j
    # k_0 es el n\'umero de onda del medio exterior.
    # r_s_square es rss**2
    # L es un int. Es el orden m\'as alto que se utiliza de los arm\'onicos esf\'ericos.
    # (L+1)**2 es la cantidad de arm\'onicos esf\'ericos que se
    # utilizan para discretizar cada traza.
    # num = (L+1)**2
    # quantity_theta_points es la cantidad de puntos en la integral en theta
    # quantity_phi_points es la cantidad de puntos en la integral en phi
    # w son los pesos de la integral en theta
    # PyQ, P2_plusP_plus_Q y P2MasPmenosQpos son variables auxiliares para vectorizar un for

    d_sj = p_j - p_s

    # temp es un array de 3 x quantity_theta_points x quantity_phi_points con los vectores en el sistema
    # coordenado j.
    # La primera fila son los x
    # La segunda fila son los y
    # La tercera fila son los z
    # temp est\'a escrito en el sistema de referencia j
    temp = np.zeros((3, quantity_theta_points, quantity_phi_points))
    temp = np.multiply(r_j, pre_vector)

    # coordenado s. Para ello se les suma d_sj (ver pdf)
    erres = np.zeros((3, quantity_theta_points, quantity_phi_points))
    erres[0, :, :] = temp[0, :, :] + d_sj[0]
    erres[1, :, :] = temp[1, :, :] + d_sj[1]
    erres[2, :, :] = temp[2, :, :] + d_sj[2]

    del d_sj

    norm_erres = np.sqrt(np.sum(np.square(erres), axis=0))

    theta_coords = np.arccos(np.divide(erres[2, :, :], norm_erres))
    phi_coords = np.arctan2(erres[1, :, :], erres[0, :, :])
    del erres

    sin_theta_coords = np.sin(theta_coords)
    cos_theta_coords = np.cos(theta_coords)

    del theta_coords

    cos_phi_coords = np.cos(phi_coords)
    sin_phi_coords = np.sin(phi_coords)

    ers = np.zeros((3, quantity_theta_points, quantity_phi_points))
    ers[0, :, :] = np.multiply(sin_theta_coords, cos_phi_coords)
    ers[1, :, :] = np.multiply(sin_theta_coords, sin_phi_coords)
    ers[2, :, :] = cos_theta_coords

    temp = np.divide(temp, r_j)

    pre_calc_1 = np.sum(np.multiply(temp, ers), axis=0)

    del ers

    etheta_coords = np.zeros((3, quantity_theta_points, quantity_phi_points))
    etheta_coords[0, :, :] = np.multiply(cos_theta_coords, cos_phi_coords)
    etheta_coords[1, :, :] = np.multiply(cos_theta_coords, sin_phi_coords)
    etheta_coords[2, :, :] = np.multiply(-1, sin_theta_coords)

    del sin_theta_coords

    for i in np.arange(0, 3):
        temp[i, :, :] = np.divide(temp[i, :, :], np.multiply(2., norm_erres))

    pre_calc_2 = np.sum(np.multiply(temp, etheta_coords), axis=0)
    del etheta_coords

    ephi_coords = np.zeros((3, quantity_theta_points, quantity_phi_points))
    ephi_coords[0, :, :] = np.multiply(-1, sin_phi_coords)
    ephi_coords[1, :, :] = cos_phi_coords

    del sin_phi_coords
    del cos_phi_coords

    pre_calc_3 = np.sum(np.multiply(temp, ephi_coords), axis=0)
    del ephi_coords
    del temp

    Legendre_functions = np.zeros((quantity_theta_points, quantity_phi_points, (L + 2) * (L + 3) // 2))
    argumentos = np.multiply(k_0, norm_erres)
    del norm_erres
    auxiliar = L + 1
    int_h_l = np.zeros((quantity_theta_points, quantity_phi_points, auxiliar), dtype=np.complex128)
    k_0_int_h_lp = np.zeros((quantity_theta_points, quantity_phi_points, auxiliar), dtype=np.complex128)
    rango = np.arange(0, auxiliar)
    arreglo = np.arange(0, quantity_theta_points)
    for jj in np.arange(0, quantity_phi_points):
        for i in arreglo:
            argumento = argumentos[i, jj]
            int_h_l[i, jj, :] = sci.spherical_jn(rango, argumento) + np.multiply(1j, sci.spherical_yn(rango, argumento))
            k_0_int_h_lp[i, jj, :] = sci.spherical_jn(rango, argumento, derivative=True) + np.multiply(1j,
                                                                                                       sci.spherical_yn(
                                                                                                           rango,
                                                                                                           argumento,
                                                                                                           derivative=True))
            Legendre_functions[i, jj, :] = psh.legendre.PlmON(auxiliar, cos_theta_coords[i, jj], csphase=-1, cnorm=1)
    del argumentos
    del cos_theta_coords
    del arreglo
    k_0_int_h_lp = np.multiply(k_0, k_0_int_h_lp)

    exp_pos = np.zeros((auxiliar, quantity_theta_points, quantity_phi_points), dtype=np.complex128)
    for m in np.arange(1, L + 2):
        exp_pos[m - 1, :, :] = np.exp(1j * m * phi_coords)
    del phi_coords

    some_spherical_harmonics = np.zeros((np.square(L + 2), quantity_theta_points, quantity_phi_points),
                                        dtype=np.complex128)
    for l in np.arange(0, L + 2):
        l_square_plus_l = l * (l + 1)
        l_square_plus_l_divided_2 = l_square_plus_l // 2
        some_spherical_harmonics[l_square_plus_l, :, :] = Legendre_functions[:, :, l_square_plus_l_divided_2]
        for m in np.arange(1, l + 1):
            auxiliar = Legendre_functions[:, :, l_square_plus_l_divided_2 + m]
            some_spherical_harmonics[(l_square_plus_l + m), :, :] = np.multiply(auxiliar, exp_pos[m - 1, :, :])
            some_spherical_harmonics[(l_square_plus_l - m), :, :] = (-1) ** m * np.conjugate(some_spherical_harmonics[
                                                                                (l_square_plus_l + m), :, :])

    del Legendre_functions
    exp_pos = exp_pos[0, :, :]
    exp_neg = np.divide(1., exp_pos)

    data_V = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)

    data_K = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)

    data_Ka = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)

    data_W = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)

    p_square_plus_p = rango * (rango + 1)
    pre_int_Ka_lm = np.zeros((quantity_theta_points, quantity_phi_points), dtype=np.complex128)
    for l in rango:
        k_0_int_h_lp[:, :, l] = np.multiply(k_0_int_h_lp[:, :, l], pre_calc_1)
        temp_2 = np.multiply(int_h_l[:, :, l], pre_calc_2)
        temp_3 = np.multiply(int_h_l[:, :, l], pre_calc_3)
        l_square = np.square(l)
        l_plus_1_square = np.square(l + 1)
        for m in np.arange(-l, l + 1):
            l_square_plus_l_plus_m = l_square + l + m

            # Array vac\'io que guardar\'a la parte del gradiente que tienen
            # algunas integrales.
            pre_int_Ka_lm = np.zeros((quantity_theta_points, quantity_phi_points), dtype=np.complex128)

            # Parte exclusiva del gradiente, asignaci\'on de valores dependiendo del caso
            if l > 0:
                if abs(m) < l:
                    pre_int_Ka_lm = pre_int_Ka_lm + np.multiply(temp_2,
                                                                np.sqrt((l - m) * (l + m + 1.)) * np.multiply(exp_neg,
                                                                                                              some_spherical_harmonics[
                                                                                                              (
                                                                                                                      l_square_plus_l_plus_m + 1),
                                                                                                              :,
                                                                                                              :]) - np.sqrt(
                                                                    (l + m) * (l - m + 1.)) * np.multiply(exp_pos,
                                                                                                          some_spherical_harmonics[
                                                                                                          (
                                                                                                                  l_square_plus_l_plus_m - 1),
                                                                                                          :, :]))
                elif m == l:
                    pre_int_Ka_lm = pre_int_Ka_lm - temp_2 * np.sqrt((l + m) * (l - m + 1.)) * np.multiply(exp_pos,
                                                                                                           some_spherical_harmonics[
                                                                                                           (
                                                                                                                   l_square_plus_l_plus_m - 1),
                                                                                                           :, :])
                else:
                    pre_int_Ka_lm = pre_int_Ka_lm + temp_2 * np.sqrt((l - m) * (l + m + 1.)) * np.multiply(exp_neg,
                                                                                                           some_spherical_harmonics[
                                                                                                           (
                                                                                                                   l_square_plus_l_plus_m + 1),
                                                                                                           :, :])
            pre_int_Ka_lm = pre_int_Ka_lm + np.multiply(k_0_int_h_lp[:, :, l],
                                                        some_spherical_harmonics[l_square_plus_l_plus_m, :,
                                                        :]) - 1j * np.sqrt(
                (2 * l + 1.) / (2 * l + 3.)) * np.multiply(temp_3, (
                    np.sqrt((l + m + 1.) * (l + m + 2.)) * np.multiply(exp_neg,
                                                                       some_spherical_harmonics[
                                                                       (l_plus_1_square + l + m + 2),
                                                                       :, :]) + np.sqrt(
                (l - m + 1.) * (l - m + 2.)) * np.multiply(exp_pos,
                                                           some_spherical_harmonics[(l_plus_1_square + l + m), :, :])))
            # Obtenci\'on de la parte de la integral que no tiene derivada
            # Pero solo la parte que requer\'ia los cambios de variables al sistema s
            int_V = psh.expand.SHExpandGLQC(
                np.multiply(some_spherical_harmonics[l_square_plus_l_plus_m, :, :], int_h_l[:, :, l]), w,
                pre_vector[2, :, :], norm=4, csphase=-1, lmax_calc=L)

            # C\'alculo de la integral con gradiente

            int_Ka = psh.expand.SHExpandGLQC(pre_int_Ka_lm, w, pre_vector[2, :, :], norm=4, csphase=-1, lmax_calc=L)

            # print(int_Ka)
            # Asignaci\'on previa de los valores, falta multiplicar por otras
            # cosas
            data_V[P2_plusP_plus_Q, l_square_plus_l_plus_m] = int_V[0, PyQ[:, 0], PyQ[:, 1]]
            data_V[P2_plusP_minus_Q, l_square_plus_l_plus_m] = int_V[1, PyQ[:, 0], PyQ[:, 1]]
            data_Ka[P2_plusP_plus_Q, l_square_plus_l_plus_m] = int_Ka[0, PyQ[:, 0], PyQ[:, 1]]
            data_Ka[P2_plusP_minus_Q, l_square_plus_l_plus_m] = int_Ka[1, PyQ[:, 0], PyQ[:, 1]]
            data_V[p_square_plus_p, l_square_plus_l_plus_m] = int_V[0, rango, 0]
            data_Ka[p_square_plus_p, l_square_plus_l_plus_m] = int_Ka[0, rango, 0]
            data_K[:, l_square_plus_l_plus_m] = data_V[:, l_square_plus_l_plus_m]
            data_W[:, l_square_plus_l_plus_m] = data_Ka[:, l_square_plus_l_plus_m]
        # Aqu\'i se multiplica por los t\'erminos que dependen solo de l
        data_V[:, l_square:l_plus_1_square] = np.multiply(j_l_0_s[l], data_V[:, l_square:l_plus_1_square])
        data_K[:, l_square:l_plus_1_square] = np.multiply(j_lp_0_s[l], data_K[:, l_square:l_plus_1_square])
        data_Ka[:, l_square:l_plus_1_square] = np.multiply(j_l_0_s[l], data_Ka[:, l_square:l_plus_1_square])
        data_W[:, l_square:l_plus_1_square] = np.multiply(j_lp_0_s[l], data_W[:, l_square:l_plus_1_square])
    del some_spherical_harmonics
    del pre_calc_1
    del pre_calc_2
    del pre_calc_3
    del int_h_l
    del k_0_int_h_lp
    del exp_neg
    del exp_pos
    common_factor = 1j * k_0

    return common_factor * data_V, -common_factor * k_0 * data_K, -common_factor * data_Ka, -common_factor * k_0 * data_W


def integral_version_2(p_j, r_j, p_s, L, k_0, quantity_quadrature_points, pre_vector, L_plus_1_square, pre_integral, j_l_0_s, j_lp_0_s):
    # Output: dataV, dataK, dataKa, dataW
    # dataV : complex matrix (np.complex128) dataV[a,b] = (V**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    #                                                   with Y_{l,m} the spherical harmonic of degree l and order m,
    #                                                   a = sum_{i=0}**{p-1}{2*p+1} + sum_{i=-p}**{q}{1}, and
    #                                                   b = sum_{i=0}**{l-1}{2*l+1} + sum_{i=-l}**{m}{1}
    # dataK : complex matrix (np.complex128) dataK[a,b] = (K**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    # dataKa : complex matrix (np.complex128) dataKa[a,b] = (K**{*0}_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    # dataW : complex matrix (np.complex128) dataW[a,b] = (W**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},

    d_sj = p_j - p_s
    temp = np.multiply(r_j, pre_vector)

    erres = np.zeros((3, quantity_quadrature_points))
    erres[0, :] = temp[0, :] + d_sj[0]
    erres[1, :] = temp[1, :] + d_sj[1]
    erres[2, :] = temp[2, :] + d_sj[2]

    norm_erres = np.linalg.norm(erres, axis=0)

    argumento = np.multiply(k_0, norm_erres)

    L_plus_1 = L + 1
    rango = np.arange(0, L_plus_1)

    int_h_l = np.zeros((quantity_quadrature_points, L_plus_1), dtype=np.complex128)
    int_h_lp = np.zeros((quantity_quadrature_points, L_plus_1), dtype=np.complex128)

    Legendre_functions = np.zeros(((L + 1) * (L + 2) // 2, quantity_quadrature_points))  # (L+1)*((L+1)+1)/2
    d_Legendre_functions = np.zeros(((L + 1) * (L + 2) // 2, quantity_quadrature_points))
    expression_1 = np.zeros((quantity_quadrature_points))

    cos_theta_coords = np.divide(erres[2, :], norm_erres)

    for pp in np.arange(0, quantity_quadrature_points):
        int_h_l[pp, :] = sci.spherical_jn(rango, argumento[pp]) + np.multiply(1j,
                                                                              sci.spherical_yn(rango, argumento[pp]))
        int_h_lp[pp, :] = sci.spherical_jn(rango, argumento[pp], derivative=True) + np.multiply(1j, sci.spherical_yn(
            rango, argumento[pp], derivative=True))
        expression_1[pp] = np.dot(d_sj, pre_vector[:, pp])
        Legendre_functions[:, pp], d_Legendre_functions[:, pp] = psh.legendre.PlmON_d1(L, cos_theta_coords[pp], csphase=-1,
                                                                                       cnorm=1)
        d_Legendre_functions[:, pp] = d_Legendre_functions[:, pp] * -np.sqrt(1 - cos_theta_coords[pp])

    del cos_theta_coords
    del d_sj

    expression_1 = (expression_1 + r_j) / norm_erres

    mask_2 = (np.abs(erres[2, :]) != norm_erres)
    expression_2 = np.zeros((quantity_quadrature_points))

    expression_2[mask_2] = (pre_vector[2, mask_2] - expression_1[mask_2] * erres[2, mask_2] / norm_erres[mask_2]) / np.sqrt(
        (norm_erres[mask_2] - erres[2, mask_2]) * (norm_erres[mask_2] + erres[2, mask_2]))

    expression_1 = - k_0 * expression_1

    expression_3 = np.zeros((quantity_quadrature_points))
    mask_3 = (erres[0, :] != np.zeros((quantity_quadrature_points))) * mask_2
    expression_3[mask_3] = ((erres[1, mask_3] / erres[0, mask_3]) * pre_vector[0, mask_3] - pre_vector[1, mask_3]) * erres[
        0, mask_3] / (erres[0, mask_3] ** 2 + erres[1, mask_3] ** 2)

    del norm_erres

    phi_coords = np.arctan2(erres[1, :], erres[0, :])
    del erres

    exp_pos = np.zeros((L, quantity_quadrature_points), dtype=np.complex128)
    for m in np.arange(1, L_plus_1):
        exp_pos[m - 1, :] = np.exp(1j * m * phi_coords)
    del phi_coords

    data_V = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)
    data_K = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)
    data_Ka = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)
    data_W = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)
    
    some_spherical_harmonics = np.zeros((quantity_quadrature_points), dtype=np.complex128)
    pre_int_Ka_lm = np.zeros((quantity_quadrature_points), dtype=np.complex128)
    temp_1 = np.zeros((quantity_quadrature_points), dtype=np.complex128)
    temp_2 = np.zeros((quantity_quadrature_points), dtype=np.complex128)
    temp_3 = np.zeros((quantity_quadrature_points), dtype=np.complex128)
    l_square = 0.
    l_plus_1_square = 0.
    for l in rango:
        l_square = np.square(l)
        l_plus_1_square = np.square(l + 1)

        l_square_plus_l = l_square + l
        l_square_plus_l_divided_2 = l_square_plus_l // 2
        some_spherical_harmonics = Legendre_functions[l_square_plus_l_divided_2, :]

        # C\'alculo de la integral sin gradiente
        int_V = np.sum(np.multiply(np.multiply(some_spherical_harmonics, int_h_l[:, l]), pre_integral), axis=1)
        
        # C\'alculo de la integral con gradiente
        temp_1 = int_h_lp[:, l] * expression_1
        temp_2 = int_h_l[mask_2, l] * expression_2[mask_2]
        temp_3 = int_h_l[mask_3, l] * expression_3[mask_3]
        pre_int_Ka_lm = temp_1 * some_spherical_harmonics
        pre_int_Ka_lm[mask_2] = pre_int_Ka_lm[mask_2] +  d_Legendre_functions[l_square_plus_l_divided_2, mask_2] * temp_2[mask_2]
        int_Ka = np.sum(np.multiply(pre_int_Ka_lm, pre_integral), axis=1)

        data_V[:, l_square_plus_l] = int_V
        data_K[:, l_square_plus_l] = int_V
        data_Ka[:, l_square_plus_l] = int_Ka
        data_W[:, l_square_plus_l] = int_Ka

        for m in np.arange(1, l + 1):
            auxiliar = Legendre_functions[l_square_plus_l_divided_2 + m, :]
            l_square_plus_l_plus_m = (l_square_plus_l + m)

            some_spherical_harmonics = auxiliar * exp_pos[m - 1, :]

            # C\'alculo de la integral sin gradiente
            int_V = np.sum(np.multiply(np.multiply(some_spherical_harmonics, int_h_l[:, l]), pre_integral), axis=1)

            # C\'alculo de la integral con gradiente
            pre_int_Ka_lm = temp_1 * some_spherical_harmonics
            pre_int_Ka_lm[mask_2] = pre_int_Ka_lm[mask_2] + temp_2[mask_2] * exp_pos[m - 1, mask_2] * d_Legendre_functions[l_square_plus_l_divided_2 + m, mask_2]
            pre_int_Ka_lm[mask_3] = pre_int_Ka_lm[mask_3] + temp_3[mask_3] * 1j * m * some_spherical_harmonics[mask_3]
            int_Ka = np.sum(np.multiply(pre_int_Ka_lm, pre_integral), axis=1)

            # Asignaci\'on previa de los valores, falta multiplicar por otras cosas
            data_V[:, l_square_plus_l_plus_m] = int_V
            data_K[:, l_square_plus_l_plus_m] = int_V
            data_Ka[:, l_square_plus_l_plus_m] = int_Ka
            data_W[:, l_square_plus_l_plus_m] = int_Ka

            l_square_plus_l_minus_m = (l_square_plus_l - m)
            some_spherical_harmonics = (-1) ** m * np.conjugate(some_spherical_harmonics)

            # C\'alculo de la integral sin gradiente
            int_V = np.sum(np.multiply(np.multiply(some_spherical_harmonics, int_h_l[:, l]), pre_integral), axis=1)

            # C\'alculo de la integral con gradiente
            pre_int_Ka_lm = temp_1 * some_spherical_harmonics
            pre_int_Ka_lm[mask_2] = pre_int_Ka_lm[mask_2] + (-1) ** m * temp_2[mask_2] * np.conjugate(exp_pos[m - 1, mask_2]) * d_Legendre_functions[l_square_plus_l_divided_2 + m, mask_2]
            pre_int_Ka_lm[mask_3] = pre_int_Ka_lm[mask_3] + temp_3[mask_3] * 1j * -m * some_spherical_harmonics[mask_3]
            int_Ka = np.sum(np.multiply(pre_int_Ka_lm, pre_integral), axis=1)

            data_V[:, l_square_plus_l_minus_m] = int_V
            data_K[:, l_square_plus_l_minus_m] = int_V
            data_Ka[:, l_square_plus_l_minus_m] = int_Ka
            data_W[:, l_square_plus_l_minus_m] = int_Ka
            # Aqu\'i se multiplica por los t\'erminos que dependen solo de l
        data_V[:, l_square:l_plus_1_square] = np.multiply(j_l_0_s[l], data_V[:, l_square:l_plus_1_square])
        data_K[:, l_square:l_plus_1_square] = np.multiply(j_lp_0_s[l], data_K[:, l_square:l_plus_1_square])
        data_Ka[:, l_square:l_plus_1_square] = np.multiply(j_l_0_s[l], data_Ka[:, l_square:l_plus_1_square])
        data_W[:, l_square:l_plus_1_square] = np.multiply(j_lp_0_s[l], data_W[:, l_square:l_plus_1_square])
    del some_spherical_harmonics
    del temp_1
    del temp_2
    del temp_3
    del mask_2
    del mask_3
    del int_h_l
    del int_h_lp
    del exp_pos
    del rango
    common_factor = 1j * k_0

    return common_factor*data_V, -common_factor*k_0*data_K, -common_factor*data_Ka, -common_factor*k_0*data_W


def integral_version_3_final_part(data_K, k_0, L, L_plus_1_square, j_l_0_s, j_lp_0_s):
    # Output: dataKa, dataW
    # dataKa : complex matrix (np.complex128) dataKa[a,b] = (K**{*0}_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    #                                                   with Y_{l,m} the spherical harmonic of degree l and order m,
    #                                                   a = sum_{i=0}**{p-1}{2*p+1} + sum_{i=-p}**{q}{1}, and
    #                                                   b = sum_{i=0}**{l-1}{2*l+1} + sum_{i=-l}**{m}{1}
    # dataW : complex matrix (np.complex128) dataW[a,b] = (W**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},

    data_Ka = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)
    data_W = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)
    l = 0
    l_squareA = l ** 2
    l2_plus_l = l + l_squareA
    for p in np.arange(0, L + 1):
        p2masp = p * (p + 1)
        q = np.arange(-p, p + 1)
        data_Ka[0, (p2masp + q)] = -(-1) ** (p + q) * data_K[(p2masp - q), 0]
        ##Segun las formulas que saque a mano no deberia ir ni el p ni la l... pero asi no mas me ha funcionado... creo
    for l in np.arange(1, L + 1):
        l_square = l ** 2
        l2_plus_l = l + l_square
        for p in np.arange(0, L + 1):
            p2masp = p * (p + 1)
            q = np.arange(-p, p + 1)
            for m in np.arange(-l, l + 1):
                data_Ka[(l2_plus_l + m), (p2masp + q)] = -(-1) ** (p + q + l + m) * data_K[(p2masp - q), (l2_plus_l - m)]
        data_W[:, l_squareA:l_square] = np.multiply(np.divide(j_lp_0_s[l - 1], j_l_0_s[l - 1]),
                                                    data_W[:, l_squareA:l_square])
        l_squareA = l_square
    data_W = data_W * k_0
    return data_Ka, data_W


def integral_version_3(p_j, r_j, p_s, L, k_0, quantity_quadrature_points, pre_vector, L_plus_1_square, pre_integral, j_l_0_s, j_lp_0_s):
    # Output: dataV, dataK, dataKa, dataW
    # dataV : complex matrix (np.complex128) dataV[a,b] = (V**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    #                                                   with Y_{l,m} the spherical harmonic of degree l and order m,
    #                                                   a = sum_{i=0}**{p-1}{2*p+1} + sum_{i=-p}**{q}{1}, and
    #                                                   b = sum_{i=0}**{l-1}{2*l+1} + sum_{i=-l}**{m}{1}
    # dataK : complex matrix (np.complex128) dataK[a,b] = (K**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    # dataKa : complex matrix (np.complex128) dataKa[a,b] = (K**{*0}_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    # dataW : complex matrix (np.complex128) dataW[a,b] = (W**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},

    d_sj = p_j - p_s

    temp = np.multiply(r_j, pre_vector)

    erres = np.zeros((3, quantity_quadrature_points))
    erres[0, :] = temp[0, :] + d_sj[0]
    erres[1, :] = temp[1, :] + d_sj[1]
    erres[2, :] = temp[2, :] + d_sj[2]

    del temp

    del d_sj

    norm_erres = np.linalg.norm(erres, axis=0)

    phi_coords = np.arctan2(erres[1, :], erres[0, :])

    cos_theta_coords = np.divide(erres[2, :], norm_erres)
    del erres

    Legendre_functions = np.zeros(((L + 1) * (L + 2) // 2, quantity_quadrature_points))
    argumento = np.multiply(k_0, norm_erres)
    del norm_erres
    L_plus_1 = L + 1
    rango = np.arange(0, L_plus_1)
    int_h_l = np.zeros((quantity_quadrature_points, L_plus_1), dtype=np.complex128)
    for i in np.arange(0, quantity_quadrature_points):
        int_h_l[i, :] = sci.spherical_jn(rango, argumento[i]) + np.multiply(1j, sci.spherical_yn(rango, argumento[i]))
        Legendre_functions[:, i] = psh.legendre.PlmON(L, cos_theta_coords[i], csphase=1, cnorm=1)
    del argumento
    del cos_theta_coords

    exp_pos = np.zeros((L_plus_1, quantity_quadrature_points), dtype=np.complex128)
    for m in np.arange(1, L_plus_1):
        exp_pos[m - 1, :] = np.exp(1j * m * phi_coords)
    del phi_coords

    data_V = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)

    data_K = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)

    for l in rango:
        l_square = np.square(l)
        l_plus_1_square = np.square(l + 1)

        l_square_plus_l = l_square + l
        l_square_plus_l_divided_2 = l_square_plus_l // 2
        some_spherical_harmonics = Legendre_functions[l_square_plus_l_divided_2, :]

        # C\'alculo de la integral sin gradiente
        int_V = np.sum(np.multiply(np.multiply(some_spherical_harmonics, int_h_l[:, l]), pre_integral), axis=1)
        data_V[:, l_square_plus_l] = int_V
        data_K[:, l_square_plus_l] = int_V

        for m in np.arange(1, l + 1):
            auxiliar = Legendre_functions[l_square_plus_l_divided_2 + m, :]
            l_square_plus_l_plus_m = (l_square_plus_l + m)

            some_spherical_harmonics = auxiliar * exp_pos[m - 1, :]

            # C\'alculo de la integral sin gradiente
            int_V = np.sum(np.multiply(np.multiply(some_spherical_harmonics, int_h_l[:, l]), pre_integral), axis=1)
            # Asignaci\'on previa de los valores, falta multiplicar por otras cosas
            data_V[:, l_square_plus_l_plus_m] = int_V
            data_K[:, l_square_plus_l_plus_m] = int_V

            l_square_plus_l_minus_m = (l_square_plus_l - m)
            some_spherical_harmonics = (-1) ** m * np.conjugate(some_spherical_harmonics)

            # C\'alculo de la integral sin gradiente
            int_V = np.sum(np.multiply(np.multiply(some_spherical_harmonics, int_h_l[:, l]), pre_integral), axis=1)
            data_V[:, l_square_plus_l_minus_m] = int_V
            data_K[:, l_square_plus_l_minus_m] = int_V
        # Aqu\'i se multiplica por los t\'erminos que dependen solo de l
        data_V[:, l_square:l_plus_1_square] = np.multiply(j_l_0_s[l], data_V[:, l_square:l_plus_1_square])
        data_K[:, l_square:l_plus_1_square] = np.multiply(j_lp_0_s[l], data_K[:, l_square:l_plus_1_square])
    del int_h_l
    del Legendre_functions

    common_factor = 1j * k_0
    data_V = common_factor * data_V
    data_K = -common_factor * k_0 * data_K

    data_Ka, data_W = integral_version_3_final_part(data_K, k_0, L, L_plus_1_square, j_l_0_s, j_lp_0_s)

    return data_V, data_K, data_Ka, data_W


def integral_version_3_fast(p_j, r_j, p_s, L, k_0, quantity_theta_points, quantity_phi_points, w, pre_vector, L_plus_1_square, j_l_0_s, j_lp_0_s, PyQ, P2_plusP_plus_Q, P2_plusP_minus_Q):
    # Output: dataV, dataK, dataKa, dataW
    # dataV : complex matrix (np.complex128) dataV[a,b] = (V**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    #                                                   with Y_{l,m} the spherical harmonic of degree l and order m,
    #                                                   a = sum_{i=0}**{p-1}{2*p+1} + sum_{i=-p}**{q}{1}, and
    #                                                   b = sum_{i=0}**{l-1}{2*l+1} + sum_{i=-l}**{m}{1}
    # dataK : complex matrix (np.complex128) dataK[a,b] = (K**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    # dataKa : complex matrix (np.complex128) dataKa[a,b] = (K**{*0}_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},
    # dataW : complex matrix (np.complex128) dataW[a,b] = (W**0_{j,s} Y_{l,m} ; Y_{p,q})_{L**2(Gamma_j)},

    d_sj = p_j - p_s

    temp = np.multiply(r_j, pre_vector)

    erres = np.zeros((3, quantity_theta_points, quantity_phi_points))
    erres[0, :, :] = temp[0, :, :] + d_sj[0]
    erres[1, :, :] = temp[1, :, :] + d_sj[1]
    erres[2, :, :] = temp[2, :, :] + d_sj[2]

    del temp
    del d_sj

    norm_erres = np.sqrt(np.sum(np.square(erres), axis=0))

    phi_coords = np.arctan2(erres[1, :, :], erres[0, :, :])

    cos_theta_coords = np.divide(erres[2, :, :], norm_erres)

    del erres

    Legendre_functions = np.zeros((quantity_theta_points, quantity_phi_points, (L + 1) * (L + 2) // 2))
    argumentos = np.multiply(k_0, norm_erres)
    del norm_erres
    L_plus_1 = L + 1
    int_h_l = np.zeros((quantity_theta_points, quantity_phi_points, L_plus_1), dtype=np.complex128)
    rango = np.arange(0, L_plus_1)
    arreglo = np.arange(0, quantity_theta_points)
    for jj in np.arange(0, quantity_phi_points):
        for i in arreglo:
            argumento = argumentos[i, jj]
            int_h_l[i, jj, :] = sci.spherical_jn(rango, argumento) + np.multiply(1j, sci.spherical_yn(rango, argumento))
            Legendre_functions[i, jj, :] = psh.legendre.PlmON(L, cos_theta_coords[i, jj], csphase=1, cnorm=1)
    del argumentos
    del cos_theta_coords

    exp_pos = np.zeros((L, quantity_theta_points, quantity_phi_points), dtype=np.complex128)
    for m in np.arange(1, L_plus_1):
        exp_pos[m - 1, :, :] = np.exp(1j * m * phi_coords)
    del phi_coords

    data_V = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)

    data_K = np.zeros((L_plus_1_square, L_plus_1_square), dtype=np.complex128)

    p_square_plus_p = rango * (rango + 1)
    some_spherical_harmonics = np.zeros((quantity_theta_points, quantity_phi_points), dtype=np.complex128)
    l_square = 0.
    l_plus_1_square = 0.
    for l in rango:
        l_square = np.square(l)
        l_plus_1_square = np.square(l + 1)

        l_square_plus_l = l_square + l
        l_square_plus_l_divided_2 = l_square_plus_l // 2
        some_spherical_harmonics = Legendre_functions[:, :, l_square_plus_l_divided_2]

        # C\'alculo de la integral sin gradiente
        int_V = psh.expand.SHExpandGLQC(np.multiply(some_spherical_harmonics, int_h_l[:, :, l]), w, pre_vector[2, :, :],
                                        norm=4, csphase=-1)
        data_V[P2_plusP_plus_Q, l_square_plus_l] = int_V[0, PyQ[:, 0], PyQ[:, 1]]
        data_V[P2_plusP_minus_Q, l_square_plus_l] = int_V[1, PyQ[:, 0], PyQ[:, 1]]
        data_V[p_square_plus_p, l_square_plus_l] = int_V[0, rango, 0]
        data_K[:, l_square_plus_l] = data_V[:, l_square_plus_l]

        for m in np.arange(1, l + 1):
            auxiliar = Legendre_functions[:, :, (l_square_plus_l_divided_2 + m)]
            l_square_plus_l_plus_m = (l_square_plus_l + m)

            some_spherical_harmonics = auxiliar * exp_pos[m - 1, :, :]

            # C\'alculo de la integral sin gradiente
            int_V = psh.expand.SHExpandGLQC(np.multiply(some_spherical_harmonics, int_h_l[:, :, l]), w,
                                            pre_vector[2, :, :], norm=4, csphase=-1)

            data_V[P2_plusP_plus_Q, l_square_plus_l_plus_m] = int_V[0, PyQ[:, 0], PyQ[:, 1]]
            data_V[P2_plusP_minus_Q, l_square_plus_l_plus_m] = int_V[1, PyQ[:, 0], PyQ[:, 1]]
            data_V[p_square_plus_p, l_square_plus_l_plus_m] = int_V[0, rango, 0]
            data_K[:, l_square_plus_l_plus_m] = data_V[:, l_square_plus_l_plus_m]

            l_square_plus_l_minus_m = (l_square_plus_l - m)
            some_spherical_harmonics = (-1) ** m * np.conjugate(some_spherical_harmonics)

            # C\'alculo de la integral sin gradiente
            int_V = psh.expand.SHExpandGLQC(np.multiply(some_spherical_harmonics, int_h_l[:, :, l]), w,
                                            pre_vector[2, :, :], norm=4, csphase=-1)

            data_V[P2_plusP_plus_Q, l_square_plus_l_minus_m] = int_V[0, PyQ[:, 0], PyQ[:, 1]]
            data_V[P2_plusP_minus_Q, l_square_plus_l_minus_m] = int_V[1, PyQ[:, 0], PyQ[:, 1]]
            data_V[p_square_plus_p, l_square_plus_l_minus_m] = int_V[0, rango, 0]
            data_K[:, l_square_plus_l_minus_m] = data_V[:, l_square_plus_l_minus_m]
        # Aqu\'i se multiplica por los t\'erminos que dependen solo de l
        data_V[:, l_square:l_plus_1_square] = np.multiply(j_l_0_s[l], data_V[:, l_square:l_plus_1_square])
        data_K[:, l_square:l_plus_1_square] = np.multiply(j_lp_0_s[l], data_K[:, l_square:l_plus_1_square])
    del some_spherical_harmonics
    del int_h_l

    common_factor = 1j * k_0
    data_V = common_factor * data_V
    data_K = -common_factor * k_0 * data_K

    data_Ka, data_W = integral_version_3_final_part(data_K, k_0, L, L_plus_1_square, j_l_0_s, j_lp_0_s)

    return data_V, data_K, data_Ka, data_W


# Computing of the cross interactions
# ---
def cross_interactions_version_1_general(L_c, L, N, k_0, r, p, L_plus_1_square, j_l_0, j_lp_0):
    quantity_quadrature_points, pre_vector, pre_integral = pre_computation_integral_quadrature(L_c, L)
    pre_V_0 = 1.
    pre_K_0 = -1.
    pre_Ka_0 = 1.
    pre_W_0 = 1.

    common_factor = 1.

    almost_A_0 = np.zeros((2 * N * L_plus_1_square, 2 * N * L_plus_1_square), dtype=np.complex128)

    r_square = np.square(r)
    
    for s in np.arange(1, N + 1):
        s_minus_1 = s - 1
        for j in np.arange(s + 1, N + 1):
            # Primero se obtienen los testeos que toman la traza j del operadores que toma la traza de la esfera s
            # Obtenci\'on de los valores con la funci\'on Integral
            data_V, data_K, data_Ka, data_W = integral_version_1(p[j - 1], r[j - 1], p[s_minus_1], L, k_0,
                                                                 quantity_quadrature_points, pre_vector,
                                                                 L_plus_1_square, pre_integral, j_l_0[:, s_minus_1],
                                                                 j_lp_0[:, s_minus_1])
            rows_sum, columns_sum = (2 * L_plus_1_square * (j - 1)), (2 * L_plus_1_square * s_minus_1)

            pre_Gamma = r_square[s_minus_1]

            # Aqu\'i se multiplica por los t\'erminos constantes
            almost_A_0[rows_sum:(rows_sum + 2 * L_plus_1_square), columns_sum:(columns_sum + 2 * L_plus_1_square)] = pre_Gamma * np.concatenate((np.concatenate((data_K * pre_K_0, data_V * pre_V_0), axis=1), np.concatenate(((data_W * pre_W_0), (data_Ka * pre_Ka_0)), axis=1)), axis=0)

            data_V, data_K, data_Ka, data_W = integral_version_1(p[s_minus_1], r[s_minus_1], p[j - 1], L, k_0,
                                                                 quantity_quadrature_points, pre_vector,
                                                                 L_plus_1_square, pre_integral, j_l_0[:, (j - 1)],
                                                                 j_lp_0[:, (j - 1)])
            rows_sum, columns_sum = (2 * L_plus_1_square * (s_minus_1)), (2 * L_plus_1_square * (j - 1))
            pre_Gamma = r_square[j - 1]
            almost_A_0[rows_sum:(rows_sum + 2 * L_plus_1_square),
            columns_sum:(columns_sum + 2 * L_plus_1_square)] = pre_Gamma * np.concatenate((np.concatenate(
                (data_K * pre_K_0, data_V * pre_V_0), axis=1), np.concatenate(((data_W * pre_W_0), (data_Ka * pre_Ka_0)),
                                                                    axis=1)), axis=0)
    return common_factor * almost_A_0


def cross_interactions_version_2_general(L_c, L, N, k_0, r, p, L_plus_1_square, j_l_0, j_lp_0):
    quantity_quadrature_points, pre_vector, pre_integral = pre_computation_integral_quadrature(L_c, L)
    pre_V_0 = 1.
    pre_K_0 = -1.  # k_0
    pre_Ka_0 = 1.  # -1.
    pre_W_0 = 1.  # * k_0

    common_factor = 1j * k_0

    almost_A_0 = np.zeros((2 * N * L_plus_1_square, 2 * N * L_plus_1_square), dtype=np.complex128)

    r_square = np.square(r)

    for s in np.arange(1, N + 1):
        s_minus_1 = s - 1
        for j in np.arange(s + 1, N + 1):
            # Primero se obtienen los testeos que toman la traza j del operadores que toma la traza de la esfera s
            # Obtenci\'on de los valores con la funci\'on Integral
            data_V, data_K, data_Ka, data_W = integral_version_2(p[j - 1], r[j - 1], p[s_minus_1], L, k_0,
                                                                 quantity_quadrature_points, pre_vector,
                                                                 L_plus_1_square, pre_integral, j_l_0[:, s_minus_1],
                                                                 j_lp_0[:, s_minus_1])
            sumaFilas, sumaColumnas = (2 * L_plus_1_square * (j - 1)), (2 * L_plus_1_square * s_minus_1)

            pre_V = r_square[s_minus_1]

            # Aqu\'i se multiplica por los t\'erminos constantes
            almost_A_0[sumaFilas:(sumaFilas + 2 * L_plus_1_square),
            sumaColumnas:(sumaColumnas + 2 * L_plus_1_square)] = pre_V * np.concatenate((np.concatenate(
                (data_K * pre_K_0, data_V), axis=1), np.concatenate(((data_W * pre_W_0), (data_Ka * pre_Ka_0)),
                                                                    axis=1)), axis=0)

            data_V, data_K, data_Ka, data_W = integral_version_2(p[s_minus_1], r[s_minus_1], p[j - 1], L, k_0,
                                                                 quantity_quadrature_points, pre_vector,
                                                                 L_plus_1_square, pre_integral, j_l_0[:, (j - 1)],
                                                                 j_lp_0[:, (j - 1)])
            sumaFilas, sumaColumnas = (2 * L_plus_1_square * (s_minus_1)), (2 * L_plus_1_square * (j - 1))
            pre_V = r_square[j - 1]
            almost_A_0[sumaFilas:(sumaFilas + 2 * L_plus_1_square),
            sumaColumnas:(sumaColumnas + 2 * L_plus_1_square)] = pre_V * np.concatenate((np.concatenate(
                (data_K * pre_K_0, data_V), axis=1), np.concatenate(((data_W * pre_W_0), (data_Ka * pre_Ka_0)),
                                                                    axis=1)), axis=0)
    return common_factor * almost_A_0


def cross_interactions_version_3_general(L_c, L, N, k_0, r, p, L_plus_1_square, j_l_0, j_lp_0):
    quantity_quadrature_points, pre_vector, pre_integral = pre_computation_integral_quadrature(L_c, L)
    pre_V_0 = 1.
    pre_K_0 = -1.  # k_0
    pre_Ka_0 = 1.  # -1.
    pre_W_0 = 1.  # * k_0

    common_factor = 1j * k_0

    almost_A_0 = np.zeros((2 * N * L_plus_1_square, 2 * N * L_plus_1_square), dtype=np.complex128)

    r_square = np.square(r)

    for s in np.arange(1, N + 1):
        s_minus_1 = s - 1
        for j in np.arange(s + 1, N + 1):
            # Primero se obtienen los testeos que toman la traza j del operadores que toma la traza de la esfera s
            # Obtenci\'on de los valores con la funci\'on Integral
            data_V, data_K, data_Ka, data_W = integral_version_3(p[j - 1], r[j - 1], p[s_minus_1], L, k_0,
                                                                 quantity_quadrature_points, pre_vector,
                                                                 L_plus_1_square, pre_integral, j_l_0[:, s_minus_1],
                                                                 j_lp_0[:, s_minus_1])
            sumaFilas, sumaColumnas = (2 * L_plus_1_square * (j - 1)), (2 * L_plus_1_square * s_minus_1)

            pre_V = r_square[s_minus_1]

            # Aqu\'i se multiplica por los t\'erminos constantes
            almost_A_0[sumaFilas:(sumaFilas + 2 * L_plus_1_square),
            sumaColumnas:(sumaColumnas + 2 * L_plus_1_square)] = pre_V * np.concatenate((np.concatenate(
                (data_K * pre_K_0, data_V), axis=1), np.concatenate(((data_W * pre_W_0), (data_Ka * pre_Ka_0)),
                                                                    axis=1)), axis=0)

            data_V, data_K, data_Ka, data_W = integral_version_3(p[s_minus_1], r[s_minus_1], p[j - 1], L, k_0,
                                                                 quantity_quadrature_points, pre_vector,
                                                                 L_plus_1_square, pre_integral, j_l_0[:, (j - 1)],
                                                                 j_lp_0[:, (j - 1)])
            sumaFilas, sumaColumnas = (2 * L_plus_1_square * (s_minus_1)), (2 * L_plus_1_square * (j - 1))
            pre_V = r_square[j - 1]
            almost_A_0[sumaFilas:(sumaFilas + 2 * L_plus_1_square),
            sumaColumnas:(sumaColumnas + 2 * L_plus_1_square)] = pre_V * np.concatenate((np.concatenate(
                (data_K * pre_K_0, data_V), axis=1), np.concatenate(((data_W * pre_W_0), (data_Ka * pre_Ka_0)),
                                                                    axis=1)), axis=0)
    return common_factor * almost_A_0


def cross_interactions_version_1_fast_general(L_c, L, N, k_0, r, p, L_plus_1_square, j_l_0, j_lp_0):
    quantity_theta_points, quantity_phi_points, w, pre_vector = pre_computation_integral_quadrature_fast(L_c)
    PyQ, P2_plusP_plus_Q, P2_plusP_minus_Q = PesyKus(L)
    pre_V_0 = 1.
    pre_K_0 = -1.  # k_0
    pre_Ka_0 = 1.  # -1.
    pre_W_0 = 1.  # * k_0

    common_factor = 1j * k_0

    almost_A_0 = np.zeros((2 * N * L_plus_1_square, 2 * N * L_plus_1_square), dtype=np.complex128)

    r_square = np.square(r)

    for s in np.arange(1, N + 1):
        s_minus_1 = s - 1
        for j in np.arange(s + 1, N + 1):
            data_V, data_K, data_Ka, data_W = integral_version_1_fast(p[j - 1], r[j - 1], p[s_minus_1], L, k_0, quantity_theta_points, quantity_phi_points, w, pre_vector,
                                                                 L_plus_1_square, j_l_0[:, s_minus_1],
                                                                 j_lp_0[:, s_minus_1], PyQ, P2_plusP_plus_Q, P2_plusP_minus_Q)
            sumaFilas, sumaColumnas = (2 * L_plus_1_square * (j - 1)), (2 * L_plus_1_square * s_minus_1)

            pre_V = r_square[s_minus_1]

            almost_A_0[sumaFilas:(sumaFilas + 2 * L_plus_1_square),
            sumaColumnas:(sumaColumnas + 2 * L_plus_1_square)] = pre_V * np.concatenate((np.concatenate(
                (data_K * pre_K_0, data_V), axis=1), np.concatenate(((data_W * pre_W_0), (data_Ka * pre_Ka_0)),
                                                                    axis=1)), axis=0)

            data_V, data_K, data_Ka, data_W = integral_version_1_fast(p[s_minus_1], r[s_minus_1], p[j - 1], L, k_0, quantity_theta_points, quantity_phi_points, w, pre_vector,
                                                                 L_plus_1_square, j_l_0[:, (j - 1)],
                                                                 j_lp_0[:, (j - 1)], PyQ, P2_plusP_plus_Q, P2_plusP_minus_Q)
            sumaFilas, sumaColumnas = (2 * L_plus_1_square * (s_minus_1)), (2 * L_plus_1_square * (j - 1))
            pre_V = r_square[j - 1]
            almost_A_0[sumaFilas:(sumaFilas + 2 * L_plus_1_square),
            sumaColumnas:(sumaColumnas + 2 * L_plus_1_square)] = pre_V * np.concatenate((np.concatenate(
                (data_K * pre_K_0, data_V), axis=1), np.concatenate(((data_W * pre_W_0), (data_Ka * pre_Ka_0)),
                                                                    axis=1)), axis=0)
    return common_factor * almost_A_0


def cross_interactions_version_3_fast_general(L_c, L, N, k_0, r, p, L_plus_1_square, j_l_0, j_lp_0):
    quantity_theta_points, quantity_phi_points, w, pre_vector = pre_computation_integral_quadrature_fast(L_c)
    PyQ, P2_plusP_plus_Q, P2_plusP_minus_Q = PesyKus(L)
    pre_V_0 = 1.
    pre_K_0 = -1.  # k_0
    pre_Ka_0 = 1.  # -1.
    pre_W_0 = 1.  # * k_0

    common_factor = 1j * k_0

    almost_A_0 = np.zeros((2 * N * L_plus_1_square, 2 * N * L_plus_1_square), dtype=np.complex128)

    r_square = np.square(r)

    for s in np.arange(1, N + 1):
        s_minus_1 = s - 1
        for j in np.arange(s + 1, N + 1):
            # Primero se obtienen los testeos que toman la traza j del operadores que toma la traza de la esfera s
            # Obtenci\'on de los valores con la funci\'on Integral
            data_V, data_K, data_Ka, data_W = integral_version_3_fast(p[j - 1], r[j - 1], p[s_minus_1], L, k_0, quantity_theta_points, quantity_phi_points, w, pre_vector,
                                                                 L_plus_1_square, j_l_0[:, s_minus_1],
                                                                 j_lp_0[:, s_minus_1], PyQ, P2_plusP_plus_Q, P2_plusP_minus_Q)
            sumaFilas, sumaColumnas = (2 * L_plus_1_square * (j - 1)), (2 * L_plus_1_square * s_minus_1)

            pre_V = r_square[s_minus_1]

            # Aqu\'i se multiplica por los t\'erminos constantes
            almost_A_0[sumaFilas:(sumaFilas + 2 * L_plus_1_square),
            sumaColumnas:(sumaColumnas + 2 * L_plus_1_square)] = pre_V * np.concatenate((np.concatenate(
                (data_K * pre_K_0, data_V), axis=1), np.concatenate(((data_W * pre_W_0), (data_Ka * pre_Ka_0)),
                                                                    axis=1)), axis=0)

            data_V, data_K, data_Ka, data_W = integral_version_3_fast(p[s_minus_1], r[s_minus_1], p[j - 1], L, k_0, quantity_theta_points, quantity_phi_points, w, pre_vector,
                                                                 L_plus_1_square, j_l_0[:, (j - 1)],
                                                                 j_lp_0[:, (j - 1)], PyQ, P2_plusP_plus_Q, P2_plusP_minus_Q)
            sumaFilas, sumaColumnas = (2 * L_plus_1_square * (s_minus_1)), (2 * L_plus_1_square * (j - 1))
            pre_V = r_square[j - 1]
            almost_A_0[sumaFilas:(sumaFilas + 2 * L_plus_1_square),
            sumaColumnas:(sumaColumnas + 2 * L_plus_1_square)] = pre_V * np.concatenate((np.concatenate(
                (data_K * pre_K_0, data_V), axis=1), np.concatenate(((data_W * pre_W_0), (data_Ka * pre_Ka_0)),
                                                                    axis=1)), axis=0)
    return common_factor * almost_A_0


# Matrix compression
def brute_force_matrix_cut(matrix, elim):
    Frobenius_norm = np.linalg.norm(matrix)
    mask = (np.abs(matrix / Frobenius_norm) < elim)
    matrix[mask] = 0. + 0. * 1j
    largo = len(matrix)
    ceros = np.sum(mask)
    percentage = 100. * ceros / (largo ** 2)
    print('Percentage of ceros ' + str(percentage))
    return matrix, percentage


def MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0):
    # print('\n I: ' + str(768 * L * N + 1152 * N + 320))
    pre_V_0 = 1j * k[0]
    pre_V_int = 1j
    pre_K_0_times_minus_one = pre_V_0 * k[0]
    pre_W_0 = -pre_K_0_times_minus_one * k[0]

    common_factor = 2.

    X_diag = np.zeros((2 * N * L_plus_1_square))
    A_sparse = np.zeros((4 * N * L_plus_1_square), dtype=np.complex128)

    rows_A_sparse = np.zeros((4 * N * L_plus_1_square), dtype=int)

    columns_A_sparse = np.zeros((4 * N * L_plus_1_square), dtype=int)

    rango = np.arange(0, L_plus_1_square)
    super_L = np.arange(0, L + 1)

    r_square = np.square(r)
    number = 0
    for s in np.arange(1, N + 1):
        s_minus_1 = s - 1
        s_minus_1_times_2 = s_minus_1 * 2

        Pre_V_0_s = pre_V_0 * r_square[s_minus_1]
        Pre_V_s = pre_V_int * k[s] * r_square[s_minus_1]

        Pre_K_0_s_times_minus_1 = pre_K_0_times_minus_one * r_square[s_minus_1]
        Pre_Ka_0_s = -Pre_K_0_s_times_minus_1

        Pre_Ka_s = Pre_V_s * k[s]
        Pre_K_s_times_minus_1 = -Pre_Ka_s

        pre_W_0_s = pre_W_0 * r_square[s_minus_1]
        Pre_W_s = Pre_K_s_times_minus_1 * k[s]

        # C\'alculo de los valores, pero sin asignar al array de la matriz
        V_0_s_p = np.multiply(Pre_V_0_s, np.multiply(j_l[:, s_minus_1, 0], h_l[:, s_minus_1, 0]))
        V_s_p = np.multiply(Pre_V_s, np.multiply(j_l[:, s_minus_1, 1], h_l[:, s_minus_1, 1]))
        K_0_s_p_times_minus_1 = np.multiply(Pre_K_0_s_times_minus_1, np.multiply(j_lp[:, s_minus_1, 0], h_l[:, s_minus_1, 0])) - 0.5
        K_s_p_times_minus_1 = np.multiply(Pre_K_s_times_minus_1, np.multiply(j_l[:, s_minus_1, 1], h_lp[:, s_minus_1, 1])) - 0.5
        Ka_0_s_p = np.multiply(Pre_Ka_0_s, np.multiply(j_l[:, s_minus_1, 0], h_lp[:, s_minus_1, 0])) - 0.5
        Ka_s_p = np.multiply(Pre_Ka_s, np.multiply(j_lp[:, s_minus_1, 1], h_l[:, s_minus_1, 1])) - 0.5
        W_0_s_p = np.multiply(pre_W_0_s, np.multiply(j_lp[:, s_minus_1, 0], h_lp[:, s_minus_1, 0]))
        W_s_p = np.multiply(Pre_W_s, np.multiply(j_lp[:, s_minus_1, 1], h_lp[:, s_minus_1, 1]))

        X_diag[(s_minus_1_times_2 * L_plus_1_square):((s_minus_1_times_2 + 1) * L_plus_1_square)] = -1.

        X_diag[((s_minus_1_times_2 + 1) * L_plus_1_square):(s * 2 * L_plus_1_square)] = a[s_minus_1]

        almost_A_0[(rango + L_plus_1_square * s_minus_1_times_2), (L_plus_1_square * (1 + s_minus_1_times_2) + rango)] = np.repeat(V_0_s_p, (
                super_L * 2 + 1))

        A_sparse[number:(number + L_plus_1_square)] = np.repeat(V_s_p, (super_L * 2 + 1))
        rows_A_sparse[number:(number + L_plus_1_square)] = rango + L_plus_1_square * (s_minus_1_times_2)
        columns_A_sparse[number:(number + L_plus_1_square)] = L_plus_1_square * (1 + s_minus_1_times_2) + rango
        number = number + L_plus_1_square

        almost_A_0[(rango + L_plus_1_square * s_minus_1_times_2), (L_plus_1_square * s_minus_1_times_2 + rango)] = np.repeat(K_0_s_p_times_minus_1, (
                super_L * 2 + 1))

        A_sparse[number:(number + L_plus_1_square)] = np.repeat(K_s_p_times_minus_1, (super_L * 2 + 1))
        rows_A_sparse[number:(number + L_plus_1_square)] = rango + L_plus_1_square * (s_minus_1_times_2)
        columns_A_sparse[number:(number + L_plus_1_square)] = L_plus_1_square * (s_minus_1_times_2) + rango
        number = number + L_plus_1_square

        almost_A_0[(rango + L_plus_1_square * (s_minus_1_times_2 + 1)), (L_plus_1_square * (s_minus_1_times_2 + 1) + rango)] = np.repeat(
            Ka_0_s_p, (super_L * 2 + 1))

        A_sparse[number:(number + L_plus_1_square)] = np.repeat(Ka_s_p, (super_L * 2 + 1))
        rows_A_sparse[number:(number + L_plus_1_square)] = rango + L_plus_1_square * (s_minus_1_times_2 + 1)
        columns_A_sparse[number:(number + L_plus_1_square)] = L_plus_1_square * (s_minus_1_times_2 + 1) + rango
        number = number + L_plus_1_square

        almost_A_0[(rango + L_plus_1_square * (s_minus_1_times_2 + 1)), (L_plus_1_square * s_minus_1_times_2 + rango)] = np.repeat(W_0_s_p, (
                super_L * 2 + 1))

        A_sparse[number:(number + L_plus_1_square)] = np.repeat(W_s_p, (super_L * 2 + 1))
        rows_A_sparse[number:(number + L_plus_1_square)] = rango + L_plus_1_square * (s_minus_1_times_2 + 1)
        columns_A_sparse[number:(number + L_plus_1_square)] = L_plus_1_square * (s_minus_1_times_2) + rango
        number = number + L_plus_1_square

    A_sparse = sp.coo_matrix((A_sparse, (rows_A_sparse, columns_A_sparse)), dtype=np.complex128, shape=(2 * N * L_plus_1_square, 2 * N * L_plus_1_square))

    del rows_A_sparse
    del columns_A_sparse
    del rango
    del super_L
    del r_square
    del number
    return common_factor * almost_A_0, X_diag, common_factor * A_sparse.tocsr()

def second_kind_ColtonKress_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0):
    pre_V_0 = 1j * k[0]
    pre_V_int = 1j
    pre_K_0_times_minus_1 = pre_V_0 * k[0]
    pre_W_0 = -pre_K_0_times_minus_1 * k[0]

    rango = np.arange(0, L_plus_1_square)
    super_L = np.arange(0, L + 1)

    r_square = np.square(r)
    for s in np.arange(1, N + 1):
        s_minus_1 = s - 1
        s_minus_1_times_2 = s_minus_1 * 2

        Pre_V_0_s = pre_V_0 * r_square[s_minus_1]
        Pre_V_s = pre_V_int * k[s] * r_square[s_minus_1]

        Pre_K_0_s_times_minus_1 = pre_K_0_times_minus_1 * r_square[s_minus_1]
        Pre_Ka_0_s = -Pre_K_0_s_times_minus_1

        Pre_Ka_s = Pre_V_s * k[s]
        Pre_K_s_times_minus_1 = -Pre_Ka_s

        pre_W_0_s = pre_W_0 * r_square[s_minus_1]
        Pre_W_s = Pre_K_s_times_minus_1 * k[s]

        # C\'alculo de los valores, pero sin asignar al array de la matriz
        V_0_s_p = np.multiply(Pre_V_0_s, np.multiply(j_l[:, s_minus_1, 0], h_l[:, s_minus_1, 0]))
        V_s_p = np.multiply(Pre_V_s, np.multiply(j_l[:, s_minus_1, 1], h_l[:, s_minus_1, 1]))
        K_0_s_p_times_minus_1 = np.multiply(Pre_K_0_s_times_minus_1, np.multiply(j_lp[:, s_minus_1, 0], h_l[:, s_minus_1, 0])) - 0.5
        K_s_p_times_minus_1 = np.multiply(Pre_K_s_times_minus_1, np.multiply(j_l[:, s_minus_1, 1], h_lp[:, s_minus_1, 1])) - 0.5
        Ka_0_s_p = np.multiply(Pre_Ka_0_s, np.multiply(j_l[:, s_minus_1, 0], h_lp[:, s_minus_1, 0])) - 0.5
        Ka_s_p = np.multiply(Pre_Ka_s, np.multiply(j_lp[:, s_minus_1, 1], h_l[:, s_minus_1, 1])) - 0.5
        W_0_s_p = np.multiply(pre_W_0_s, np.multiply(j_lp[:, s_minus_1, 0], h_lp[:, s_minus_1, 0]))
        W_s_p = np.multiply(Pre_W_s, np.multiply(j_lp[:, s_minus_1, 1], h_lp[:, s_minus_1, 1]))

        almost_A_0[(rango + L_plus_1_square * s_minus_1_times_2), (
                    L_plus_1_square * (1 + s_minus_1_times_2) + rango)] = np.repeat(V_0_s_p - V_s_p, (super_L * 2 + 1))

        almost_A_0[
            (rango + L_plus_1_square * s_minus_1_times_2), (L_plus_1_square * s_minus_1_times_2 + rango)] = np.repeat(
            (K_0_s_p_times_minus_1 + K_s_p_times_minus_1 + 1.), (super_L * 2 + 1))

        almost_A_0[(rango + L_plus_1_square * (s_minus_1_times_2 + 1)), (
                    L_plus_1_square * (s_minus_1_times_2 + 1) + rango)] = np.repeat((Ka_0_s_p + Ka_s_p + 1.), (super_L * 2 + 1))

        almost_A_0[(rango + L_plus_1_square * (s_minus_1_times_2 + 1)), (
                    L_plus_1_square * s_minus_1_times_2 + rango)] = np.repeat(W_0_s_p - W_s_p, (super_L * 2 + 1))
    del rango
    del super_L
    del r_square

    return almost_A_0

# Matrices building
# ---
def MTF_matrix_building_v2(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0):
    pre_V_0 = 1j * k[0]
    pre_V_int = 1j
    pre_K_0_times_minus_one = pre_V_0 * k[0]
    pre_W_0 = -pre_K_0_times_minus_one * k[0]

    common_factor = 2.

    X_diag = np.zeros((2 * N * L_plus_1_square))
    A_sparse = np.zeros((4 * N * L_plus_1_square), dtype=np.complex128)

    rows_A_sparse = np.zeros((4 * N * L_plus_1_square), dtype=int)

    columns_A_sparse = np.zeros((4 * N * L_plus_1_square), dtype=int)

    rango = np.arange(0, L_plus_1_square)
    super_L = np.arange(0, L + 1)

    r_square = np.square(r)
    number = 0
    for s in np.arange(1, N + 1):
        s_minus_1 = s - 1
        s_minus_1_times_2 = s_minus_1 * 2

        Pre_V_0_s = pre_V_0 * r_square[s_minus_1]
        Pre_V_s = pre_V_int * k[s] * r_square[s_minus_1]

        Pre_K_0_s_times_minus_1 = pre_K_0_times_minus_one * r_square[s_minus_1]
        Pre_Ka_0_s = -Pre_K_0_s_times_minus_1

        Pre_Ka_s = Pre_V_s * k[s]
        Pre_K_s_times_minus_1 = -Pre_Ka_s

        pre_W_0_s = pre_W_0 * r_square[s_minus_1]
        Pre_W_s = Pre_K_s_times_minus_1 * k[s]

        # Boundary integral operators pre-values
        V_0_s_p = np.multiply(Pre_V_0_s, np.multiply(j_l[:, s_minus_1, 0], h_l[:, s_minus_1, 0]))
        V_s_p = np.multiply(Pre_V_s, np.multiply(j_l[:, s_minus_1, 1], h_l[:, s_minus_1, 1]))
        K_0_s_p_times_minus_1 = np.multiply(Pre_K_0_s_times_minus_1,
                                            np.multiply(j_lp[:, s_minus_1, 0], h_l[:, s_minus_1, 0]) + np.multiply(j_l[:, s_minus_1, 0], h_lp[:, s_minus_1, 0]))/2. - 0.5
        K_s_p_times_minus_1 = np.multiply(Pre_K_s_times_minus_1,
                                          np.multiply(j_lp[:, s_minus_1, 0], h_l[:, s_minus_1, 0]) + np.multiply(j_lp[:, s_minus_1, 0], h_l[:, s_minus_1, 0]))/2. - 0.5
        Ka_0_s_p = np.multiply(Pre_Ka_0_s, np.multiply(j_l[:, s_minus_1, 0], h_lp[:, s_minus_1, 0]) + np.multiply(j_lp[:, s_minus_1, 0], h_l[:, s_minus_1, 0]))/2. - 0.5
        Ka_s_p = np.multiply(Pre_Ka_s, np.multiply(j_lp[:, s_minus_1, 1], h_l[:, s_minus_1, 1]) + np.multiply(j_lp[:, s_minus_1, 1], h_l[:, s_minus_1, 1]))/2. - 0.5
        W_0_s_p = np.multiply(pre_W_0_s, np.multiply(j_lp[:, s_minus_1, 0], h_lp[:, s_minus_1, 0]))
        W_s_p = np.multiply(Pre_W_s, np.multiply(j_lp[:, s_minus_1, 1], h_lp[:, s_minus_1, 1]))

        X_diag[(s_minus_1_times_2 * L_plus_1_square):((s_minus_1_times_2 + 1) * L_plus_1_square)] = -1.

        X_diag[((s_minus_1_times_2 + 1) * L_plus_1_square):(s * 2 * L_plus_1_square)] = a[s_minus_1]

        almost_A_0[(rango + L_plus_1_square * s_minus_1_times_2), (
                    L_plus_1_square * (1 + s_minus_1_times_2) + rango)] = np.repeat(V_0_s_p, (
                super_L * 2 + 1))

        A_sparse[number:(number + L_plus_1_square)] = np.repeat(V_s_p, (super_L * 2 + 1))
        rows_A_sparse[number:(number + L_plus_1_square)] = rango + L_plus_1_square * (s_minus_1_times_2)
        columns_A_sparse[number:(number + L_plus_1_square)] = L_plus_1_square * (1 + s_minus_1_times_2) + rango
        number = number + L_plus_1_square

        almost_A_0[
            (rango + L_plus_1_square * s_minus_1_times_2), (L_plus_1_square * s_minus_1_times_2 + rango)] = np.repeat(
            K_0_s_p_times_minus_1, (
                    super_L * 2 + 1))

        A_sparse[number:(number + L_plus_1_square)] = np.repeat(K_s_p_times_minus_1, (super_L * 2 + 1))
        rows_A_sparse[number:(number + L_plus_1_square)] = rango + L_plus_1_square * (s_minus_1_times_2)
        columns_A_sparse[number:(number + L_plus_1_square)] = L_plus_1_square * (s_minus_1_times_2) + rango
        number = number + L_plus_1_square

        almost_A_0[(rango + L_plus_1_square * (s_minus_1_times_2 + 1)), (
                    L_plus_1_square * (s_minus_1_times_2 + 1) + rango)] = np.repeat(
            Ka_0_s_p, (super_L * 2 + 1))

        A_sparse[number:(number + L_plus_1_square)] = np.repeat(Ka_s_p, (super_L * 2 + 1))
        rows_A_sparse[number:(number + L_plus_1_square)] = rango + L_plus_1_square * (s_minus_1_times_2 + 1)
        columns_A_sparse[number:(number + L_plus_1_square)] = L_plus_1_square * (s_minus_1_times_2 + 1) + rango
        number = number + L_plus_1_square

        almost_A_0[(rango + L_plus_1_square * (s_minus_1_times_2 + 1)), (
                    L_plus_1_square * s_minus_1_times_2 + rango)] = np.repeat(W_0_s_p, (
                super_L * 2 + 1))

        A_sparse[number:(number + L_plus_1_square)] = np.repeat(W_s_p, (super_L * 2 + 1))
        rows_A_sparse[number:(number + L_plus_1_square)] = rango + L_plus_1_square * (s_minus_1_times_2 + 1)
        columns_A_sparse[number:(number + L_plus_1_square)] = L_plus_1_square * (s_minus_1_times_2) + rango
        number = number + L_plus_1_square

    A_sparse = sp.coo_matrix((A_sparse, (rows_A_sparse, columns_A_sparse)), dtype=np.complex128,
                             shape=(2 * N * L_plus_1_square, 2 * N * L_plus_1_square))

    del rows_A_sparse
    del columns_A_sparse
    del rango
    del super_L
    del r_square
    del number
    return common_factor * almost_A_0, X_diag, common_factor * A_sparse.tocsr()


def second_kind_ColtonKress_matrix_building_v2(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0):
    pre_V_0 = 1j * k[0]
    pre_V_int = 1j
    pre_K_0_times_minus_1 = pre_V_0 * k[0]
    pre_W_0 = -pre_K_0_times_minus_1 * k[0]

    rango = np.arange(0, L_plus_1_square)
    super_L = np.arange(0, L + 1)

    r_square = np.square(r)
    for s in np.arange(1, N + 1):
        s_minus_1 = s - 1
        s_minus_1_times_2 = s_minus_1 * 2

        Pre_V_0_s = pre_V_0 * r_square[s_minus_1]
        Pre_V_s = pre_V_int * k[s] * r_square[s_minus_1]

        Pre_K_0_s_times_minus_1 = pre_K_0_times_minus_1 * r_square[s_minus_1]
        Pre_Ka_0_s = -Pre_K_0_s_times_minus_1

        Pre_Ka_s = Pre_V_s * k[s]
        Pre_K_s_times_minus_1 = -Pre_Ka_s

        pre_W_0_s = pre_W_0 * r_square[s_minus_1]
        Pre_W_s = Pre_K_s_times_minus_1 * k[s]

        # C\'alculo de los valores, pero sin asignar al array de la matriz
        V_0_s_p = np.multiply(Pre_V_0_s, np.multiply(j_l[:, s_minus_1, 0], h_l[:, s_minus_1, 0]))
        V_s_p = np.multiply(Pre_V_s, np.multiply(j_l[:, s_minus_1, 1], h_l[:, s_minus_1, 1]))
        K_0_s_p_times_minus_1 = np.multiply(Pre_K_0_s_times_minus_1,
                                            np.multiply(j_lp[:, s_minus_1, 0], h_l[:, s_minus_1, 0]) + np.multiply(
                                                j_l[:, s_minus_1, 0], h_lp[:, s_minus_1, 0])) / 2. - 0.5
        K_s_p_times_minus_1 = np.multiply(Pre_K_s_times_minus_1,
                                          np.multiply(j_lp[:, s_minus_1, 0], h_l[:, s_minus_1, 0]) + np.multiply(
                                              j_lp[:, s_minus_1, 0], h_l[:, s_minus_1, 0])) / 2. - 0.5
        Ka_0_s_p = np.multiply(Pre_Ka_0_s, np.multiply(j_l[:, s_minus_1, 0], h_lp[:, s_minus_1, 0]) + np.multiply(
            j_lp[:, s_minus_1, 0], h_l[:, s_minus_1, 0])) / 2. - 0.5
        Ka_s_p = np.multiply(Pre_Ka_s, np.multiply(j_lp[:, s_minus_1, 1], h_l[:, s_minus_1, 1]) + np.multiply(
            j_lp[:, s_minus_1, 1], h_l[:, s_minus_1, 1])) / 2. - 0.5
        W_0_s_p = np.multiply(pre_W_0_s, np.multiply(j_lp[:, s_minus_1, 0], h_lp[:, s_minus_1, 0]))
        W_s_p = np.multiply(Pre_W_s, np.multiply(j_lp[:, s_minus_1, 1], h_lp[:, s_minus_1, 1]))

        almost_A_0[(rango + L_plus_1_square * s_minus_1_times_2), (
                    L_plus_1_square * (1 + s_minus_1_times_2) + rango)] = np.repeat(V_0_s_p - V_s_p, (super_L * 2 + 1))

        almost_A_0[
            (rango + L_plus_1_square * s_minus_1_times_2), (L_plus_1_square * s_minus_1_times_2 + rango)] = np.repeat(
            (K_0_s_p_times_minus_1 + K_s_p_times_minus_1 + 1.), (super_L * 2 + 1))

        almost_A_0[(rango + L_plus_1_square * (s_minus_1_times_2 + 1)), (
                    L_plus_1_square * (s_minus_1_times_2 + 1) + rango)] = np.repeat((Ka_0_s_p + Ka_s_p + 1.), (super_L * 2 + 1))

        almost_A_0[(rango + L_plus_1_square * (s_minus_1_times_2 + 1)), (
                    L_plus_1_square * s_minus_1_times_2 + rango)] = np.repeat(W_0_s_p - W_s_p, (super_L * 2 + 1))
    del rango
    del super_L
    del r_square

    return almost_A_0


# b buildings
# ---
def MTF_b_building_plane_wave(N, a, p, k_0, j_l_0, j_lp_0, L, L_plus_1_square):
    #Right hand side of the MTF formulation
    common_factor = 2. * np.sqrt(np.pi)

    b_sparse = np.zeros(4 * N * (L + 1), dtype=np.complex128)
    b_rows = np.zeros(4 * N * (L + 1), dtype=int)

    rango = np.arange(0, L + 1)

    L_plus_1 = L + 1
    for s in np.arange(1, N + 1):
        # Variables auxiliares.
        s_minus_1 = s - 1
        s_minus_1_times_2 = s_minus_1 * 2
        # Variable auxiliar
        aux = common_factor * np.exp(1j * np.dot([0., 0., k_0], p[s - 1])) * 1j ** rango * np.sqrt(2 * rango + 1)
        number = 4 * s_minus_1 * L_plus_1
        b_sparse[number:number + L_plus_1] = -aux * j_l_0[:, s_minus_1]
        b_rows[number:number + L_plus_1] = (rango ** 2 + rango) + L_plus_1_square * s_minus_1_times_2
        number = number + L_plus_1

        b_sparse[number:number + L_plus_1] = aux * j_lp_0[:, s_minus_1] * k_0
        b_rows[number:number + L_plus_1] = (rango ** 2 + rango) + L_plus_1_square * (s_minus_1_times_2 + 1)
        number = number + L_plus_1

        b_sparse[number:number + L_plus_1] = aux * j_l_0[:, s_minus_1]
        b_rows[number:number + L_plus_1] = (rango ** 2 + rango) + L_plus_1_square * (s_minus_1_times_2 + 2 * N)
        number = number + L_plus_1

        b_sparse[number:number + L_plus_1] = aux * k_0 * j_lp_0[:, s_minus_1] * a[s_minus_1]
        b_rows[number:number + L_plus_1] = (rango ** 2 + rango) + L_plus_1_square * (s_minus_1_times_2 + 2 * N + 1)
        number = number + L_plus_1

    return b_sparse, b_rows


def second_kind_ColtonKress_b_building(N, a, p, k_0, j_l_0, j_lp_0, L, L_plus_1_square):
    #Right hand side of the MTF formulation
    common_factor = 2. * np.sqrt(np.pi)

    b_sparse = np.zeros(2 * N * (L + 1), dtype=np.complex128)
    b_rows = np.zeros(2 * N * (L + 1), dtype=int)

    rango = np.arange(0, L + 1)

    L_plus_1 = L + 1
    for s in np.arange(1, N + 1):
        # Variables auxiliares.
        s_minus_1 = s - 1
        s_minus_1_times_2 = s_minus_1 * 2
        # Variable auxiliar
        aux = common_factor * np.exp(1j * np.dot([0., 0., k_0], p[s - 1])) * 1j ** rango * np.sqrt(2 * rango + 1)
        number = 2 * s_minus_1 * L_plus_1
        b_sparse[number:number + L_plus_1] = -aux * j_l_0[:, s_minus_1]
        b_rows[number:number + L_plus_1] = (rango ** 2 + rango) + L_plus_1_square * s_minus_1_times_2
        number = number + L_plus_1

        b_sparse[number:number + L_plus_1] = aux * j_lp_0[:, s_minus_1] * k_0
        b_rows[number:number + L_plus_1] = (rango ** 2 + rango) + L_plus_1_square * (s_minus_1_times_2 + 1)
        number = number + L_plus_1

    return b_sparse, b_rows


def plane_wave_traces(N, p, k_0, j_l_0, j_lp_0, L, L_plus_1_square):
    # output: b_sparse, b_rows
    #
    # upper half of the vector is the Dirichlet trace of a plane wave
    # down half of the vector is minus the Neumann trace a plane wave
    common_factor = 2. * np.sqrt(np.pi)

    b_sparse = np.zeros(2 * N * (L + 1), dtype=np.complex128)
    b_rows = np.zeros(2 * N * (L + 1), dtype=int)

    rango = np.arange(0, L + 1)

    L_plus_1 = L + 1
    for s in np.arange(1, N + 1):
        # Variables auxiliares.
        s_minus_1 = s - 1
        s_minus_1_times_2 = s_minus_1 * 2
        # Variable auxiliar
        aux = common_factor * np.exp(1j * np.dot([0., 0., k_0], p[s - 1])) * 1j ** rango * np.sqrt(2 * rango + 1)
        number = 2 * s_minus_1 * L_plus_1
        b_sparse[number:number + L_plus_1] = - aux * j_l_0[:, s_minus_1]
        b_rows[number:number + L_plus_1] = (rango ** 2 + rango) + L_plus_1_square * s_minus_1_times_2
        number = number + L_plus_1

        b_sparse[number:number + L_plus_1] = aux * j_lp_0[:, s_minus_1] * k_0
        b_rows[number:number + L_plus_1] = (rango ** 2 + rango) + L_plus_1_square * (s_minus_1_times_2 + 1)
        number = number + L_plus_1

    return b_sparse, b_rows


# Direct solutions
# ---
def numpy_direct_solver(matrix, b):
    # matrix in numpy format
    # b in numpy format
    return np.linalg.solve(matrix, b)


def direct_solution_MTF(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0):
    A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                almost_A_0)

    MTF_matrix = np.concatenate((np.concatenate((A_0_times_2, sp.dia_matrix(((1. / X_diag), np.array([0])), shape=np.shape(A_0_times_2)).toarray()), axis=1), np.concatenate((sp.dia_matrix((X_diag, np.array([0])), shape=np.shape(A_sparse_times_2)).toarray(), A_sparse_times_2.toarray()), axis=1)), axis=0)
    del A_0_times_2
    del X_diag
    del A_sparse_times_2

    b_sparse, b_rows = MTF_b_building_plane_wave(N, a, p, k[0], j_l[:, :, 0], j_lp[:, :, 0], L, L_plus_1_square)
    if L == 1:
        b_sparse[0:len(b_sparse)] = 0.
        b_sparse[1] = - 6.13996025
        b_sparse[3] = -1. * - 6.13996025
        b_sparse[5] = 6.13996025
        b_sparse[7] = - 6.13996025 * - 0.5
    b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(4 * N * (L + 1), dtype=int))), dtype=np.complex128,
                      shape=(4 * N * L_plus_1_square, 1))
    del b_rows
    del b_sparse
    p = b.toarray()
    # print('p ' +str(p))
    return MTF_matrix, p, numpy_direct_solver(MTF_matrix, b.toarray())


def direct_solution_second_kind_ColtonKress(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0):
    second_kind_matrix = second_kind_ColtonKress_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                 almost_A_0)
    b_sparse, b_rows = second_kind_ColtonKress_b_building(N, a, p, k[0], j_l[:, :, 0], j_lp[:, :, 0], L, L_plus_1_square)
    b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(2 * N * (L + 1), dtype=int))), dtype=np.complex128,
                      shape=(2 * N * L_plus_1_square, 1))
    del b_rows
    del b_sparse
    return numpy_direct_solver(second_kind_matrix, b.toarray())


def both_formulations_direct_solution(L_c, L, N, k, r, p, a, compression, elim):
    # compression: boolean      indicates if the matrix is compressed and how:
    #                           if compression==True it uses the brutal compression
    L_plus_1_square = (L + 1)**2
    j_l, j_lp, h_l, h_lp = helmholtz_pre_computations(N, k, r, L)

    almost_A_0 = cross_interactions_version_1_fast_general(L_c, L, N, k[0], r, p, L_plus_1_square, j_l[:, :, 0], j_lp[:, :, 0])

    if compression:
        almost_A_0, percentage = brute_force_matrix_cut(almost_A_0, elim)

    MTF_solution = direct_solution_MTF(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0)

    second_kind_ColtonKress_solution = direct_solution_second_kind_ColtonKress(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0)
    second_kind_traces = traces_second_kind_formulation(second_kind_ColtonKress_solution, r, a, p, k[0], L, L_plus_1_square,
                                                        N, k, j_l, j_lp, h_l, h_lp)
    print(np.linalg.norm(MTF_solution - second_kind_traces))
    return MTF_solution, second_kind_ColtonKress_solution, second_kind_traces


# Experiments with one sphere.
#---
def direct_solution_MTF_with_one_sphere(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp):
    almost_A_0 = np.zeros((2 * (L + 1) ** 2, 2 * (L + 1) ** 2), dtype=np.complex128)
    A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                almost_A_0)
    MTF_matrix = np.concatenate((np.concatenate((A_0_times_2, sp.dia_matrix(((1. / X_diag), np.array([0])), shape=np.shape(A_0_times_2)).toarray()), axis=1), np.concatenate((sp.dia_matrix((X_diag, np.array([0])), shape=np.shape(A_sparse_times_2)).toarray(), A_sparse_times_2.toarray()), axis=1)), axis=0)
    del A_0_times_2
    del X_diag
    del A_sparse_times_2
    b_sparse, b_rows = MTF_b_building_plane_wave(N, a, p, k[0], j_l[:, :, 0], j_lp[:, :, 0], L, L_plus_1_square)
    b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(4 * N * (L + 1), dtype=int))), dtype=np.complex128,
                      shape=(4 * N * L_plus_1_square, 1))
    del b_rows
    del b_sparse
    return numpy_direct_solver(MTF_matrix, b.toarray())


def solving_MTF_system_with_one_sphere_gmres_with_Calderon_preconditioning(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, tolerance):
    almost_A_0 = np.zeros((2 * (L + 1) ** 2, 2 * (L + 1) ** 2), dtype=np.complex128)
    A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                almost_A_0)
    A_0_times_2 = sp.csr_matrix(A_0_times_2)

    def MTF_block_matrix_times_vector(v):
        num = len(X_diag)
        x = np.zeros(np.shape(v), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = A_0_times_2.dot(v_up) + (1. / X_diag) * v_down
        x[num:(2 * num)] = X_diag * v_up + A_sparse_times_2.dot(v_down)
        return x

    def MTF_block_matrix_conjugate_transpose_times_vector(v):
        num = len(X_diag)
        x = np.zeros(np.shape(v), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = A_0_times_2.conjugate().transpose().dot(v_up) + X_diag * v_down
        x[num:(2 * num)] = (1. / X_diag) * v_up + A_sparse_times_2.conjugate().transpose().dot(v_down)
        return x

    MTF_linear_operator = sp.linalg.LinearOperator((4 * N * L_plus_1_square, 4 * N * L_plus_1_square),
                                                   matvec=MTF_block_matrix_times_vector,
                                                   matmat=MTF_block_matrix_times_vector,
                                                   rmatvec=MTF_block_matrix_conjugate_transpose_times_vector,
                                                   rmatmat=MTF_block_matrix_conjugate_transpose_times_vector)
    b_sparse, b_rows = MTF_b_building_plane_wave(N, a, p, k[0], j_l[:, :, 0], j_lp[:, :, 0], L, L_plus_1_square)
    b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(4 * N * (L + 1), dtype=int))), dtype=np.complex128,
                      shape=(4 * N * L_plus_1_square, 1))
    del b_sparse
    del b_rows
    norms = []

    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.

    def Calderon_preconditioner(v):
        num = len(X_diag)
        x = np.zeros(np.shape(v), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = A_0_times_2.dot(v_up) / 2.
        x[num:(2 * num)] = A_sparse_times_2.dot(v_down) / 2.
        return x

    Calderon_preconditioner_linear_operator = sp.linalg.LinearOperator(
        (4 * N * L_plus_1_square, 4 * N * L_plus_1_square),
        matvec=Calderon_preconditioner,
        matmat=Calderon_preconditioner)
    solution, info = sp.linalg.gmres(MTF_linear_operator, b.toarray(), tol=tolerance,
                                     restart=4 * N * L_plus_1_square, M=Calderon_preconditioner_linear_operator,
                                     callback=callback_function,
                                     callback_type='pr_norm')
    return solution, info, np.asarray(norms)


def solving_MTF_system_with_one_sphere_gmres_with_Calderon_preconditioning_version_0(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, tolerance):
    almost_A_0 = np.zeros((2 * (L + 1) ** 2, 2 * (L + 1) ** 2), dtype=np.complex128)
    A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                almost_A_0)
    A_0_times_2 = sp.csr_matrix(A_0_times_2)

    def MTF_block_matrix_times_vector(v):
        num = len(X_diag)
        x = np.zeros(np.shape(v), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = A_0_times_2.dot(A_0_times_2.dot(v_up) + (1. / X_diag) * v_down)
        x[num:(2 * num)] = A_sparse_times_2.dot(X_diag * v_up + A_sparse_times_2.dot(v_down))
        return x

    MTF_linear_operator = sp.linalg.LinearOperator((4 * N * L_plus_1_square, 4 * N * L_plus_1_square),
                                                   matvec=MTF_block_matrix_times_vector,
                                                   matmat=MTF_block_matrix_times_vector)
    b_sparse, b_rows = MTF_b_building_plane_wave(N, a, p, k[0], j_l[:, :, 0], j_lp[:, :, 0], L, L_plus_1_square)
    b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(4 * N * (L + 1), dtype=int))), dtype=np.complex128,
                      shape=(4 * N * L_plus_1_square, 1))
    del b_sparse
    del b_rows
    bx = b.toarray()
    del b
    num = len(X_diag)
    bx[0:num] = A_0_times_2.dot(bx[0:num])
    bx[num:(2 * num)] = A_sparse_times_2.dot(bx[num:(2 * num)])
    norms = []

    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.

    solution, info = sp.linalg.gmres(MTF_linear_operator, bx, tol=tolerance,
                                     restart=4 * N * L_plus_1_square, callback=callback_function,
                                     callback_type='pr_norm')

    return solution, info, np.asarray(norms)


def solving_MTF_system_with_one_sphere_gmres(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, tolerance):
    almost_A_0 = np.zeros((2 * (L + 1) ** 2, 2 * (L + 1) ** 2), dtype=np.complex128)
    A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                almost_A_0)
    A_0_times_2 = sp.csr_matrix(A_0_times_2)

    def MTF_block_matrix_times_vector(v):
        num = len(X_diag)
        x = np.zeros(np.shape(v), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = A_0_times_2.dot(v_up) + (1. / X_diag) * v_down
        x[num:(2 * num)] = X_diag * v_up + A_sparse_times_2.dot(v_down)
        return x

    def MTF_block_matrix_conjugate_transpose_times_vector(v):
        num = len(X_diag)
        x = np.zeros(np.shape(v), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = A_0_times_2.conjugate().transpose().dot(v_up) + X_diag * v_down
        x[num:(2 * num)] = (1. / X_diag) * v_up + A_sparse_times_2.conjugate().transpose().dot(v_down)
        return x

    MTF_linear_operator = sp.linalg.LinearOperator((4 * N * L_plus_1_square, 4 * N * L_plus_1_square),
                                                   matvec=MTF_block_matrix_times_vector,
                                                   matmat=MTF_block_matrix_times_vector,
                                                   rmatvec=MTF_block_matrix_conjugate_transpose_times_vector,
                                                   rmatmat=MTF_block_matrix_conjugate_transpose_times_vector)
    b_sparse, b_rows = MTF_b_building_plane_wave(N, a, p, k[0], j_l[:, :, 0], j_lp[:, :, 0], L, L_plus_1_square)
    b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(4 * N * (L + 1), dtype=int))), dtype=np.complex128,
                      shape=(4 * N * L_plus_1_square, 1))
    del b_sparse
    del b_rows
    norms = []

    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.

    solution, info = sp.linalg.gmres(MTF_linear_operator, b.toarray(), tol=tolerance,
                                     restart=4 * N * L_plus_1_square, callback=callback_function,
                                     callback_type='pr_norm')
    return solution, info, np.asarray(norms)


def solving_second_kind_ColtonKress_system_with_one_spheres_with_gmres(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, tolerance):
    almost_A_0 = np.zeros((2 * (L + 1) ** 2, 2 * (L + 1) ** 2), dtype=np.complex128)
    second_kind_matrix = second_kind_ColtonKress_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                 almost_A_0)
    def second_kind_ColtonKress_matrix_times_vector(v):
        return np.matmul(second_kind_matrix, v)

    second_kind_ColtonKress_operator = sp.linalg.LinearOperator(np.shape(second_kind_matrix), matvec=second_kind_ColtonKress_matrix_times_vector)
    b_sparse, b_rows = second_kind_ColtonKress_b_building(N, a, p, k[0], j_l[:, :, 0], j_lp[:, :, 0], L,
                                                          L_plus_1_square)
    b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(2 * N * (L + 1), dtype=int))), dtype=np.complex128,
                      shape=(2 * N * L_plus_1_square, 1))
    del b_sparse
    del b_rows
    norms =[]

    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.

    solution, info = sp.linalg.gmres(second_kind_ColtonKress_operator, b.toarray(), tol=tolerance, restart=4 * N * L_plus_1_square, callback=callback_function, callback_type='pr_norm')
    return solution, info, np.asarray(norms)


def MTF_eigenvalues_one_sphere(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp):
    almost_A_0 = np.zeros((2 * (L + 1) ** 2, 2 * (L + 1) ** 2), dtype=np.complex128)
    A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                almost_A_0)
    MTF_matrix = np.concatenate((np.concatenate(
        (A_0_times_2, sp.dia_matrix(((1. / X_diag), np.array([0])), shape=np.shape(A_0_times_2)).toarray()), axis=1),
                                 np.concatenate((sp.dia_matrix((X_diag, np.array([0])),
                                                               shape=np.shape(A_sparse_times_2)).toarray(),
                                                 A_sparse_times_2.toarray()), axis=1)), axis=0)
    del X_diag
    eigenvalues_MTF = np.linalg.eigvals(MTF_matrix)

    Calderon_preconditioner = np.concatenate(
        (np.concatenate((A_0_times_2, np.zeros(np.shape(A_0_times_2), dtype=np.complex128)), axis=1),
         np.concatenate((np.zeros(np.shape(A_0_times_2), dtype=np.complex128),
                         A_sparse_times_2.toarray()), axis=1)), axis=0)
    del almost_A_0
    del A_0_times_2
    del A_sparse_times_2
    eigenvalues_preconditioned = np.linalg.eigvals(np.matmul(Calderon_preconditioner,MTF_matrix))

    return eigenvalues_MTF, eigenvalues_preconditioned


def returning_V_y_W(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp):
    almost_A_0 = np.zeros((2 * (L + 1) ** 2, 2 * (L + 1) ** 2), dtype=np.complex128)
    A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                almost_A_0)
    del X_diag
    del A_sparse_times_2
    V = A_0_times_2[0:(L+1)**2, (L+1)**2:2*(L+1)**2]
    W = A_0_times_2[(L+1)**2:2*(L+1)**2, 0:(L+1)**2]
    return V, W


def checking_K(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp):
    almost_A_0 = np.zeros((2 * (L + 1) ** 2, 2 * (L + 1) ** 2), dtype=np.complex128)
    A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                almost_A_0)
    del X_diag
    del A_sparse_times_2
    K = A_0_times_2[0:(L+1)**2,0:(L+1)**2]
    print("K checking")
    print("condicionamiento " + str(np.linalg.cond(K)))
    print("determinante: " + str(np.linalg.det(K)))
    print("Multiplications of the values of its diagonal one by one")
    multiplication = 1.
    for l in np.arange(0, np.shape(K)[0]):
        multiplication = multiplication * K[l, l]
        print(multiplication)
    return K


# Iterative solutions
# ---
def linear_operator_MTF_construction_version_0(A_0_times_2, X_diag, A_sparse_times_2):
    MTF_matrix = np.concatenate((np.concatenate(
        (A_0_times_2, sp.dia_matrix(((1. / X_diag), np.array([0])), shape=np.shape(A_0_times_2)).toarray()), axis=1),
                                 np.concatenate((sp.dia_matrix((X_diag, np.array([0])),
                                                               shape=np.shape(A_sparse_times_2)).toarray(),
                                                 A_sparse_times_2.toarray()), axis=1)), axis=0)
    del A_0_times_2
    del X_diag
    del A_sparse_times_2
    def linear_operator_MTF(v):
        return MTF_matrix*v
    return sp.linalg.LinearOperator(np.shape(MTF_matrix), matvec=MTF_matrix)


def linear_operator_MTF_construction_version_1(A_0_times_2, X_diag, A_sparse_times_2):
    def MTF_block_matrix_times_vector(v):
        num = len(X_diag)
        x = np.zeros((2 * num), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = np.matmul(A_0_times_2, v_up) + (1. / X_diag) * v_down
        x[num:(2 * num)] = X_diag * v_up + A_sparse_times_2.dot(v_down)
        return x

    return 0.


def solving_MTF_system_with_gmres_version_0(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0, tolerance):
    A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                almost_A_0)
    MTF_matrix = np.concatenate((np.concatenate(
        (A_0_times_2, sp.dia_matrix(((1. / X_diag), np.array([0])), shape=np.shape(A_0_times_2)).toarray()), axis=1),
                                 np.concatenate((sp.dia_matrix((X_diag, np.array([0])),
                                                               shape=np.shape(A_sparse_times_2)).toarray(),
                                                 A_sparse_times_2.toarray()), axis=1)), axis=0)
    del A_0_times_2
    del X_diag
    del A_sparse_times_2
    def MTF_matrix_times_vector(v):
        return np.matmul(MTF_matrix, v)
    MTF_linear_operator = sp.linalg.LinearOperator(np.shape(MTF_matrix), matvec=MTF_matrix_times_vector)
    b_sparse, b_rows = MTF_b_building_plane_wave(N, a, p, k[0], j_l[:, :, 0], j_lp[:, :, 0], L, L_plus_1_square)
    b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(4 * N * (L + 1), dtype=int))), dtype=np.complex128,
                      shape=(4 * N * L_plus_1_square, 1))
    del b_sparse
    del b_rows
    norms =[]
    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.
    solution, info = sp.linalg.gmres(MTF_linear_operator, b.toarray(), tol=tolerance, restart=4*N*L_plus_1_square, callback=callback_function, callback_type='pr_norm')
    return solution, info, np.asarray(norms)


def solving_MTF_system_with_gmres_version_1(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0, tolerance):
    A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                almost_A_0)
    def MTF_block_matrix_times_vector(v):
        num = len(X_diag)
        x = np.zeros(np.shape(v), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = np.matmul(A_0_times_2, v_up) + (1./X_diag)* v_down
        x[num:(2 * num)] = X_diag*v_up + A_sparse_times_2.dot(v_down)
        return x
    MTF_linear_operator = sp.linalg.LinearOperator((4*N*L_plus_1_square, 4*N*L_plus_1_square), matvec=MTF_block_matrix_times_vector, matmat=MTF_block_matrix_times_vector)
    b_sparse, b_rows = MTF_b_building_plane_wave(N, a, p, k[0], j_l[:, :, 0], j_lp[:, :, 0], L, L_plus_1_square)
    b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(4 * N * (L + 1), dtype=int))), dtype=np.complex128,
                      shape=(4 * N * L_plus_1_square, 1))
    del b_sparse
    del b_rows
    norms =[]
    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.
    solution, info = sp.linalg.gmres(MTF_linear_operator, b.toarray(), tol=tolerance, restart=4*N*L_plus_1_square, callback=callback_function, callback_type='pr_norm')
    return solution, info, np.asarray(norms)


def solving_MTF_system_with_gmres_version_3(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0, tolerance):
    A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                almost_A_0)
    def MTF_block_matrix_times_vector(v):
        num = len(X_diag)
        x = np.zeros(np.shape(v), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = np.matmul(A_0_times_2, v_up) + (1./X_diag)* v_down
        x[num:(2 * num)] = X_diag*v_up + A_sparse_times_2.dot(v_down)
        return x
    def MTF_block_matrix_conjugate_transpose_times_vector(v):
        num = len(X_diag)
        x = np.zeros(np.shape(v), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = np.matmul(np.conjugate(np.transpose(A_0_times_2)), v_up) + X_diag* v_down
        x[num:(2 * num)] = (1./X_diag)*v_up + A_sparse_times_2.conjugate().transpose().dot(v_down)
        return x
    MTF_linear_operator = sp.linalg.LinearOperator((4*N*L_plus_1_square, 4*N*L_plus_1_square), matvec=MTF_block_matrix_times_vector, matmat=MTF_block_matrix_times_vector, rmatvec=MTF_block_matrix_conjugate_transpose_times_vector, rmatmat=MTF_block_matrix_conjugate_transpose_times_vector)
    b_sparse, b_rows = MTF_b_building_plane_wave(N, a, p, k[0], j_l[:, :, 0], j_lp[:, :, 0], L, L_plus_1_square)
    b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(4 * N * (L + 1), dtype=int))), dtype=np.complex128,
                      shape=(4 * N * L_plus_1_square, 1))
    del b_sparse
    del b_rows
    norms =[]
    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.
    solution, info = sp.linalg.gmres(MTF_linear_operator, b.toarray(), tol=tolerance, restart=4*N*L_plus_1_square, callback=callback_function, callback_type='pr_norm')
    return solution, info, np.asarray(norms)


# def solving_MTF_system_with_gmres_version_2(r, p, a, L, num, N, k, j_l, j_lp, h_l, h_lp, almost_A_0, tolerance):
#     #It isn't working
#     A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, num, N, k, j_l, j_lp, h_l, h_lp,
#                                                                 almost_A_0)
#
#     def MTF_block_matrix_times_vector(v):
#         num = len(X_diag)
#         x = np.zeros(np.shape(v), dtype=np.complex128)
#         v_up = v[0:num]
#         v_down = v[num:(2 * num)]
#         x[0:num] = np.matmul(A_0_times_2, v_up) + (1. / X_diag) * v_down
#         x[num:(2 * num)] = X_diag * v_up + A_sparse_times_2.dot(v_down)
#         return x
#
#     def MTF_block_matrix_conjugate_transpose_times_vector(v):
#         num = len(X_diag)
#         x = np.zeros(np.shape(v), dtype=np.complex128)
#         v_up = v[0:num]
#         v_down = v[num:(2 * num)]
#         x[0:num] = np.matmul(np.conjugate(np.transpose(A_0_times_2)), v_up) + X_diag * v_down
#         x[num:(2 * num)] = (1. / X_diag) * v_up + A_sparse_times_2.conjugate().transpose().dot(v_down)
#         return x
#
#     MTF_linear_operator = sp.linalg.LinearOperator((4 * N * num, 4 * N * num),
#                                                    matvec=MTF_block_matrix_times_vector,
#                                                    matmat=MTF_block_matrix_times_vector,
#                                                    rmatvec=MTF_block_matrix_conjugate_transpose_times_vector,
#                                                    rmatmat=MTF_block_matrix_conjugate_transpose_times_vector)
#     import krypy as kp
#     b_sparse, b_rows = MTF_b_building_plane_wave(N, a, p, k[0], j_l[:, :, 0], j_lp[:, :, 0], L, num)
#     b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(4 * N * (L + 1), dtype=int))), dtype=np.complex128,
#                       shape=(4 * N * num, 1))
#     del b_sparse
#     del b_rows
#     MTF_linear_operator_krypy = kp.linsys.LinearSystem(MTF_linear_operator, b.toarray())
#     gmres = kp.linsys.Gmres(MTF_linear_operator_krypy, tol=tolerance)
#     return np.copy(gmres.xk), np.copy(gmres.resnorms)


def solving_MTF_system_with_gmres_version_0_with_Calderon_preconditioning(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0, tolerance):
    A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                almost_A_0)
    MTF_matrix = np.concatenate((np.concatenate(
        (A_0_times_2, sp.dia_matrix(((1. / X_diag), np.array([0])), shape=np.shape(A_0_times_2)).toarray()), axis=1),
                                 np.concatenate((sp.dia_matrix((X_diag, np.array([0])),
                                                               shape=np.shape(A_sparse_times_2)).toarray(),
                                                 A_sparse_times_2.toarray()), axis=1)), axis=0)
    Calderon_preconditioner = np.concatenate((np.concatenate((A_0_times_2, np.zeros(np.shape(A_0_times_2),dtype=np.complex128)), axis=1),
                                 np.concatenate((np.zeros(np.shape(A_0_times_2),dtype=np.complex128),
                                                 A_sparse_times_2.toarray()), axis=1)), axis=0)
    del A_0_times_2
    del X_diag
    del A_sparse_times_2
    def MTF_matrix_times_vector(v):
        return np.matmul(np.matmul(Calderon_preconditioner,MTF_matrix), v)/2.
    MTF_linear_operator = sp.linalg.LinearOperator(np.shape(MTF_matrix), matvec=MTF_matrix_times_vector)
    b_sparse, b_rows = MTF_b_building_plane_wave(N, a, p, k[0], j_l[:, :, 0], j_lp[:, :, 0], L, L_plus_1_square)
    b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(4 * N * (L + 1), dtype=int))), dtype=np.complex128,
                      shape=(4 * N * L_plus_1_square, 1))
    del b_sparse
    del b_rows
    norms = []

    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.
    solution, info = sp.linalg.gmres(MTF_linear_operator, b.toarray(), tol=tolerance,
                                     restart=4 * N * L_plus_1_square, callback=callback_function,
                                     callback_type='pr_norm')
    return solution, info, np.asarray(norms)


def solving_MTF_system_with_gmres_version_3_with_Calderon_preconditioning(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0, tolerance):
    A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                almost_A_0)

    def MTF_block_matrix_times_vector(v):
        num = len(X_diag)
        x = np.zeros(np.shape(v), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = np.matmul(A_0_times_2, v_up) + (1. / X_diag) * v_down
        x[num:(2 * num)] = X_diag * v_up + A_sparse_times_2.dot(v_down)
        return x

    def MTF_block_matrix_conjugate_transpose_times_vector(v):
        num = len(X_diag)
        x = np.zeros(np.shape(v), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = np.matmul(np.conjugate(np.transpose(A_0_times_2)), v_up) + X_diag * v_down
        x[num:(2 * num)] = (1. / X_diag) * v_up + A_sparse_times_2.conjugate().transpose().dot(v_down)
        return x

    MTF_linear_operator = sp.linalg.LinearOperator((4 * N * L_plus_1_square, 4 * N * L_plus_1_square),
                                                   matvec=MTF_block_matrix_times_vector,
                                                   matmat=MTF_block_matrix_times_vector,
                                                   rmatvec=MTF_block_matrix_conjugate_transpose_times_vector,
                                                   rmatmat=MTF_block_matrix_conjugate_transpose_times_vector)
    b_sparse, b_rows = MTF_b_building_plane_wave(N, a, p, k[0], j_l[:, :, 0], j_lp[:, :, 0], L, L_plus_1_square)
    b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(4 * N * (L + 1), dtype=int))), dtype=np.complex128,
                      shape=(4 * N * L_plus_1_square, 1))
    del b_sparse
    del b_rows
    norms = []

    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.

    def Calderon_preconditioner(v):
        num = len(X_diag)
        x = np.zeros(np.shape(v), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = np.matmul(A_0_times_2, v_up)/2.
        x[num:(2 * num)] = A_sparse_times_2.dot(v_down)/2.
        return x
    Calderon_preconditioner_linear_operator=sp.linalg.LinearOperator((4 * N * L_plus_1_square, 4 * N * L_plus_1_square),
                                                   matvec=Calderon_preconditioner,
                                                   matmat=Calderon_preconditioner)
    solution, info = sp.linalg.gmres(MTF_linear_operator, b.toarray(), tol=tolerance,
                                     restart=4 * N * L_plus_1_square, M=Calderon_preconditioner_linear_operator, callback=callback_function,
                                     callback_type='pr_norm')
    return solution, info, np.asarray(norms)


def building_another_Calderon_preconditioner(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp):
    # print('\n I: ' + str(768 * L * N + 1152 * N + 320))
    pre_V_0 = 1j * k[0]
    pre_V_int = 1j
    pre_K_0_times_minus_one = pre_V_0 * k[0]
    pre_W_0 = -pre_K_0_times_minus_one * k[0]
    A_sparse = np.zeros((4 * N * L_plus_1_square), dtype=np.complex128)

    rows_A_sparse = np.zeros((4 * N * L_plus_1_square), dtype=int)

    columns_A_sparse = np.zeros((4 * N * L_plus_1_square), dtype=int)

    rango = np.arange(0, L_plus_1_square)
    super_L = np.arange(0, L + 1)

    r_square = np.square(r)
    number = 0
    for s in np.arange(1, N + 1):
        s_minus_1 = s - 1
        s_minus_1_times_2 = s_minus_1 * 2

        Pre_V_0_s = pre_V_0 * r_square[s_minus_1]
        Pre_V_s = pre_V_int * k[s] * r_square[s_minus_1]

        Pre_K_0_s_times_minus_1 = pre_K_0_times_minus_one * r_square[s_minus_1]
        Pre_Ka_0_s = -Pre_K_0_s_times_minus_1

        Pre_Ka_s = Pre_V_s * k[s]
        Pre_K_s_times_minus_1 = -Pre_Ka_s

        pre_W_0_s = pre_W_0 * r_square[s_minus_1]
        Pre_W_s = Pre_K_s_times_minus_1 * k[s]

        # C\'alculo de los valores, pero sin asignar al array de la matriz
        V_0_s_p = np.multiply(Pre_V_0_s, np.multiply(j_l[:, s_minus_1, 0], h_l[:, s_minus_1, 0]))
        V_s_p = np.multiply(Pre_V_s, np.multiply(j_l[:, s_minus_1, 1], h_l[:, s_minus_1, 1]))
        K_0_s_p_times_minus_1 = np.multiply(Pre_K_0_s_times_minus_1,
                                            np.multiply(j_lp[:, s_minus_1, 0], h_l[:, s_minus_1, 0])) - 0.5
        K_s_p_times_minus_1 = np.multiply(Pre_K_s_times_minus_1,
                                          np.multiply(j_l[:, s_minus_1, 1], h_lp[:, s_minus_1, 1])) - 0.5
        Ka_0_s_p = np.multiply(Pre_Ka_0_s, np.multiply(j_l[:, s_minus_1, 0], h_lp[:, s_minus_1, 0])) - 0.5
        Ka_s_p = np.multiply(Pre_Ka_s, np.multiply(j_lp[:, s_minus_1, 1], h_l[:, s_minus_1, 1])) - 0.5
        W_0_s_p = np.multiply(pre_W_0_s, np.multiply(j_lp[:, s_minus_1, 0], h_lp[:, s_minus_1, 0]))
        W_s_p = np.multiply(Pre_W_s, np.multiply(j_lp[:, s_minus_1, 1], h_lp[:, s_minus_1, 1]))

        A_sparse[number:(number + L_plus_1_square)] = np.repeat(V_0_s_p, (super_L * 2 + 1))
        rows_A_sparse[number:(number + L_plus_1_square)] = rango + L_plus_1_square * (s_minus_1_times_2)
        columns_A_sparse[number:(number + L_plus_1_square)] = L_plus_1_square * (1 + s_minus_1_times_2) + rango
        number = number + L_plus_1_square

        A_sparse[number:(number + L_plus_1_square)] = np.repeat(K_0_s_p_times_minus_1, (super_L * 2 + 1))
        rows_A_sparse[number:(number + L_plus_1_square)] = rango + L_plus_1_square * (s_minus_1_times_2)
        columns_A_sparse[number:(number + L_plus_1_square)] = L_plus_1_square * (s_minus_1_times_2) + rango
        number = number + L_plus_1_square

        A_sparse[number:(number + L_plus_1_square)] = np.repeat(Ka_0_s_p, (super_L * 2 + 1))
        rows_A_sparse[number:(number + L_plus_1_square)] = rango + L_plus_1_square * (s_minus_1_times_2 + 1)
        columns_A_sparse[number:(number + L_plus_1_square)] = L_plus_1_square * (s_minus_1_times_2 + 1) + rango
        number = number + L_plus_1_square
        A_sparse[number:(number + L_plus_1_square)] = np.repeat(W_0_s_p, (super_L * 2 + 1))
        rows_A_sparse[number:(number + L_plus_1_square)] = rango + L_plus_1_square * (s_minus_1_times_2 + 1)
        columns_A_sparse[number:(number + L_plus_1_square)] = L_plus_1_square * (s_minus_1_times_2) + rango
        number = number + L_plus_1_square

    A_sparse = sp.coo_matrix((A_sparse, (rows_A_sparse, columns_A_sparse)), dtype=np.complex128,
                             shape=(2 * N * L_plus_1_square, 2 * N * L_plus_1_square))

    del rows_A_sparse
    del columns_A_sparse
    del rango
    del super_L
    del r_square
    del number
    return A_sparse.tocsr()


def solving_MTF_system_with_gmres_version_3_with_another_Calderon_preconditioning(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0, tolerance):
    A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                almost_A_0)

    def MTF_block_matrix_times_vector(v):
        num = len(X_diag)
        x = np.zeros(np.shape(v), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = np.matmul(A_0_times_2, v_up) + (1. / X_diag) * v_down
        x[num:(2 * num)] = X_diag * v_up + A_sparse_times_2.dot(v_down)
        return x

    def MTF_block_matrix_conjugate_transpose_times_vector(v):
        num = len(X_diag)
        x = np.zeros(np.shape(v), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = np.matmul(np.conjugate(np.transpose(A_0_times_2)), v_up) + X_diag * v_down
        x[num:(2 * num)] = (1. / X_diag) * v_up + A_sparse_times_2.conjtransp().dot(v_down)
        return x

    MTF_linear_operator = sp.linalg.LinearOperator((4 * N * L_plus_1_square, 4 * N * L_plus_1_square),
                                                   matvec=MTF_block_matrix_times_vector,
                                                   matmat=MTF_block_matrix_times_vector,
                                                   rmatvec=MTF_block_matrix_conjugate_transpose_times_vector,
                                                   rmatmat=MTF_block_matrix_conjugate_transpose_times_vector)
    b_sparse, b_rows = MTF_b_building_plane_wave(N, a, p, k[0], j_l[:, :, 0], j_lp[:, :, 0], L, L_plus_1_square)
    b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(4 * N * (L + 1), dtype=int))), dtype=np.complex128,
                      shape=(4 * N * L_plus_1_square, 1))
    del b_sparse
    del b_rows
    norms = []

    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.
    A_up = building_another_Calderon_preconditioner(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp)
    def Calderon_preconditioner(v):
        num = len(X_diag)
        x = np.zeros(np.shape(v), dtype=np.complex128)
        v_up = v[0:num]
        v_down = v[num:(2 * num)]
        x[0:num] = A_up.dot(v_up)/2.
        x[num:(2 * num)] = A_sparse_times_2.dot(v_down)/2.
        return x
    Calderon_preconditioner_linear_operator=sp.linalg.LinearOperator((4 * N * L_plus_1_square, 4 * N * L_plus_1_square),
                                                   matvec=Calderon_preconditioner,
                                                   matmat=Calderon_preconditioner)
    solution, info = sp.linalg.gmres(MTF_linear_operator, b.toarray(), tol=tolerance,
                                     restart=4 * N * L_plus_1_square, M=Calderon_preconditioner_linear_operator, callback=callback_function,
                                     callback_type='pr_norm')
    return solution, info, np.asarray(norms)


def solving_second_kind_ColtonKress_system_with_gmres_version_0(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0, tolerance):
    second_kind_matrix = second_kind_ColtonKress_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                 almost_A_0)
    def second_kind_ColtonKress_matrix_times_vector(v):
        return np.matmul(second_kind_matrix, v)
    second_kind_ColtonKress_operator = sp.linalg.LinearOperator(np.shape(second_kind_matrix), matvec=second_kind_ColtonKress_matrix_times_vector)
    b_sparse, b_rows = second_kind_ColtonKress_b_building(N, a, p, k[0], j_l[:, :, 0], j_lp[:, :, 0], L,
                                                          L_plus_1_square)
    b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(2 * N * (L + 1), dtype=int))), dtype=np.complex128,
                      shape=(2 * N * L_plus_1_square, 1))
    del b_sparse
    del b_rows
    norms =[]
    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.
    solution, info = sp.linalg.gmres(second_kind_ColtonKress_operator, b.toarray(), tol=tolerance, restart=4 * N * L_plus_1_square, callback=callback_function, callback_type='pr_norm')
    return solution, info, np.asarray(norms)


#Computing the eigenvalues
# ---
def MTF_eigenvalues(r, p, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp, almost_A_0):
    A_0_times_2, X_diag, A_sparse_times_2 = MTF_matrix_building(r, a, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp,
                                                                almost_A_0)
    MTF_matrix = np.concatenate((np.concatenate(
        (A_0_times_2, sp.dia_matrix(((1. / X_diag), np.array([0])), shape=np.shape(A_0_times_2)).toarray()), axis=1),
                                 np.concatenate((sp.dia_matrix((X_diag, np.array([0])),
                                                               shape=np.shape(A_sparse_times_2)).toarray(),
                                                 A_sparse_times_2.toarray()), axis=1)), axis=0)
    del X_diag
    eigenvalues_MTF = np.linalg.eigvals(MTF_matrix)
    Calderon_preconditioner = np.concatenate(
        (np.concatenate((A_0_times_2, np.zeros(np.shape(A_0_times_2), dtype=np.complex128)), axis=1),
         np.concatenate((np.zeros(np.shape(A_0_times_2), dtype=np.complex128),
                         A_sparse_times_2.toarray()), axis=1)), axis=0)
    eigenvalues_preconditioned = np.linalg.eigvals(Calderon_preconditioner*MTF_matrix)
    return eigenvalues_MTF, eigenvalues_preconditioned


# Post-procesing of solutions
# ---
def traces_second_kind_formulation(second_kind_ColtonKress_solution, r, a, p, k_0, L, L_plus_1_square, N, k, j_l, j_lp, h_l, h_lp):
    interior_traces = np.zeros((2 * N * L_plus_1_square, 1), dtype=np.complex128)

    pre_V_0 = 1j * k[0]
    pre_V_int = 1j
    pre_K_0 = - pre_V_0 * k[0]
    pre_W_0 = pre_K_0 * k[0]

    common_factor = 2.
    r_square = np.square(r)

    super_L = np.arange(0, L + 1)
    rango = np.arange(0, L_plus_1_square)
    for s in np.arange(1, N + 1):
        s_minus_1 = s - 1
        s_minus_1_times_2 = s_minus_1 * 2

        Pre_V_s = pre_V_int * k[s] * r_square[s_minus_1]

        Pre_Ka_s = Pre_V_s * k[s]
        Pre_K_s = Pre_Ka_s
        Pre_W_s = Pre_K_s * k[s]

        V_s_p = np.multiply(Pre_V_s, np.multiply(j_l[:, s_minus_1, 1], h_l[:, s_minus_1, 1]))
        K_s_p_minus_half = np.multiply(Pre_K_s, np.multiply(j_l[:, s_minus_1, 1], h_lp[:, s_minus_1, 1]))

        Ka_s_p_plus_half = np.multiply(Pre_Ka_s, np.multiply(j_lp[:, s_minus_1, 1], h_l[:, s_minus_1, 1]))

        W_s_p = np.multiply(Pre_W_s, np.multiply(j_lp[:, s_minus_1, 1], h_lp[:, s_minus_1, 1]))

        interior_traces[(rango + L_plus_1_square * (s_minus_1_times_2)), 0] = interior_traces[(rango + L_plus_1_square * (s_minus_1_times_2)), 0] + np.repeat(V_s_p, (super_L * 2 + 1)) * second_kind_ColtonKress_solution[(L_plus_1_square * (1 + s_minus_1_times_2) + rango), 0]
        #rows_A_sparse[number:(number + num)] = rango + num * (s_minus_1_times_2)
        #columns_A_sparse[number:(number + num)] = num * (1 + s_minus_1_times_2) + rango

        interior_traces[(rango + L_plus_1_square * (s_minus_1_times_2)), 0] = interior_traces[(rango + L_plus_1_square * (s_minus_1_times_2)), 0] - np.repeat(K_s_p_minus_half, (super_L * 2 + 1)) * - second_kind_ColtonKress_solution[(L_plus_1_square * (s_minus_1_times_2) + rango), 0]
        #rows_A_sparse[number:(number + num)] = rango + num * (s_minus_1_times_2)
        #columns_A_sparse[number:(number + num)] = num * (s_minus_1_times_2) + rango

        interior_traces[(rango + L_plus_1_square * (s_minus_1_times_2 + 1)), 0] = interior_traces[(rango + L_plus_1_square * (s_minus_1_times_2 + 1)), 0] + np.repeat(Ka_s_p_plus_half, (super_L * 2 + 1)) * second_kind_ColtonKress_solution[(L_plus_1_square * (s_minus_1_times_2 + 1) + rango), 0]
        #rows_A_sparse[number:(number + num)] = rango + num * (s_minus_1_times_2 + 1)
        #columns_A_sparse[number:(number + num)] = num * (s_minus_1_times_2 + 1) + rango

        interior_traces[(rango + L_plus_1_square * (s_minus_1_times_2 + 1)), 0] = interior_traces[(rango + L_plus_1_square * (s_minus_1_times_2 + 1)), 0] + np.repeat(W_s_p, (super_L * 2 + 1)) * - second_kind_ColtonKress_solution[(L_plus_1_square * (s_minus_1_times_2) + rango), 0]
        #rows_A_sparse[number:(number + num)] = rango + num * (s_minus_1_times_2 + 1)
        #columns_A_sparse[number:(number + num)] = num * (s_minus_1_times_2) + rango
    del rango
    del super_L
    del r_square
    b_sparse, b_rows = plane_wave_traces(N, p, k_0, j_l[:, :, 0], j_lp[:, :, 0], L, L_plus_1_square)
    b = sp.coo_matrix((b_sparse, (b_rows, np.zeros(2 * N * (L + 1), dtype=int))), dtype=np.complex128,
                      shape=(2 * N * L_plus_1_square, 1))
    del b_sparse
    del b_rows
    exterior_traces = b.toarray() - interior_traces
    exterior_traces[0: (N*L_plus_1_square), 0] = exterior_traces[0: (N*L_plus_1_square), 0] * -1.
    traces = np.concatenate((exterior_traces, interior_traces), axis=0)
    return traces #traces[:, 0]


# Setting positions routines
# ---
def cube_vertex_positions(number, r, d):
    # To obtain an arrangement of number**3 spheres.
    ps = []
    p = np.array([0., 0., 0.])
    two_r_plus_d = 2 * r + d
    for ii in np.arange(0, number):
        p[1] = 0.
        p = p + np.array([0., 0., two_r_plus_d * ii])
        for jj in np.arange(0, number):
            p[0] = 0.
            p = p + np.array([0., two_r_plus_d * jj, 0.])
            for zz in np.arange(0, number):
                ps.append(p + np.array([two_r_plus_d * zz, 0., 0.]))
                p = p + np.array([two_r_plus_d * zz, 0., 0.])
    return ps


def cube_vertex_positions_and_center(numero, r):
    # To obtain an arrangement of de number**3 + (number-1)**3 spheres.
    ps = []
    p = np.array([0., 0., 0.])
    three_r = 3 * r
    for ii in np.arange(0, numero):
        p[1] = 0.
        p = p + np.array([0., 0., three_r * ii])
        for jj in np.arange(0, numero):
            p[0] = 0.
            p = p + np.array([0., three_r * jj, 0.])
            for zz in np.arange(0, numero):
                ps.append(p + np.array([three_r * zz, 0., 0.]))
                p = p + np.array([three_r * zz, 0., 0.])
    p = np.array([three_r / 2., three_r / 2., three_r / 2.])
    for ii in np.arange(0, numero - 1):
        p[1] = three_r / 2.
        p = p + np.array([0., 0., three_r * ii])
        for jj in np.arange(0, numero - 1):
            p[0] = three_r / 2.
            p = p + np.array([0., three_r * jj, 0.])
            for zz in np.arange(0, numero - 1):
                ps.append(p + np.array([three_r * zz, 0., 0.]))
                p = p + np.array([three_r * zz, 0., 0.])
    return ps

# Norms and error routines.
#---
def checking_discrete_Calderon_ext_int_sparse_matrices(solution, L, N, A_0_times_2, A_sparse_times_2):
    identity_matrix = sp.dia_matrix((np.ones(np.shape(A_0_times_2)[0]), np.array([0])), shape=np.shape(A_0_times_2))
    up = (A_0_times_2 - identity_matrix).dot(solution[0:2 * N * (L + 1) ** 2])
    norm_up = np.linalg.norm(up)
    del up
    Int = (A_sparse_times_2 - identity_matrix).dot(solution[2 * N * (L + 1) ** 2:4 * N * (L + 1) ** 2])
    return np.asarray([norm_up, np.linalg.norm(Int)])


def checking_discrete_Calderon_ext_sparse_matrices(solution, L, N, A_0_times_2):
    identity_matrix = sp.dia_matrix((np.ones(np.shape(A_0_times_2)[0]), np.array([0])), shape=np.shape(A_0_times_2))
    up = (A_0_times_2 - identity_matrix).dot(solution[0:2 * N * (L + 1) ** 2])
    return np.linalg.norm(up)

# Pendiente de terminar
def jump_error_MTF(M, solution, b, k_0, a, r, p, N, L, L_overkill, j_l_0, j_lp_0,):
    M = M.tolil()

    MenosXAlaMenos1 = M[0:2 * N * (L + 1) ** 2, 2 * N * (L + 1) ** 2:4 * N * (L + 1) ** 2]
    del M
    MenosXAlaMenos1 = MenosXAlaMenos1.todia()
    VectorExt = solution[0:2 * N * (L + 1) ** 2]
    VectorInt = solution[2 * N * (L + 1) ** 2:4 * N * (L + 1) ** 2]
    ErrorPrevio = 0.
    b = b[0:2 * N * (L + 1) ** 2]
    ErrorPrevio = np.linalg.norm(VectorExt + np.matmul(MenosXAlaMenos1, VectorInt) - b) ** 2
    print(ErrorPrevio)
    del VectorExt
    del VectorInt
    del MenosXAlaMenos1

    b_overkill, b_rows_overkill = MTF_b_building_plane_wave(N, a, p, k_0, j_l_0, j_lp_0, L_overkill, (L_overkill + 1) ** 2)

    b_overkill = sp.coo_matrix((b_overkill, (b_rows_overkill, np.zeros(4 * N * (L_overkill + 1), dtype=int))), dtype=np.complex128,
                               shape=(4 * N * (L_overkill + 1) ** 2, 1))
    del b_rows_overkill

    b_overkill = b_overkill.tolil()

    RestoDelError = 0.
    aux = (L_overkill + 1) ** 2
    aux2 = (L + 1) ** 2
    for i in np.arange(0, 2 * n):
        RestoDelError = RestoDelError + splg.norm(b_overkill[i * (aux) + aux2:(i + 1) * aux]) ** 2
    ErrorPrevio = ErrorPrevio + RestoDelError
    return np.sqrt(ErrorPrevio)


def L2_norm_on_surface(coefficients, r, bool_with_r):
    suma = np.linalg.norm(coefficients)
    if bool_with_r:
        return r * suma
    else:
        return suma


# For checking results with phantom spheres
# ---
def dirichlet_trace_result_minus_u_inc_norm(coefficients, radius, position, L, k_0):
    sum = 0.
    counter = 0
    extra = 0.
    j_l = sci.spherical_jn(np.arange(0, L + 1), radius * k_0)
    for l in np.arange(0, L + 1):
        for m in np.arange(-l, l + 1):
            if m != 0:
                sum = sum + np.linalg.norm(coefficients[counter]) ** 2
                extra = extra + np.linalg.norm(coefficients[counter]) ** 2
            else:
                sum = sum + np.linalg.norm(
                    coefficients[counter] - 2 * np.sqrt(np.pi) * np.exp(1j * k_0 * position[2]) * 1j ** l * np.sqrt(
                        2 * l + 1) * j_l[l]) ** 2
            counter = counter + 1
    print('Index checking. It should be zero: ' + str(len(coefficients) - counter))
    print('(Extra root u_inc) ' + str(np.sqrt(extra) * radius))
    return np.sqrt(sum) * radius


def neumann_trace_result_minus_u_inc_norm(coefficients, radius, position, L, k_0):
    sum = 0.
    counter = 0
    j_lp = sci.spherical_jn(np.arange(0, L + 1), radius * k_0, derivative=True)
    extra = 0.
    for l in np.arange(0, L + 1):
        for m in np.arange(-l, l + 1):
            if m != 0:
                sum = sum + np.linalg.norm(coefficients[counter]) ** 2
                extra = extra + np.linalg.norm(coefficients[counter]) ** 2
            else:
                sum = sum + np.linalg.norm(
                    coefficients[counter] - 2 * k_0 * np.sqrt(np.pi) * np.exp(1j * k_0 * position[2]) * 1j ** l * np.sqrt(
                        2 * l + 1) * j_lp[l]) ** 2
            counter = counter + 1
    print('Index checking. It should be zero: ' + str(len(coefficients) - counter))
    print('(Extra root u_inc) ' + str(np.sqrt(extra) * radius))
    return np.sqrt(sum) * radius


# For checking results with all spheres being phantom except one.
def checking_with_analytic_result(L, coefficients, k_0, k_j, a, p_j, r_j, which_trace, overkill):
    r_j_times_k_0 = r_j * k_0
    r_j_times_k_1 = r_j * k_j

    rango = np.arange(0, overkill + 1)

    j_l_0 = sci.spherical_jn(rango, r_j_times_k_0)
    j_lp_0 = sci.spherical_jn(rango, r_j_times_k_0, derivative=True)

    j_l_j = sci.spherical_jn(rango, r_j_times_k_1)
    j_lp_j = sci.spherical_jn(rango, r_j_times_k_1, derivative=True)

    h_l_0 = j_l_0 + 1j * sci.spherical_yn(rango, r_j_times_k_0)
    h_lp_0 = j_lp_0 + 1j * sci.spherical_yn(rango, r_j_times_k_0, derivative=True)

    sum = -1.
    aux = 1.
    if which_trace == 0:
        theo_lambda_0 = np.exp(1j * k_0 * p_j[2]) * 2 * 1j ** rango * np.sqrt(np.pi * (2 * rango + 1)) * h_l_0[rango] * (
                -a * j_lp_0[rango] * j_l_j[rango] * k_0 + j_l_0[rango] * j_lp_j[rango] * k_j) / (
                a * h_lp_0[rango] * j_l_j[rango] * k_0 - h_l_0[rango] * j_lp_j[rango] * k_j)
        aux = np.linalg.norm(theo_lambda_0) ** 2
        sum = 0.
        contador = 0
        for l in np.arange(0, L + 1):
            for m in np.arange(-l, l + 1):
                if m != 0:
                    sum = sum + np.linalg.norm(coefficients[contador]) ** 2
                else:
                    sum = sum + np.linalg.norm(coefficients[contador] - theo_lambda_0[l]) ** 2
                contador = contador + 1
        sum = sum + np.linalg.norm(theo_lambda_0[(L + 1):overkill]) ** 2
        '''
        print '\n Norma**2 del resultado anal\'itico' +str(aux)
        print 'El relativo es '+str(np.sqrt(sum/aux))
        print 'El absoluto es' +str(np.sqrt(sum))
        '''
    elif which_trace == 1:
        theo_lambda_1 = -np.exp(1j * k_0 * p_j[2]) * 2 * k_0 * 1j ** rango * np.sqrt(np.pi * (2 * rango + 1)) * h_lp_0[
            rango] * (-a * j_lp_0[rango] * j_l_j[rango] * k_0 + j_l_0[rango] * j_lp_j[rango] * k_j) / (
                                a * h_lp_0[rango] * j_l_j[rango] * k_0 - h_l_0[rango] * j_lp_j[rango] * k_j)
        aux = np.linalg.norm(theo_lambda_1) ** 2
        sum = 0.
        contador = 0
        for l in np.arange(0, L + 1):
            for m in np.arange(-l, l + 1):
                if m != 0:
                    sum = sum + np.linalg.norm(coefficients[contador]) ** 2
                else:
                    sum = sum + np.linalg.norm(coefficients[contador] - theo_lambda_1[l]) ** 2
                contador = contador + 1
        sum = sum + np.linalg.norm(theo_lambda_1[(L + 1):overkill]) ** 2
        '''
        print '\n Norma**2 del resultado anal\'itico' +str(aux)
        print 'El relativo es '+str(np.sqrt(sum/aux))
        print 'El absoluto es' +str(np.sqrt(sum))
        '''
    elif which_trace == 2:
        theo_lambda_2 = np.exp(1j * k_0 * p_j[2]) * 2 * a * k_0 * 1j ** rango * np.sqrt(np.pi * (2 * rango + 1)) * j_l_j[
            rango] * (-h_l_0[rango] * j_lp_0[rango] + h_lp_0[rango] * j_l_0[rango]) / (
                                a * h_lp_0[rango] * j_l_j[rango] * k_0 - h_l_0[rango] * j_lp_j[rango] * k_j)
        aux = np.linalg.norm(theo_lambda_2) ** 2
        sum = 0.
        contador = 0
        for l in np.arange(0, L + 1):
            for m in np.arange(-l, l + 1):
                if m != 0:
                    sum = sum + np.linalg.norm(coefficients[contador]) ** 2
                else:
                    sum = sum + np.linalg.norm(coefficients[contador] - theo_lambda_2[l]) ** 2
                contador = contador + 1
        sum = sum + np.linalg.norm(theo_lambda_2[(L + 1):overkill]) ** 2
        '''
        print '\n Norma**2 del resultado anal\'itico' +str(aux)
        print 'El relativo es '+str(np.sqrt(sum/aux))
        print 'El absoluto es' +str(np.sqrt(sum))
        '''
    elif which_trace == 3:
        theo_lambda_3 = np.exp(1j * k_0 * p_j[2]) * 2 * a * k_0 * k_j * 1j ** rango * np.sqrt(np.pi * (2 * rango + 1)) * j_lp_j[
            rango] * (-h_l_0[rango] * j_lp_0[rango] + h_lp_0[rango] * j_l_0[rango]) / (
                                a * h_lp_0[rango] * j_l_j[rango] * k_0 - h_l_0[rango] * j_lp_j[rango] * k_j)
        aux = np.linalg.norm(theo_lambda_3) ** 2
        sum = 0.
        contador = 0
        for l in np.arange(0, L + 1):
            for m in np.arange(-l, l + 1):
                if m != 0:
                    sum = sum + np.linalg.norm(coefficients[contador]) ** 2
                else:
                    sum = sum + np.linalg.norm(coefficients[contador] - theo_lambda_3[l]) ** 2
                contador = contador + 1
        sum = sum + np.linalg.norm(theo_lambda_3[(L + 1):overkill]) ** 2
        '''
        print '\n Norma**2 del resultado anal\'itico' +str(aux)
        print 'El relativo es '+str(np.sqrt(sum/aux))
        print 'El absoluto es' +str(np.sqrt(sum))
        '''
    return np.sqrt(sum)


def ErrorVsLOverkillFantasma(n, L, fantasma, nombreM, nombreb, k, aes, ps, rs):
    M = sp.load_npz(nombreM + '.npz').tolil()
    b = sp.load_npz(nombreb + '.npz').tolil()
    ErrorAnaliticoExteriorDirichlet = np.zeros((L + 1))
    ErrorAnaliticoExteriorNeumann = np.zeros((L + 1))
    ErrorAnaliticoInteriorDirichlet = np.zeros((L + 1))
    ErrorAnaliticoInteriorNeumann = np.zeros((L + 1))
    ErrorCalderon = np.zeros((L + 1, 2))
    ErrorSal = np.zeros((L + 1))
    for l in np.arange(0, L + 1):
        Maux1 = M[0:(l + 1) ** 2, 0:(l + 1) ** 2].toarray()
        jj = 0
        baux = b[0:(l + 1) ** 2, 0].toarray()
        for ii in np.arange(1, 4 * n):
            Maux1 = np.concatenate((Maux1, M[ii * (L + 1) ** 2:ii * (L + 1) ** 2 + (l + 1) ** 2,
                                           jj * (L + 1) ** 2:jj * (L + 1) ** 2 + (l + 1) ** 2].toarray()), axis=0)
            baux = np.concatenate((baux, b[ii * (L + 1) ** 2:ii * (L + 1) ** 2 + (l + 1) ** 2, 0].toarray()), axis=0)
        for jj in np.arange(1, 4 * n):
            Maux2 = M[0:(l + 1) ** 2, jj * (L + 1) ** 2:jj * (L + 1) ** 2 + (l + 1) ** 2].toarray()
            for ii in np.arange(1, 4 * n):
                Maux2 = np.concatenate((Maux2, M[ii * (L + 1) ** 2:ii * (L + 1) ** 2 + (l + 1) ** 2,
                                               jj * (L + 1) ** 2:jj * (L + 1) ** 2 + (l + 1) ** 2].toarray()), axis=0)
            Maux1 = np.concatenate((Maux1, Maux2), axis=1)
        solucionaux = np.linalg.solve(Maux1, baux)
        j = fantasma
        overkill = L
        ErrorAnaliticoExteriorDirichlet[l] = checking_with_analytic_result(l, solucionaux[
                                                                              2 * (l + 1) ** 2 * (j - 1):2 * (
                                                                                          l + 1) ** 2 * (
                                                                                                                 j - 1) + (
                                                                                                                     l + 1) ** 2],
                                                                           k[0], k[j], aes[j - 1], ps[j - 1], rs[j - 1],
                                                                           0, overkill)
        ErrorAnaliticoExteriorNeumann[l] = checking_with_analytic_result(l, solucionaux[
                                                                            2 * (l + 1) ** 2 * (j - 1) + (
                                                                                        l + 1) ** 2:2 * (
                                                                                    l + 1) ** 2 * j], k[0], k[j],
                                                                         aes[j - 1], ps[j - 1], rs[j - 1], 1, overkill)
        ErrorAnaliticoInteriorDirichlet[l] = checking_with_analytic_result(l, solucionaux[
                                                                              2 * n * (l + 1) ** 2 + 2 * (
                                                                                          l + 1) ** 2 * (
                                                                                      j - 1):2 * n * (
                                                                                          l + 1) ** 2 + 2 * (
                                                                                                     l + 1) ** 2 * (
                                                                                                         j - 1) + (
                                                                                                         l + 1) ** 2],
                                                                           k[0], k[j], aes[j - 1], ps[j - 1], rs[j - 1],
                                                                           2, overkill)
        ErrorAnaliticoInteriorNeumann[l] = checking_with_analytic_result(l, solucionaux[
                                                                            2 * n * (l + 1) ** 2 + 2 * (l + 1) ** 2 * (
                                                                                        j - 1) + (
                                                                                    l + 1) ** 2:2 * n * (
                                                                                        l + 1) ** 2 + 2 * (
                                                                                                        l + 1) ** 2 * j],
                                                                         k[0], k[j], aes[j - 1], ps[j - 1], rs[j - 1],
                                                                         3, overkill)
        ErrorCalderon[l, :] = checking_discrete_Calderon_MTF_sparse_matrices(solucionaux, l, n, Maux1, )
        ErrorSal[l] = jump_error_MTF(Maux1, solucionaux, baux, k, aes, rs, ps, l, L)
    return [ErrorAnaliticoExteriorDirichlet, ErrorAnaliticoExteriorNeumann, ErrorAnaliticoInteriorDirichlet,
            ErrorAnaliticoInteriorNeumann, ErrorCalderon, ErrorSal]


def ErrorVsLOverkill(n, L, nombreM, nombreb, k, aes, ps, rs):
    M = sp.load_npz(nombreM + '.npz').tolil()
    b = sp.load_npz(nombreb + '.npz').tolil()
    ErrorCalderon = np.zeros((L + 1, 2))
    ErrorSal = np.zeros((L + 1))
    for l in np.arange(0, L + 1):
        Maux1 = M[0:(l + 1) ** 2, 0:(l + 1) ** 2].toarray()
        jj = 0
        baux = b[0:(l + 1) ** 2, 0].toarray()
        for ii in np.arange(1, 4 * n):
            Maux1 = np.concatenate((Maux1, M[ii * (L + 1) ** 2:ii * (L + 1) ** 2 + (l + 1) ** 2,
                                           jj * (L + 1) ** 2:jj * (L + 1) ** 2 + (l + 1) ** 2].toarray()), axis=0)
            baux = np.concatenate((baux, b[ii * (L + 1) ** 2:ii * (L + 1) ** 2 + (l + 1) ** 2, 0].toarray()), axis=0)
        for jj in np.arange(1, 4 * n):
            Maux2 = M[0:(l + 1) ** 2, jj * (L + 1) ** 2:jj * (L + 1) ** 2 + (l + 1) ** 2].toarray()
            for ii in np.arange(1, 4 * n):
                Maux2 = np.concatenate((Maux2, M[ii * (L + 1) ** 2:ii * (L + 1) ** 2 + (l + 1) ** 2,
                                               jj * (L + 1) ** 2:jj * (L + 1) ** 2 + (l + 1) ** 2].toarray()), axis=0)
            Maux1 = np.concatenate((Maux1, Maux2), axis=1)
        solucionaux = np.linalg.solve(Maux1, baux)
        Maux1 = sp.lil_matrix(Maux1)
        baux = sp.lil_matrix(baux)
        ErrorCalderon[l, :] = checking_discrete_Calderon_MTF_sparse_matrices(solucionaux, l, n, Maux1, )
        ErrorSal[l] = jump_error_MTF(Maux1, solucionaux, baux, k, aes, rs, ps, l, L)
        Maux1 = Maux1.toarray()
        baux = baux.toarray()
    return [ErrorCalderon, ErrorSal]


def ErrorVsLOverkillVersion2(n, L, nombreM, nombreb, solucion, k, aes, ps, rs):
    M = sp.load_npz(nombreM + '.npz').tolil()
    b = sp.load_npz(nombreb + '.npz').tolil()
    ErrorCalderon = np.zeros((L, 2))
    ErrorSal = np.zeros((L))
    ErrorLoL = np.zeros((L, 2))
    for l in np.arange(0, L):
        Maux1 = M[0:(l + 1) ** 2, 0:(l + 1) ** 2].toarray()
        jj = 0
        baux = b[0:(l + 1) ** 2, 0].toarray()
        for ii in np.arange(1, 4 * n):
            Maux1 = np.concatenate((Maux1, M[ii * (L + 1) ** 2:ii * (L + 1) ** 2 + (l + 1) ** 2,
                                           jj * (L + 1) ** 2:jj * (L + 1) ** 2 + (l + 1) ** 2].toarray()), axis=0)
            baux = np.concatenate((baux, b[ii * (L + 1) ** 2:ii * (L + 1) ** 2 + (l + 1) ** 2, 0].toarray()), axis=0)
        for jj in np.arange(1, 4 * n):
            Maux2 = M[0:(l + 1) ** 2, jj * (L + 1) ** 2:jj * (L + 1) ** 2 + (l + 1) ** 2].toarray()
            for ii in np.arange(1, 4 * n):
                Maux2 = np.concatenate((Maux2, M[ii * (L + 1) ** 2:ii * (L + 1) ** 2 + (l + 1) ** 2,
                                               jj * (L + 1) ** 2:jj * (L + 1) ** 2 + (l + 1) ** 2].toarray()), axis=0)
            Maux1 = np.concatenate((Maux1, Maux2), axis=1)
        solucionaux = np.linalg.solve(Maux1, baux)
        Maux1 = sp.lil_matrix(Maux1)
        baux = sp.lil_matrix(baux)
        ErrorCalderon[l, :] = checking_discrete_Calderon_MTF_sparse_matrices(solucionaux, l, n, Maux1, )
        ErrorSal[l] = jump_error_MTF(Maux1, solucionaux, baux, k, aes, rs, ps, l, L)
        ErrorLoL[l, :] = Error2trazas(solucionaux, solucion, L, l, n)
        Maux1 = Maux1.toarray()
        baux = baux.toarray()
    return [ErrorCalderon, ErrorSal, ErrorLoL]


def Error2trazas(solucionaux, solucion, L, l, n):
    Error = 0.
    ErrorResto = 0.
    for ini in np.arange(0, 4 * n):
        Error = Error + np.linalg.norm(solucionaux[ini * (l + 1) ** 2:(ini * (l + 1) ** 2 + (l + 1) ** 2)] - solucion[
                                                                                                             ini * (
                                                                                                                     L + 1) ** 2:(
                                                                                                                     ini * (
                                                                                                                     L + 1) ** 2 + (
                                                                                                                             l + 1) ** 2)]) ** 2
        ErrorResto = Error + np.linalg.norm(
            solucion[(ini * (L + 1) ** 2 + (l + 1) ** 2):((ini + 1) * (L + 1) ** 2)]) ** 2
    return np.asarray(
        [np.sqrt(Error) / np.linalg.norm(solucion), np.sqrt(Error + ErrorResto) / np.linalg.norm(solucion)])


def ErrorL(n, L, nombreM, nombreb, solucion):
    M = sp.load_npz(nombreM + '.npz').tolil()
    b = sp.load_npz(nombreb + '.npz').tolil()
    ErrorLoL = np.zeros((L, 2))
    for l in np.arange(0, L):
        Maux1 = M[0:(l + 1) ** 2, 0:(l + 1) ** 2].toarray()
        jj = 0
        baux = b[0:(l + 1) ** 2, 0].toarray()
        for ii in np.arange(1, 4 * n):
            Maux1 = np.concatenate((Maux1, M[ii * (L + 1) ** 2:ii * (L + 1) ** 2 + (l + 1) ** 2,
                                           jj * (L + 1) ** 2:jj * (L + 1) ** 2 + (l + 1) ** 2].toarray()), axis=0)
            baux = np.concatenate((baux, b[ii * (L + 1) ** 2:ii * (L + 1) ** 2 + (l + 1) ** 2, 0].toarray()), axis=0)
        for jj in np.arange(1, 4 * n):
            Maux2 = M[0:(l + 1) ** 2, jj * (L + 1) ** 2:jj * (L + 1) ** 2 + (l + 1) ** 2].toarray()
            for ii in np.arange(1, 4 * n):
                Maux2 = np.concatenate((Maux2, M[ii * (L + 1) ** 2:ii * (L + 1) ** 2 + (l + 1) ** 2,
                                               jj * (L + 1) ** 2:jj * (L + 1) ** 2 + (l + 1) ** 2].toarray()), axis=0)
            Maux1 = np.concatenate((Maux1, Maux2), axis=1)
        solucionaux = np.linalg.solve(Maux1, baux)
        Maux1 = sp.lil_matrix(Maux1)
        baux = sp.lil_matrix(baux)
        ErrorLoL[l, :] = Error2trazas(solucionaux, solucion, L, l, n)
        Maux1 = Maux1.toarray()
        baux = baux.toarray()
    return ErrorLoL


###Fin de las rutinas de normas de trazas y c\'alculos de errores y  restas de trazas, etc.

##Comienzo de rutinas de dibujo----------------------------------------------------------------------------

def TerceraVersionDeDibujo(dd, centro, ancho, alto, interancho, interlargo, Resultado, t, omega, c, rs, rho, ps, L):
    # Funci\'on para dibujar el campo escalar. Function for drawing the scalar field

    # Inputs:
    ##dd: int. Indica si el dibujo se va a hacer en un corte paralelo al plano:
    # xy si dd==1, xz si dd==2, yz si dd=3. Indicates if the drawing is a
    # paralel cut of the plane: xy if dd==1, xz if dd==2, yz if dd==3.
    ##centro: array de floats de largo 3. Coordenadas del centro del dibujo.
    # Array of floats of length 3. Coordinates of the center of the drawing.
    ##ancho: float. Ancho del rectangulo a dibujar. Wide of the rectangle that
    # is going to be drawn.
    ##alto: float. Alto del rectangulo a dibujar. High of the rectangle that it
    # is going to be drawn.
    ##interancho: int. Cantidad de puntos que se calcular\'an a lo ancho.
    # Quantity of points that there are going to be calculate along the wide of
    # the drawing.
    ##interlargo: int. Cantidad de puntos que se calcular\'an a lo largo.
    # Quantity of points that there are going to be calculate along the high of
    # the drawing.
    ##Resultado: output solucion de la funci\'on ResolverTodo. output solucion
    # of the function ResolverTodo
    # t: float, tiempo que se va a dibujar. Time that it is going to be drawn.
    ##omega: float o complex. Es la frecuencia de la onda incidente. (float or
    # complex. It's the frequency of the incident wave).
    ##c: array de floats. Array de la rapidez del sonido de cada medio.
    # (Array of floats. Array of the sound speed in each medium).
    ##rs: array de nx1 de floats. Radio de las esferas. (Array of nx1 of
    # floats. Sphere radius).
    ##rho: array de floats. Array de las densidades de cada medio. (Array of
    # floats. Array of the density of each medium).
    ##ps: posiciones del centro de las esferas.
    ##L: int. Orden m\'aximo de los arm\'nicos esf\'ericos que se utilizan para
    # discretizar las trazas. (L+1)**2 es la cantidad total de arm\'onicos
    # esf\'ericos que se utilizan para discretizar una traza. (Maximum order of
    # spherical harmonics that are used to discretize the traces. (L+1)**2 is
    # the total number of spherical harmonics that are used to discretize a
    # trace).
    ##paralelo: int. N\'umero de procesadores que se utilizar\'an para
    # paralelizar el proceso.

    x1 = np.linspace(-ancho / 2, ancho / 2, interancho)
    y1 = np.linspace(-alto / 2, alto / 2, interlargo)
    pgraf = np.zeros((len(y1), len(x1)))

    [n, k, aes] = DefinicionDeAlgunosParametros(omega, c, rho)

    # Radio al cuadrado. Este es para ahorrar operaciones en el futuro.
    rcuad = rs ** 2

    [jn, jnp, hn, hnp] = PreCalculo(n, k, rs, L)
    LMas1 = L + 1
    rango = np.arange(0, LMas1)
    if dd == 3:
        for ii in np.arange(0, len(x1)):
            x = x1[ii] + centro[0]
            for jj in np.arange(0, len(y1)):
                theta = 0.0
                y = y1[jj] + centro[1]
                z = 0. + centro[2]

                xvectorcart = np.asarray([x, y, z])
                Esfera = 0
                num = 0

                while num < n:
                    aux = xvectorcart - ps[num]
                    raux = np.linalg.norm(aux)
                    if raux < rs[num]:
                        Esfera = num + 1
                        num = n
                    num = num + 1

                u = np.complex128(0.)

                if Esfera == 0:
                    for num in np.arange(0, n):
                        aux = xvectorcart - ps[num]
                        r = np.linalg.norm(aux)
                        x = aux[0]
                        y = aux[1]
                        z = aux[2]

                        theta = np.arccos(z / r)
                        phi = np.arctan2(y, x)

                        haux = sci.spherical_jn(rango, r * k[0]) + 1j * sci.spherical_yn(rango, r * k[0])
                        contador = 0
                        utemp = np.complex128(0.)

                        PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                        ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                        ExpNeg = np.divide(1., ExpPos)
                        SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                        for l in rango:
                            lcuadradomasl = l * (l + 1)
                            lcuadradomasldivididoendos = lcuadradomasl / 2
                            SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                            temp = np.complex128(0.)
                            for m in np.arange(1, l + 1):
                                Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar, ExpNeg[m - 1])
                            for m in np.arange(-l, l + 1):
                                temp = temp + (Resultado[contador + 2 * (L + 1) ** 2 * num] * k[0] * jnp[l, num, 0] +
                                               Resultado[contador + (L + 1) ** 2 * (2 * num + 1)] * jn[l, num, 0]) * \
                                       SuperEsfericos[contador]
                                contador = contador + 1
                            temp = haux[l] * temp
                            utemp = utemp + temp
                        del PolinomiosLegendre
                        utemp = rcuad[num] * utemp
                        u = u + utemp
                    # print u
                    auxiliar = np.exp(1j * k[0] * xvectorcart[2] - 1j * omega * t) + 1j * k[0] * u * np.exp(
                        -1j * omega * t)
                    pgrafReal[jj, ii] = np.real(auxiliar)
                    pgrafImag[jj, ii] = np.imag(auxiliar)
                else:
                    aux = xvectorcart - ps[Esfera - 1]
                    r = np.linalg.norm(aux)
                    x = aux[0]
                    y = aux[1]
                    z = aux[2]

                    theta = np.arccos(z / r)

                    phi = np.arctan2(y, x)
                    jaux = sci.spherical_jn(rango, r * k[Esfera])

                    contador = 0
                    PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                    ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                    ExpNeg = np.divide(1., ExpPos)
                    SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                    for l in rango:
                        lcuadradomasl = l * (l + 1)
                        lcuadradomasldivididoendos = lcuadradomasl / 2
                        SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                        temp = np.complex128(0.)
                        for m in np.arange(1, l + 1):
                            Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                            SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                            SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar, ExpNeg[m - 1])
                        for m in np.arange(-l, l + 1):
                            temp = temp + (-Resultado[contador + 2 * (L + 1) ** 2 * (Esfera - 1 + n)] * k[Esfera] * hnp[
                                l, Esfera - 1, 1] + Resultado[
                                               contador + (L + 1) ** 2 * (2 * (Esfera - 1) + 1 + 2 * n)] * hn[
                                               l, Esfera - 1, 1]) * SuperEsfericos[contador]
                            contador = contador + 1
                        temp = jaux[l] * temp
                        u = u + temp
                    auxiliar = 1j * k[Esfera] * rcuad[Esfera - 1] * u * np.exp(-1j * omega * t)
                    pgrafReal[jj, ii] = np.real(auxiliar)
                    pgrafImag[jj, ii] = np.imag(auxiliar)
        return pgrafReal + 1j * pgrafImag
    elif dd == 2:
        with pymp.Parallel(paralelo) as pp:
            for ii in pp.xrange(0, len(x1)):
                for jj in np.arange(0, len(y1)):
                    x = x1[ii]
                    y = 0.
                    z = y1[jj]
                    xvectorcart = np.asarray([x, y, z]) + centro
                    Esfera = 0
                    num = 0

                    while num < n:
                        aux = xvectorcart - ps[num]
                        raux = np.linalg.norm(aux)
                        if raux < rs[num]:
                            Esfera = num + 1
                            num = n
                        num = num + 1

                    u = np.complex128(0.)

                    if Esfera == 0:
                        for num in np.arange(0, n):
                            aux = xvectorcart - ps[num]
                            r = np.linalg.norm(aux)
                            x = aux[0]
                            y = aux[1]
                            z = aux[2]

                            theta = np.arccos(z / r)
                            phi = np.arctan2(y, x)

                            haux = sci.spherical_jn(rango, r * k[0]) + 1j * sci.spherical_yn(rango, r * k[0])
                            contador = 0
                            utemp = np.complex128(0.)

                            PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                            ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                            ExpNeg = np.divide(1., ExpPos)
                            SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                            for l in rango:
                                lcuadradomasl = l * (l + 1)
                                lcuadradomasldivididoendos = lcuadradomasl / 2
                                SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                                temp = np.complex128(0.)
                                for m in np.arange(1, l + 1):
                                    Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                    SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                    SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar,
                                                                                                  ExpNeg[m - 1])
                                for m in np.arange(-l, l + 1):
                                    temp = temp + (
                                                Resultado[contador + 2 * (L + 1) ** 2 * num] * k[0] * jnp[l, num, 0] +
                                                Resultado[contador + (L + 1) ** 2 * (2 * num + 1)] * jn[l, num, 0]) * \
                                           SuperEsfericos[contador]
                                    contador = contador + 1
                                temp = haux[l] * temp
                                utemp = utemp + temp
                            del PolinomiosLegendre
                            utemp = rcuad[num] * utemp
                            u = u + utemp
                        # print u
                        auxiliar = np.exp(1j * k[0] * xvectorcart[2] - 1j * omega * t) + 1j * k[0] * u * np.exp(
                            -1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
                    else:
                        aux = xvectorcart - ps[Esfera - 1]
                        r = np.linalg.norm(aux)
                        x = aux[0]
                        y = aux[1]
                        z = aux[2]

                        theta = np.arccos(z / r)

                        phi = np.arctan2(y, x)
                        jaux = sci.spherical_jn(rango, r * k[Esfera])

                        contador = 0
                        PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                        ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                        ExpNeg = np.divide(1., ExpPos)
                        SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                        for l in rango:
                            lcuadradomasl = l * (l + 1)
                            lcuadradomasldivididoendos = lcuadradomasl / 2
                            SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                            temp = np.complex128(0.)
                            for m in np.arange(1, l + 1):
                                Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar, ExpNeg[m - 1])
                            for m in np.arange(-l, l + 1):
                                temp = temp + (-Resultado[contador + 2 * (L + 1) ** 2 * (Esfera - 1 + n)] * k[Esfera] *
                                               hnp[l, Esfera - 1, 1] + Resultado[
                                                   contador + (L + 1) ** 2 * (2 * (Esfera - 1) + 1 + 2 * n)] * hn[
                                                   l, Esfera - 1, 1]) * SuperEsfericos[contador]
                                contador = contador + 1
                            temp = jaux[l] * temp
                            u = u + temp
                        auxiliar = 1j * k[Esfera] * rcuad[Esfera - 1] * u * np.exp(-1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
        return pgrafReal + 1j * pgrafImag
    else:
        with pymp.Parallel(paralelo) as pp:
            for ii in pp.xrange(0, len(x1)):
                for jj in np.arange(0, len(y1)):
                    theta = 0.0
                    x = 0.
                    y = x1[ii]
                    z = y1[jj]
                    xvectorcart = np.asarray([x, y, z]) + centro
                    Esfera = 0
                    num = 0

                    while num < n:
                        aux = xvectorcart - ps[num]
                        raux = np.linalg.norm(aux)
                        if raux < rs[num]:
                            Esfera = num + 1
                            num = n
                        num = num + 1

                    u = np.complex128(0.)

                    if Esfera == 0:
                        for num in np.arange(0, n):
                            aux = xvectorcart - ps[num]
                            r = np.linalg.norm(aux)
                            x = aux[0]
                            y = aux[1]
                            z = aux[2]

                            theta = np.arccos(z / r)
                            phi = np.arctan2(y, x)

                            haux = sci.spherical_jn(rango, r * k[0]) + 1j * sci.spherical_yn(rango, r * k[0])
                            contador = 0
                            utemp = np.complex128(0.)

                            PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                            ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                            ExpNeg = np.divide(1., ExpPos)
                            SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                            for l in rango:
                                lcuadradomasl = l * (l + 1)
                                lcuadradomasldivididoendos = lcuadradomasl / 2
                                SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                                temp = np.complex128(0.)
                                for m in np.arange(1, l + 1):
                                    Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                    SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                    SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar,
                                                                                                  ExpNeg[m - 1])
                                for m in np.arange(-l, l + 1):
                                    temp = temp + (
                                                Resultado[contador + 2 * (L + 1) ** 2 * num] * k[0] * jnp[l, num, 0] +
                                                Resultado[contador + (L + 1) ** 2 * (2 * num + 1)] * jn[l, num, 0]) * \
                                           SuperEsfericos[contador]
                                    contador = contador + 1
                                temp = haux[l] * temp
                                utemp = utemp + temp
                            del PolinomiosLegendre
                            utemp = rcuad[num] * utemp
                            u = u + utemp
                        # print u
                        auxiliar = np.exp(1j * k[0] * xvectorcart[2] - 1j * omega * t) + 1j * k[0] * u * np.exp(
                            -1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
                    else:
                        aux = xvectorcart - ps[Esfera - 1]
                        r = np.linalg.norm(aux)
                        x = aux[0]
                        y = aux[1]
                        z = aux[2]

                        theta = np.arccos(z / r)

                        phi = np.arctan2(y, x)
                        jaux = sci.spherical_jn(rango, r * k[Esfera])

                        contador = 0
                        PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                        ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                        ExpNeg = np.divide(1., ExpPos)
                        SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                        for l in rango:
                            lcuadradomasl = l * (l + 1)
                            lcuadradomasldivididoendos = lcuadradomasl / 2
                            SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                            temp = np.complex128(0.)
                            for m in np.arange(1, l + 1):
                                Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar, ExpNeg[m - 1])
                            for m in np.arange(-l, l + 1):
                                temp = temp + (-Resultado[contador + 2 * (L + 1) ** 2 * (Esfera - 1 + n)] * k[Esfera] *
                                               hnp[l, Esfera - 1, 1] + Resultado[
                                                   contador + (L + 1) ** 2 * (2 * (Esfera - 1) + 1 + 2 * n)] * hn[
                                                   l, Esfera - 1, 1]) * SuperEsfericos[contador]
                                contador = contador + 1
                            temp = jaux[l] * temp
                            u = u + temp
                        auxiliar = 1j * k[Esfera] * rcuad[Esfera - 1] * u * np.exp(-1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
        return pgrafReal + 1j * pgrafImag


def DibujaruParalelo(dd, centro, ancho, alto, interancho, interlargo, Resultado, t, omega, c, rs, rho, ps, L, ConCotas,
                     maxi, mini, paralelo):
    # Funci\'on para dibujar

    # Inputs: (falta terminar de anotarlos y anotarlos bien)
    # dd: corte DEBE SER UN INT
    # centro: centro del dibujo DEBE SER UN ARRAY DE FLOAT DE LARGO 3
    # ancho: ancho del rectangulo a dibujar DEBE SER UN FLOAT
    # alto: alto del rectangulo a dibujar DEBE SER UN FLOAT
    # interancho: cantidad de los intervalos del ancho DEBE SER UN INT
    # interlargo: cantidad de los intervalos del alto DEBE SER UN INT
    # Resultado: son los lambdas de las trazas.
    # t: el tiempo que se est\'a mirando. float
    # omega debe ser un float.
    # c debe ser una lista de floats.
    # rs radios, lista de floats.
    # rho densidades
    # ps, posiciones
    # L debe ser un int.
    # ConCotas: cotas en los valores que se presentan
    # maxi: valor superior de la cota
    # mini: valor inferior de la cota
    # paralelo

    x1 = np.linspace(-ancho / 2, ancho / 2, interancho)
    y1 = np.linspace(-alto / 2, alto / 2, interlargo)
    pgrafReal = pymp.shared.array((len(y1), len(x1)))
    pgrafImag = pymp.shared.array((len(y1), len(x1)))

    [n, k, aes] = DefinicionDeAlgunosParametros(omega, c, rho)

    import matplotlib.pyplot as plt

    # Radio al cuadrado. Este es para ahorrar operaciones en el futuro.
    rcuad = rs ** 2

    [jn, jnp, hn, hnp] = PreCalculo(n, k, rs, L)
    LMas1 = L + 1
    rango = np.arange(0, LMas1)
    if dd == 3:
        with pymp.Parallel(paralelo) as pp:
            for ii in pp.xrange(0, len(x1)):
                for jj in np.arange(0, len(y1)):
                    theta = 0.0
                    x = x1[ii]
                    y = y1[jj]
                    z = 0.

                    xvectorcart = np.asarray([x, y, z]) + centro
                    Esfera = 0
                    num = 0

                    while num < n:
                        aux = xvectorcart - ps[num]
                        raux = np.linalg.norm(aux)
                        if raux < rs[num]:
                            Esfera = num + 1
                            num = n
                        num = num + 1

                    u = np.complex128(0.)

                    if Esfera == 0:
                        for num in np.arange(0, n):
                            aux = xvectorcart - ps[num]
                            r = np.linalg.norm(aux)
                            x = aux[0]
                            y = aux[1]
                            z = aux[2]

                            theta = np.arccos(z / r)
                            phi = np.arctan2(y, x)

                            haux = sci.spherical_jn(rango, r * k[0]) + 1j * sci.spherical_yn(rango, r * k[0])
                            contador = 0
                            utemp = np.complex128(0.)

                            PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                            ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                            ExpNeg = np.divide(1., ExpPos)
                            SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                            for l in rango:
                                lcuadradomasl = l * (l + 1)
                                lcuadradomasldivididoendos = lcuadradomasl / 2
                                SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                                temp = np.complex128(0.)
                                for m in np.arange(1, l + 1):
                                    Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                    SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                    SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar,
                                                                                                  ExpNeg[m - 1])
                                for m in np.arange(-l, l + 1):
                                    temp = temp + (
                                                Resultado[contador + 2 * (L + 1) ** 2 * num] * k[0] * jnp[l, num, 0] +
                                                Resultado[contador + (L + 1) ** 2 * (2 * num + 1)] * jn[l, num, 0]) * \
                                           SuperEsfericos[contador]
                                    contador = contador + 1
                                temp = haux[l] * temp
                                utemp = utemp + temp
                            del PolinomiosLegendre
                            utemp = rcuad[num] * utemp
                            u = u + utemp
                        # print u
                        auxiliar = np.exp(1j * k[0] * xvectorcart[2] - 1j * omega * t) + 1j * k[0] * u * np.exp(
                            -1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
                    else:
                        aux = xvectorcart - ps[Esfera - 1]
                        r = np.linalg.norm(aux)
                        x = aux[0]
                        y = aux[1]
                        z = aux[2]

                        theta = np.arccos(z / r)

                        phi = np.arctan2(y, x)
                        jaux = sci.spherical_jn(rango, r * k[Esfera])

                        contador = 0
                        PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                        ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                        ExpNeg = np.divide(1., ExpPos)
                        SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                        for l in rango:
                            lcuadradomasl = l * (l + 1)
                            lcuadradomasldivididoendos = lcuadradomasl / 2
                            SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                            temp = np.complex128(0.)
                            for m in np.arange(1, l + 1):
                                Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar, ExpNeg[m - 1])
                            for m in np.arange(-l, l + 1):
                                temp = temp + (-Resultado[contador + 2 * (L + 1) ** 2 * (Esfera - 1 + n)] * k[Esfera] *
                                               hnp[l, Esfera - 1, 1] + Resultado[
                                                   contador + (L + 1) ** 2 * (2 * (Esfera - 1) + 1 + 2 * n)] * hn[
                                                   l, Esfera - 1, 1]) * SuperEsfericos[contador]
                                contador = contador + 1
                            temp = jaux[l] * temp
                            u = u + temp
                        auxiliar = 1j * k[Esfera] * rcuad[Esfera - 1] * u * np.exp(-1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
        plt.figure(facecolor='w')
        if ConCotas:
            plt.imshow(pgrafReal, vmin=mini, vmax=maxi, origin='lower',
                       extent=[-ancho / 2 + centro[0], ancho / 2 + centro[0], -alto / 2 + centro[1],
                               alto / 2 + centro[1]])
        else:
            plt.imshow(pgrafReal, origin='lower',
                       extent=[-ancho / 2 + centro[0], ancho / 2 + centro[0], -alto / 2 + centro[1],
                               alto / 2 + centro[1]])
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.colorbar()
    elif dd == 2:
        with pymp.Parallel(paralelo) as pp:
            for ii in pp.xrange(0, len(x1)):
                for jj in np.arange(0, len(y1)):
                    x = x1[ii]
                    y = 0.
                    z = y1[jj]
                    xvectorcart = np.asarray([x, y, z]) + centro
                    Esfera = 0
                    num = 0

                    while num < n:
                        aux = xvectorcart - ps[num]
                        raux = np.linalg.norm(aux)
                        if raux < rs[num]:
                            Esfera = num + 1
                            num = n
                        num = num + 1

                    u = np.complex128(0.)

                    if Esfera == 0:
                        for num in np.arange(0, n):
                            aux = xvectorcart - ps[num]
                            r = np.linalg.norm(aux)
                            x = aux[0]
                            y = aux[1]
                            z = aux[2]

                            theta = np.arccos(z / r)
                            phi = np.arctan2(y, x)

                            haux = sci.spherical_jn(rango, r * k[0]) + 1j * sci.spherical_yn(rango, r * k[0])
                            contador = 0
                            utemp = np.complex128(0.)

                            PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                            ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                            ExpNeg = np.divide(1., ExpPos)
                            SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                            for l in rango:
                                lcuadradomasl = l * (l + 1)
                                lcuadradomasldivididoendos = lcuadradomasl / 2
                                SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                                temp = np.complex128(0.)
                                for m in np.arange(1, l + 1):
                                    Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                    SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                    SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar,
                                                                                                  ExpNeg[m - 1])
                                for m in np.arange(-l, l + 1):
                                    temp = temp + (
                                                Resultado[contador + 2 * (L + 1) ** 2 * num] * k[0] * jnp[l, num, 0] +
                                                Resultado[contador + (L + 1) ** 2 * (2 * num + 1)] * jn[l, num, 0]) * \
                                           SuperEsfericos[contador]
                                    contador = contador + 1
                                temp = haux[l] * temp
                                utemp = utemp + temp
                            del PolinomiosLegendre
                            utemp = rcuad[num] * utemp
                            u = u + utemp
                        # print u
                        auxiliar = np.exp(1j * k[0] * xvectorcart[2] - 1j * omega * t) + 1j * k[0] * u * np.exp(
                            -1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
                    else:
                        aux = xvectorcart - ps[Esfera - 1]
                        r = np.linalg.norm(aux)
                        x = aux[0]
                        y = aux[1]
                        z = aux[2]

                        theta = np.arccos(z / r)

                        phi = np.arctan2(y, x)
                        jaux = sci.spherical_jn(rango, r * k[Esfera])

                        contador = 0
                        PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                        ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                        ExpNeg = np.divide(1., ExpPos)
                        SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                        for l in rango:
                            lcuadradomasl = l * (l + 1)
                            lcuadradomasldivididoendos = lcuadradomasl / 2
                            SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                            temp = np.complex128(0.)
                            for m in np.arange(1, l + 1):
                                Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar, ExpNeg[m - 1])
                            for m in np.arange(-l, l + 1):
                                temp = temp + (-Resultado[contador + 2 * (L + 1) ** 2 * (Esfera - 1 + n)] * k[Esfera] *
                                               hnp[l, Esfera - 1, 1] + Resultado[
                                                   contador + (L + 1) ** 2 * (2 * (Esfera - 1) + 1 + 2 * n)] * hn[
                                                   l, Esfera - 1, 1]) * SuperEsfericos[contador]
                                contador = contador + 1
                            temp = jaux[l] * temp
                            u = u + temp
                        auxiliar = 1j * k[Esfera] * rcuad[Esfera - 1] * u * np.exp(-1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
        plt.figure(facecolor='w')
        if ConCotas:
            plt.imshow(pgrafReal, vmin=mini, vmax=maxi, origin='lower',
                       extent=[-ancho / 2 + centro[0], ancho / 2 + centro[0], -alto / 2 + centro[2],
                               alto / 2 + centro[2]])
        else:
            plt.imshow(pgrafReal, origin='lower',
                       extent=[-ancho / 2 + centro[0], ancho / 2 + centro[0], -alto / 2 + centro[2],
                               alto / 2 + centro[2]])
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.colorbar()
    else:
        with pymp.Parallel(paralelo) as pp:
            for ii in pp.xrange(0, len(x1)):
                for jj in np.arange(0, len(y1)):
                    theta = 0.0
                    x = 0.
                    y = x1[ii]
                    z = y1[jj]
                    xvectorcart = np.asarray([x, y, z]) + centro
                    Esfera = 0
                    num = 0

                    while num < n:
                        aux = xvectorcart - ps[num]
                        raux = np.linalg.norm(aux)
                        if raux < rs[num]:
                            Esfera = num + 1
                            num = n
                        num = num + 1

                    u = np.complex128(0.)

                    if Esfera == 0:
                        for num in np.arange(0, n):
                            aux = xvectorcart - ps[num]
                            r = np.linalg.norm(aux)
                            x = aux[0]
                            y = aux[1]
                            z = aux[2]

                            theta = np.arccos(z / r)
                            phi = np.arctan2(y, x)

                            haux = sci.spherical_jn(rango, r * k[0]) + 1j * sci.spherical_yn(rango, r * k[0])
                            contador = 0
                            utemp = np.complex128(0.)

                            PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                            ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                            ExpNeg = np.divide(1., ExpPos)
                            SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                            for l in rango:
                                lcuadradomasl = l * (l + 1)
                                lcuadradomasldivididoendos = lcuadradomasl / 2
                                SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                                temp = np.complex128(0.)
                                for m in np.arange(1, l + 1):
                                    Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                    SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                    SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar,
                                                                                                  ExpNeg[m - 1])
                                for m in np.arange(-l, l + 1):
                                    temp = temp + (
                                                Resultado[contador + 2 * (L + 1) ** 2 * num] * k[0] * jnp[l, num, 0] +
                                                Resultado[contador + (L + 1) ** 2 * (2 * num + 1)] * jn[l, num, 0]) * \
                                           SuperEsfericos[contador]
                                    contador = contador + 1
                                temp = haux[l] * temp
                                utemp = utemp + temp
                            del PolinomiosLegendre
                            utemp = rcuad[num] * utemp
                            u = u + utemp
                        # print u
                        auxiliar = np.exp(1j * k[0] * xvectorcart[2] - 1j * omega * t) + 1j * k[0] * u * np.exp(
                            -1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
                    else:
                        aux = xvectorcart - ps[Esfera - 1]
                        r = np.linalg.norm(aux)
                        x = aux[0]
                        y = aux[1]
                        z = aux[2]

                        theta = np.arccos(z / r)

                        phi = np.arctan2(y, x)
                        jaux = sci.spherical_jn(rango, r * k[Esfera])

                        contador = 0
                        PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                        ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                        ExpNeg = np.divide(1., ExpPos)
                        SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                        for l in rango:
                            lcuadradomasl = l * (l + 1)
                            lcuadradomasldivididoendos = lcuadradomasl / 2
                            SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                            temp = np.complex128(0.)
                            for m in np.arange(1, l + 1):
                                Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar, ExpNeg[m - 1])
                            for m in np.arange(-l, l + 1):
                                temp = temp + (-Resultado[contador + 2 * (L + 1) ** 2 * (Esfera - 1 + n)] * k[Esfera] *
                                               hnp[l, Esfera - 1, 1] + Resultado[
                                                   contador + (L + 1) ** 2 * (2 * (Esfera - 1) + 1 + 2 * n)] * hn[
                                                   l, Esfera - 1, 1]) * SuperEsfericos[contador]
                                contador = contador + 1
                            temp = jaux[l] * temp
                            u = u + temp
                        auxiliar = 1j * k[Esfera] * rcuad[Esfera - 1] * u * np.exp(-1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
        plt.figure(facecolor='w')
        if ConCotas:
            plt.imshow(pgrafReal, vmin=mini, vmax=maxi, origin='lower',
                       extent=[-ancho / 2 + centro[0], ancho / 2 + centro[0], -alto / 2 + centro[2],
                               alto / 2 + centro[2]])
        else:
            plt.imshow(pgrafReal, origin='lower',
                       extent=[-ancho / 2 + centro[1], ancho / 2 + centro[1], -alto / 2 + centro[2],
                               alto / 2 + centro[2]])
        plt.xlabel('y [m]')
        plt.ylabel('z [m]')
        plt.colorbar()


def DibujarPlanoValoru(dd, centro, ancho, alto, interancho, interlargo, Resultado, t, omega, c, rs, rho, ps, L,
                       paralelo):
    # Funci\'on para dibujar el campo escalar. Function for drawing the scalar field

    # Inputs:
    ##dd: int. Indica si el dibujo se va a hacer en un corte paralelo al plano:
    # xy si dd==1, xz si dd==2, yz si dd=3. Indicates if the drawing is a
    # paralel cut of the plane: xy if dd==1, xz if dd==2, yz if dd==3.
    ##centro: array de floats de largo 3. Coordenadas del centro del dibujo.
    # Array of floats of length 3. Coordinates of the center of the drawing.
    ##ancho: float. Ancho del rectangulo a dibujar. Wide of the rectangle that
    # is going to be drawn.
    ##alto: float. Alto del rectangulo a dibujar. High of the rectangle that it
    # is going to be drawn.
    ##interancho: int. Cantidad de puntos que se calcular\'an a lo ancho.
    # Quantity of points that there are going to be calculate along the wide of
    # the drawing.
    ##interlargo: int. Cantidad de puntos que se calcular\'an a lo largo.
    # Quantity of points that there are going to be calculate along the high of
    # the drawing.
    ##Resultado: output solucion de la funci\'on ResolverTodo. output solucion
    # of the function ResolverTodo
    # t: float, tiempo que se va a dibujar. Time that it is going to be drawn.
    ##omega: float o complex. Es la frecuencia de la onda incidente. (float or
    # complex. It's the frequency of the incident wave).
    ##c: array de floats. Array de la rapidez del sonido de cada medio.
    # (Array of floats. Array of the sound speed in each medium).
    ##rs: array de nx1 de floats. Radio de las esferas. (Array of nx1 of
    # floats. Sphere radius).
    ##rho: array de floats. Array de las densidades de cada medio. (Array of
    # floats. Array of the density of each medium).
    ##ps: posiciones del centro de las esferas.
    ##L: int. Orden m\'aximo de los arm\'nicos esf\'ericos que se utilizan para
    # discretizar las trazas. (L+1)**2 es la cantidad total de arm\'onicos
    # esf\'ericos que se utilizan para discretizar una traza. (Maximum order of
    # spherical harmonics that are used to discretize the traces. (L+1)**2 is
    # the total number of spherical harmonics that are used to discretize a
    # trace).
    ##paralelo: int. N\'umero de procesadores que se utilizar\'an para
    # paralelizar el proceso.

    x1 = np.linspace(-ancho / 2, ancho / 2, interancho)
    y1 = np.linspace(-alto / 2, alto / 2, interlargo)
    pgrafReal = pymp.shared.array((len(y1), len(x1)))
    pgrafImag = pymp.shared.array((len(y1), len(x1)))

    [n, k, aes] = DefinicionDeAlgunosParametros(omega, c, rho)

    # Radio al cuadrado. Este es para ahorrar operaciones en el futuro.
    rcuad = rs ** 2

    [jn, jnp, hn, hnp] = PreCalculo(n, k, rs, L)
    LMas1 = L + 1
    rango = np.arange(0, LMas1)
    if dd == 3:
        with pymp.Parallel(paralelo) as pp:
            for ii in pp.xrange(0, len(x1)):
                for jj in np.arange(0, len(y1)):
                    theta = 0.0
                    x = x1[ii]
                    y = y1[jj]
                    z = 0.

                    xvectorcart = np.asarray([x, y, z]) + centro
                    Esfera = 0
                    num = 0

                    while num < n:
                        aux = xvectorcart - ps[num]
                        raux = np.linalg.norm(aux)
                        if raux < rs[num]:
                            Esfera = num + 1
                            num = n
                        num = num + 1

                    u = np.complex128(0.)

                    if Esfera == 0:
                        for num in np.arange(0, n):
                            aux = xvectorcart - ps[num]
                            r = np.linalg.norm(aux)
                            x = aux[0]
                            y = aux[1]
                            z = aux[2]

                            theta = np.arccos(z / r)
                            phi = np.arctan2(y, x)

                            haux = sci.spherical_jn(rango, r * k[0]) + 1j * sci.spherical_yn(rango, r * k[0])
                            contador = 0
                            utemp = np.complex128(0.)

                            PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                            ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                            ExpNeg = np.divide(1., ExpPos)
                            SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                            for l in rango:
                                lcuadradomasl = l * (l + 1)
                                lcuadradomasldivididoendos = lcuadradomasl / 2
                                SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                                temp = np.complex128(0.)
                                for m in np.arange(1, l + 1):
                                    Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                    SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                    SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar,
                                                                                                  ExpNeg[m - 1])
                                for m in np.arange(-l, l + 1):
                                    temp = temp + (
                                                Resultado[contador + 2 * (L + 1) ** 2 * num] * k[0] * jnp[l, num, 0] +
                                                Resultado[contador + (L + 1) ** 2 * (2 * num + 1)] * jn[l, num, 0]) * \
                                           SuperEsfericos[contador]
                                    contador = contador + 1
                                temp = haux[l] * temp
                                utemp = utemp + temp
                            del PolinomiosLegendre
                            utemp = rcuad[num] * utemp
                            u = u + utemp
                        # print u
                        auxiliar = np.exp(1j * k[0] * xvectorcart[2] - 1j * omega * t) + 1j * k[0] * u * np.exp(
                            -1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
                    else:
                        aux = xvectorcart - ps[Esfera - 1]
                        r = np.linalg.norm(aux)
                        x = aux[0]
                        y = aux[1]
                        z = aux[2]

                        theta = np.arccos(z / r)

                        phi = np.arctan2(y, x)
                        jaux = sci.spherical_jn(rango, r * k[Esfera])

                        contador = 0
                        PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                        ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                        ExpNeg = np.divide(1., ExpPos)
                        SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                        for l in rango:
                            lcuadradomasl = l * (l + 1)
                            lcuadradomasldivididoendos = lcuadradomasl / 2
                            SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                            temp = np.complex128(0.)
                            for m in np.arange(1, l + 1):
                                Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar, ExpNeg[m - 1])
                            for m in np.arange(-l, l + 1):
                                temp = temp + (-Resultado[contador + 2 * (L + 1) ** 2 * (Esfera - 1 + n)] * k[Esfera] *
                                               hnp[l, Esfera - 1, 1] + Resultado[
                                                   contador + (L + 1) ** 2 * (2 * (Esfera - 1) + 1 + 2 * n)] * hn[
                                                   l, Esfera - 1, 1]) * SuperEsfericos[contador]
                                contador = contador + 1
                            temp = jaux[l] * temp
                            u = u + temp
                        auxiliar = 1j * k[Esfera] * rcuad[Esfera - 1] * u * np.exp(-1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
        return pgrafReal + 1j * pgrafImag
    elif dd == 2:
        with pymp.Parallel(paralelo) as pp:
            for ii in pp.xrange(0, len(x1)):
                for jj in np.arange(0, len(y1)):
                    x = x1[ii]
                    y = 0.
                    z = y1[jj]
                    xvectorcart = np.asarray([x, y, z]) + centro
                    Esfera = 0
                    num = 0

                    while num < n:
                        aux = xvectorcart - ps[num]
                        raux = np.linalg.norm(aux)
                        if raux < rs[num]:
                            Esfera = num + 1
                            num = n
                        num = num + 1

                    u = np.complex128(0.)

                    if Esfera == 0:
                        for num in np.arange(0, n):
                            aux = xvectorcart - ps[num]
                            r = np.linalg.norm(aux)
                            x = aux[0]
                            y = aux[1]
                            z = aux[2]

                            theta = np.arccos(z / r)
                            phi = np.arctan2(y, x)

                            haux = sci.spherical_jn(rango, r * k[0]) + 1j * sci.spherical_yn(rango, r * k[0])
                            contador = 0
                            utemp = np.complex128(0.)

                            PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                            ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                            ExpNeg = np.divide(1., ExpPos)
                            SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                            for l in rango:
                                lcuadradomasl = l * (l + 1)
                                lcuadradomasldivididoendos = lcuadradomasl / 2
                                SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                                temp = np.complex128(0.)
                                for m in np.arange(1, l + 1):
                                    Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                    SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                    SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar,
                                                                                                  ExpNeg[m - 1])
                                for m in np.arange(-l, l + 1):
                                    temp = temp + (
                                                Resultado[contador + 2 * (L + 1) ** 2 * num] * k[0] * jnp[l, num, 0] +
                                                Resultado[contador + (L + 1) ** 2 * (2 * num + 1)] * jn[l, num, 0]) * \
                                           SuperEsfericos[contador]
                                    contador = contador + 1
                                temp = haux[l] * temp
                                utemp = utemp + temp
                            del PolinomiosLegendre
                            utemp = rcuad[num] * utemp
                            u = u + utemp
                        # print u
                        auxiliar = np.exp(1j * k[0] * xvectorcart[2] - 1j * omega * t) + 1j * k[0] * u * np.exp(
                            -1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
                    else:
                        aux = xvectorcart - ps[Esfera - 1]
                        r = np.linalg.norm(aux)
                        x = aux[0]
                        y = aux[1]
                        z = aux[2]

                        theta = np.arccos(z / r)

                        phi = np.arctan2(y, x)
                        jaux = sci.spherical_jn(rango, r * k[Esfera])

                        contador = 0
                        PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                        ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                        ExpNeg = np.divide(1., ExpPos)
                        SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                        for l in rango:
                            lcuadradomasl = l * (l + 1)
                            lcuadradomasldivididoendos = lcuadradomasl / 2
                            SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                            temp = np.complex128(0.)
                            for m in np.arange(1, l + 1):
                                Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar, ExpNeg[m - 1])
                            for m in np.arange(-l, l + 1):
                                temp = temp + (-Resultado[contador + 2 * (L + 1) ** 2 * (Esfera - 1 + n)] * k[Esfera] *
                                               hnp[l, Esfera - 1, 1] + Resultado[
                                                   contador + (L + 1) ** 2 * (2 * (Esfera - 1) + 1 + 2 * n)] * hn[
                                                   l, Esfera - 1, 1]) * SuperEsfericos[contador]
                                contador = contador + 1
                            temp = jaux[l] * temp
                            u = u + temp
                        auxiliar = 1j * k[Esfera] * rcuad[Esfera - 1] * u * np.exp(-1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
        return pgrafReal + 1j * pgrafImag
    else:
        with pymp.Parallel(paralelo) as pp:
            for ii in pp.xrange(0, len(x1)):
                for jj in np.arange(0, len(y1)):
                    theta = 0.0
                    x = 0.
                    y = x1[ii]
                    z = y1[jj]
                    xvectorcart = np.asarray([x, y, z]) + centro
                    Esfera = 0
                    num = 0

                    while num < n:
                        aux = xvectorcart - ps[num]
                        raux = np.linalg.norm(aux)
                        if raux < rs[num]:
                            Esfera = num + 1
                            num = n
                        num = num + 1

                    u = np.complex128(0.)

                    if Esfera == 0:
                        for num in np.arange(0, n):
                            aux = xvectorcart - ps[num]
                            r = np.linalg.norm(aux)
                            x = aux[0]
                            y = aux[1]
                            z = aux[2]

                            theta = np.arccos(z / r)
                            phi = np.arctan2(y, x)

                            haux = sci.spherical_jn(rango, r * k[0]) + 1j * sci.spherical_yn(rango, r * k[0])
                            contador = 0
                            utemp = np.complex128(0.)

                            PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                            ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                            ExpNeg = np.divide(1., ExpPos)
                            SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                            for l in rango:
                                lcuadradomasl = l * (l + 1)
                                lcuadradomasldivididoendos = lcuadradomasl / 2
                                SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                                temp = np.complex128(0.)
                                for m in np.arange(1, l + 1):
                                    Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                    SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                    SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar,
                                                                                                  ExpNeg[m - 1])
                                for m in np.arange(-l, l + 1):
                                    temp = temp + (
                                                Resultado[contador + 2 * (L + 1) ** 2 * num] * k[0] * jnp[l, num, 0] +
                                                Resultado[contador + (L + 1) ** 2 * (2 * num + 1)] * jn[l, num, 0]) * \
                                           SuperEsfericos[contador]
                                    contador = contador + 1
                                temp = haux[l] * temp
                                utemp = utemp + temp
                            del PolinomiosLegendre
                            utemp = rcuad[num] * utemp
                            u = u + utemp
                        # print u
                        auxiliar = np.exp(1j * k[0] * xvectorcart[2] - 1j * omega * t) + 1j * k[0] * u * np.exp(
                            -1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
                    else:
                        aux = xvectorcart - ps[Esfera - 1]
                        r = np.linalg.norm(aux)
                        x = aux[0]
                        y = aux[1]
                        z = aux[2]

                        theta = np.arccos(z / r)

                        phi = np.arctan2(y, x)
                        jaux = sci.spherical_jn(rango, r * k[Esfera])

                        contador = 0
                        PolinomiosLegendre = psh.legendre.PlmON(LMas1, np.cos(theta), csphase=-1, cnorm=1)
                        ExpPos = np.exp(1j * np.arange(1, L + 1) * phi)
                        ExpNeg = np.divide(1., ExpPos)
                        SuperEsfericos = np.zeros((np.square(L + 1)), dtype=np.complex128)
                        for l in rango:
                            lcuadradomasl = l * (l + 1)
                            lcuadradomasldivididoendos = lcuadradomasl / 2
                            SuperEsfericos[lcuadradomasl] = PolinomiosLegendre[lcuadradomasldivididoendos]
                            temp = np.complex128(0.)
                            for m in np.arange(1, l + 1):
                                Auxiliar = PolinomiosLegendre[lcuadradomasldivididoendos + m]
                                SuperEsfericos[(lcuadradomasl + m)] = np.multiply(Auxiliar, ExpPos[m - 1])
                                SuperEsfericos[(lcuadradomasl - m)] = (-1) ** m * np.multiply(Auxiliar, ExpNeg[m - 1])
                            for m in np.arange(-l, l + 1):
                                temp = temp + (-Resultado[contador + 2 * (L + 1) ** 2 * (Esfera - 1 + n)] * k[Esfera] *
                                               hnp[l, Esfera - 1, 1] + Resultado[
                                                   contador + (L + 1) ** 2 * (2 * (Esfera - 1) + 1 + 2 * n)] * hn[
                                                   l, Esfera - 1, 1]) * SuperEsfericos[contador]
                                contador = contador + 1
                            temp = jaux[l] * temp
                            u = u + temp
                        auxiliar = 1j * k[Esfera] * rcuad[Esfera - 1] * u * np.exp(-1j * omega * t)
                        pgrafReal[jj, ii] = np.real(auxiliar)
                        pgrafImag[jj, ii] = np.imag(auxiliar)
        return pgrafReal + 1j * pgrafImag


def DibujoWindows(dd, centro, ancho, alto, ConCotas, maxi, mini, pgraf, color):
    ###Inputs
    ##dd: int. Indica si el dibujo se va a hacer en un corte paralelo al plano:
    # xy si dd==1, xz si dd==2, yz si dd=3. Indicates if the drawing is a
    # paralel cut of the plane: xy if dd==1, xz if dd==2, yz if dd==3.
    ##centro: array de floats de largo 3. Coordenadas del centro del dibujo.
    # Array of floats of length 3. Coordinates of the center of the drawing.
    ##ancho: float. Ancho del rectangulo a dibujar. Wide of the rectangle that
    # is going to be drawn.
    ##alto: float. Alto del rectangulo a dibujar. High of the rectangle that it
    # is going to be drawn.
    ##ConCotas: bool. Si se van a considerar cotas de valores para el campo en
    # el gr\'afico. (If there are going to consider bounds of values of the
    # field for the drawing).
    ##maxi: float. Valor superior de la cota. Upper bound.
    ##mini: float. Valor inferior de la cota. Lower bound.
    ##pgraf: array 2D, con los valores de campo. (2D array with the values of
    # the field).

    import matplotlib.pyplot as plt

    plt.figure(facecolor='w')
    if dd == 1:
        if ConCotas:
            plt.imshow(pgraf, vmin=mini, vmax=maxi, origin='lower',
                       extent=[-ancho / 2 + centro[0], ancho / 2 + centro[0], -alto / 2 + centro[1],
                               alto / 2 + centro[1]], cmap=color)
        else:
            plt.imshow(pgraf, origin='lower',
                       extent=[-ancho / 2 + centro[0], ancho / 2 + centro[0], -alto / 2 + centro[1],
                               alto / 2 + centro[1]], cmap=color)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
    elif dd == 2:
        if ConCotas:
            plt.imshow(pgraf, vmin=mini, vmax=maxi, origin='lower',
                       extent=[-ancho / 2 + centro[0], ancho / 2 + centro[0], -alto / 2 + centro[2],
                               alto / 2 + centro[2]], cmap=color)
        else:
            plt.imshow(pgraf, origin='lower',
                       extent=[-ancho / 2 + centro[0], ancho / 2 + centro[0], -alto / 2 + centro[2],
                               alto / 2 + centro[2]], cmap=color)
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
    else:
        if ConCotas:
            plt.imshow(pgraf, vmin=mini, vmax=maxi, origin='lower',
                       extent=[-ancho / 2 + centro[0], ancho / 2 + centro[0], -alto / 2 + centro[2],
                               alto / 2 + centro[2]], cmap=color)
        else:
            plt.imshow(pgraf, origin='lower',
                       extent=[-ancho / 2 + centro[1], ancho / 2 + centro[1], -alto / 2 + centro[2],
                               alto / 2 + centro[2]], cmap=color)
        plt.xlabel('y [m]')
        plt.ylabel('z [m]')
    plt.colorbar()


##Fin de rutinas de dibujo----------------------------------------------------------------------------


'''
def ConvergenciaAnalitica1Esfera(Lo,L,omega,c,rs,ps,rho,regla):
    [n,k,aes]=DefinicionDeAlgunosParametros(omega,c,rho)
    [k0,k1,a,r1]=[k[0],k[1],aes[0],rs[0]]

    #Radio por n\'umero de onda
    r1k0=r1*k0
    r1k1=r1*k1

    LMas1=Lo+1
    rango=np.arange(0,LMas1)

    #Funciones esfericas de Bessel y hankel precalculadas en r1k0 y r1k1 hasta L
    jn0=sci.spherical_jn(rango,r1k0)
    jn0p=sci.spherical_jn(rango,r1k0,derivative=True)

    jn1=sci.spherical_jn(rango,r1k1)
    jn1p=sci.spherical_jn(rango,r1k1,derivative=True)

    hn0=jn0+1j*sci.spherical_yn(rango,r1k0)
    hn0p=jn0p+1j*sci.spherical_yn(rango,r1k0,derivative=True)

    Auxiliar1=(-a*jn0p[rango]*jn1[rango]*k0+jn0[rango]*jn1p[rango]*k1)/(a*hn0p[rango]*jn1[rango]*k0-hn0[rango]*jn1p[rango]*k1)
    Auxiliar2=(-hn0[rango]*jn0p[rango]+hn0p[rango]*jn0[rango])/(a*hn0p[rango]*jn1[rango]*k0-hn0[rango]*jn1p[rango]*k1)
    lambdateo0=2*1j**rango*np.sqrt(np.pi*(2*rango+1))*hn0[rango]*Auxiliar1
    lambdateo1=-2*k0*1j**rango*np.sqrt(np.pi*(2*rango+1))*hn0p[rango]*Auxiliar1
    lambdateo2=2*a*k0*1j**rango*np.sqrt(np.pi*(2*rango+1))*jn1[rango]*Auxiliar2
    lambdateo3=2*a*k0*k1*1j**rango*np.sqrt(np.pi*(2*rango+1))*jn1p[rango]*Auxiliar2

    extra0=np.zeros((L+1))
    extra1=np.zeros((L+1))
    extra2=np.zeros((L+1))
    extra3=np.zeros((L+1))
    for l in range(0,L+1):
        Resolucion=ResolverTodo(omega,c,rs,ps,rho,l,regla)
        Resolucion=Resolucion[0]
        contador =0
        for laux in range(0,l+1):
            for m in range(-laux,laux+1):
                if m!=0:
                    extra0[l]=extra0[l]+np.abs(Resolucion[contador])**2
                    extra1[l]=extra1[l]+np.abs(Resolucion[contador+(l+1)**2])**2
                    extra2[l]=extra2[l]+np.abs(Resolucion[contador+2*(l+1)**2])**2
                    extra3[l]=extra3[l]+np.abs(Resolucion[contador+3*(l+1)**2])**2
                else:
                    extra0[l]=extra0[l]+np.abs(Resolucion[contador]-lambdateo0[laux])**2
                    extra1[l]=extra1[l]+np.abs(Resolucion[contador+(l+1)**2]-lambdateo1[laux])**2
                    extra2[l]=extra2[l]+np.abs(Resolucion[contador+2*(l+1)**2]-lambdateo2[laux])**2
                    extra3[l]=extra3[l]+np.abs(Resolucion[contador+3*(l+1)**2]-lambdateo3[laux])**2
                contador=contador+1
        extra0[l]=extra0[l]+np.linalg.norm(lambdateo0[np.asarray(range(l+1,LMas1))])**2      
        extra1[l]=extra1[l]+np.linalg.norm(lambdateo1[np.asarray(range(l+1,LMas1))])**2
        extra2[l]=extra2[l]+np.linalg.norm(lambdateo2[np.asarray(range(l+1,LMas1))])**2
        extra3[l]=extra3[l]+np.linalg.norm(lambdateo3[np.asarray(range(l+1,LMas1))])**2
    extra0=np.sqrt(extra0)
    extra1=np.sqrt(extra1)
    extra2=np.sqrt(extra2)
    extra3=np.sqrt(extra3)
    return [[extra0,extra1,extra2,extra3],[extra0/extra0[0],extra1/extra1[0],extra2/extra2[0],extra3/extra3[0]]]

def ConvergenciaAnalitica1EsferaCalderonYSalto(Lo,L,omega,c,rs,ps,rho,regla):
    [n,k,aes]=DefinicionDeAlgunosParametros(omega,c,rho)
    [k0,k1,a,r1]=[k[0],k[1],aes[0],rs[0]]

    #Radio por n\'umero de onda
    r1k0=r1*k0
    r1k1=r1*k1

    LMas1=Lo+1
    rango=np.arange(0,LMas1)

    #Funciones esfericas de Bessel y hankel precalculadas en r1k0 y r1k1 hasta L
    jn0=sci.spherical_jn(rango,r1k0)
    jn0p=sci.spherical_jn(rango,r1k0,derivative=True)

    jn1=sci.spherical_jn(rango,r1k1)
    jn1p=sci.spherical_jn(rango,r1k1,derivative=True)

    hn0=jn0+1j*sci.spherical_yn(rango,r1k0)
    hn0p=jn0p+1j*sci.spherical_yn(rango,r1k0,derivative=True)

    Auxiliar1=(-a*jn0p[rango]*jn1[rango]*k0+jn0[rango]*jn1p[rango]*k1)/(a*hn0p[rango]*jn1[rango]*k0-hn0[rango]*jn1p[rango]*k1)
    Auxiliar2=(-hn0[rango]*jn0p[rango]+hn0p[rango]*jn0[rango])/(a*hn0p[rango]*jn1[rango]*k0-hn0[rango]*jn1p[rango]*k1)
    lambdateo0=2*1j**rango*np.sqrt(np.pi*(2*rango+1))*hn0[rango]*Auxiliar1
    lambdateo1=-2*k0*1j**rango*np.sqrt(np.pi*(2*rango+1))*hn0p[rango]*Auxiliar1
    lambdateo2=2*a*k0*1j**rango*np.sqrt(np.pi*(2*rango+1))*jn1[rango]*Auxiliar2
    lambdateo3=2*a*k0*k1*1j**rango*np.sqrt(np.pi*(2*rango+1))*jn1p[rango]*Auxiliar2

    extra0=np.zeros((L+1))
    extra1=np.zeros((L+1))
    extra2=np.zeros((L+1))
    extra3=np.zeros((L+1))
    Calderon=np.zeros((L+1,2))
    ErrorSal=np.zeros((L+1))
    ErrorResolucionSistema=np.zeros((L+1))
    for l in range(0,L+1):
        Resolucion=ResolverTodo(omega,c,rs,ps,rho,l,regla)
        ErrorSal[l]=ErrorSalto(Resolucion[1],Resolucion[0],Resolucion[2],k,aes,rs,ps,l,Lo)
        Calderon[l]=ComprobacionCalderonDiscreto(Resolucion[1],Resolucion[0],l,n)
        ErrorResolucionSistema[l]=np.linalg.norm(Resolucion[1]*Resolucion[0]-Resolucion[2].toarray()[:,0])
        Resolucion=Resolucion[0]
        contador =0
        for laux in range(0,l+1):
            for m in range(-laux,laux+1):
                if m!=0:
                    extra0[l]=extra0[l]+np.abs(Resolucion[contador])**2
                    extra1[l]=extra1[l]+np.abs(Resolucion[contador+(l+1)**2])**2
                    extra2[l]=extra2[l]+np.abs(Resolucion[contador+2*(l+1)**2])**2
                    extra3[l]=extra3[l]+np.abs(Resolucion[contador+3*(l+1)**2])**2
                else:
                    extra0[l]=extra0[l]+np.abs(Resolucion[contador]-lambdateo0[laux])**2
                    extra1[l]=extra1[l]+np.abs(Resolucion[contador+(l+1)**2]-lambdateo1[laux])**2
                    extra2[l]=extra2[l]+np.abs(Resolucion[contador+2*(l+1)**2]-lambdateo2[laux])**2
                    extra3[l]=extra3[l]+np.abs(Resolucion[contador+3*(l+1)**2]-lambdateo3[laux])**2
                contador=contador+1
        extra0[l]=extra0[l]+np.linalg.norm(lambdateo0[np.asarray(range(l+1,LMas1))])**2      
        extra1[l]=extra1[l]+np.linalg.norm(lambdateo1[np.asarray(range(l+1,LMas1))])**2
        extra2[l]=extra2[l]+np.linalg.norm(lambdateo2[np.asarray(range(l+1,LMas1))])**2
        extra3[l]=extra3[l]+np.linalg.norm(lambdateo3[np.asarray(range(l+1,LMas1))])**2
    extra0=np.sqrt(extra0)
    extra1=np.sqrt(extra1)
    extra2=np.sqrt(extra2)
    extra3=np.sqrt(extra3)
    return [[extra0,extra1,extra2,extra3],[extra0/extra0[0],extra1/extra1[0],extra2/extra2[0],extra3/extra3[0]],Calderon,ErrorSal,ErrorResolucionSistema]

def ConvergenciaAnaliticaCalderonYSalto(Lo,L,omega,c,rs,ps,rho,regla,cual):
    [n,k,aes]=DefinicionDeAlgunosParametros(omega,c,rho)
    [k0,k1,a,r1]=[k[0],k[cual],aes[cual-1],rs[cual-1]]

    #Radio por n\'umero de onda
    r1k0=r1*k0
    r1k1=r1*k1

    LMas1=Lo+1
    rango=np.arange(0,LMas1)

    #Funciones esfericas de Bessel y hankel precalculadas en r1k0 y r1k1 hasta L
    jn0=sci.spherical_jn(rango,r1k0)
    jn0p=sci.spherical_jn(rango,r1k0,derivative=True)

    jn1=sci.spherical_jn(rango,r1k1)
    jn1p=sci.spherical_jn(rango,r1k1,derivative=True)

    hn0=jn0+1j*sci.spherical_yn(rango,r1k0)
    hn0p=jn0p+1j*sci.spherical_yn(rango,r1k0,derivative=True)

    Auxiliar1=(-a*jn0p[rango]*jn1[rango]*k0+jn0[rango]*jn1p[rango]*k1)/(a*hn0p[rango]*jn1[rango]*k0-hn0[rango]*jn1p[rango]*k1)
    Auxiliar2=(-hn0[rango]*jn0p[rango]+hn0p[rango]*jn0[rango])/(a*hn0p[rango]*jn1[rango]*k0-hn0[rango]*jn1p[rango]*k1)
    lambdateo0=2*1j**rango*np.sqrt(np.pi*(2*rango+1))*hn0[rango]*Auxiliar1
    lambdateo1=-2*k0*1j**rango*np.sqrt(np.pi*(2*rango+1))*hn0p[rango]*Auxiliar1
    lambdateo2=2*a*k0*1j**rango*np.sqrt(np.pi*(2*rango+1))*jn1[rango]*Auxiliar2
    lambdateo3=2*a*k0*k1*1j**rango*np.sqrt(np.pi*(2*rango+1))*jn1p[rango]*Auxiliar2

    extra0=np.zeros((L+1))
    extra1=np.zeros((L+1))
    extra2=np.zeros((L+1))
    extra3=np.zeros((L+1))
    Calderon=np.zeros((L+1,2))
    ErrorSal=np.zeros((L+1))
    ErrorResolucionSistema=np.zeros((L+1))
    for l in range(0,L+1):
        Resolucion=ResolverTodo(omega,c,rs,ps,rho,l,regla)
        ErrorSal[l]=ErrorSalto(Resolucion[1],Resolucion[0],Resolucion[2],k,aes,rs,ps,l,Lo)
        Calderon[l]=ComprobacionCalderonDiscreto(Resolucion[1],Resolucion[0],l,n)
        ErrorResolucionSistema[l]=np.linalg.norm(Resolucion[1]*Resolucion[0]-Resolucion[2].toarray()[:,0])
        Resolucion=Resolucion[0]
        contador =0
        for laux in range(0,l+1):
            for m in range(-laux,laux+1):
                if m!=0:
                    extra0[l]=extra0[l]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador])**2
                    extra1[l]=extra1[l]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador+(l+1)**2])**2
                    extra2[l]=extra2[l]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador+2*n*(l+1)**2])**2
                    extra3[l]=extra3[l]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador+(2*n+1)*(l+1)**2])**2
                else:
                    extra0[l]=extra0[l]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador]-lambdateo0[laux])**2
                    extra1[l]=extra1[l]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador+(l+1)**2]-lambdateo1[laux])**2
                    extra2[l]=extra2[l]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador+2*n*(l+1)**2]-lambdateo2[laux])**2
                    extra3[l]=extra3[l]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador+(2*n+1)*(l+1)**2]-lambdateo3[laux])**2
                contador=contador+1
        extra0[l]=extra0[l]+np.linalg.norm(lambdateo0[np.asarray(range(l+1,LMas1))])**2      
        extra1[l]=extra1[l]+np.linalg.norm(lambdateo1[np.asarray(range(l+1,LMas1))])**2
        extra2[l]=extra2[l]+np.linalg.norm(lambdateo2[np.asarray(range(l+1,LMas1))])**2
        extra3[l]=extra3[l]+np.linalg.norm(lambdateo3[np.asarray(range(l+1,LMas1))])**2
    extra0=np.sqrt(extra0)
    extra1=np.sqrt(extra1)
    extra2=np.sqrt(extra2)
    extra3=np.sqrt(extra3)
    return [[extra0,extra1,extra2,extra3],[extra0/extra0[0],extra1/extra1[0],extra2/extra2[0],extra3/extra3[0]],Calderon,ErrorSal,ErrorResolucionSistema]


def ConvergenciaAnaliticaCalderonYSaltoDesdeUnL(Lo,L,Li,omega,c,rs,ps,rho,regla,cual):
    [n,k,aes]=DefinicionDeAlgunosParametros(omega,c,rho)
    [k0,k1,a,r1]=[k[0],k[cual],aes[cual-1],rs[cual-1]]

    #Radio por n\'umero de onda
    r1k0=r1*k0
    r1k1=r1*k1

    LMas1=Lo+1
    rango=np.arange(0,LMas1)

    #Funciones esfericas de Bessel y hankel precalculadas en r1k0 y r1k1 hasta L
    jn0=sci.spherical_jn(rango,r1k0)
    jn0p=sci.spherical_jn(rango,r1k0,derivative=True)

    jn1=sci.spherical_jn(rango,r1k1)
    jn1p=sci.spherical_jn(rango,r1k1,derivative=True)

    hn0=jn0+1j*sci.spherical_yn(rango,r1k0)
    hn0p=jn0p+1j*sci.spherical_yn(rango,r1k0,derivative=True)

    Auxiliar1=(-a*jn0p[rango]*jn1[rango]*k0+jn0[rango]*jn1p[rango]*k1)/(a*hn0p[rango]*jn1[rango]*k0-hn0[rango]*jn1p[rango]*k1)
    Auxiliar2=(-hn0[rango]*jn0p[rango]+hn0p[rango]*jn0[rango])/(a*hn0p[rango]*jn1[rango]*k0-hn0[rango]*jn1p[rango]*k1)
    lambdateo0=2*1j**rango*np.sqrt(np.pi*(2*rango+1))*hn0[rango]*Auxiliar1
    lambdateo1=-2*k0*1j**rango*np.sqrt(np.pi*(2*rango+1))*hn0p[rango]*Auxiliar1
    lambdateo2=2*a*k0*1j**rango*np.sqrt(np.pi*(2*rango+1))*jn1[rango]*Auxiliar2
    lambdateo3=2*a*k0*k1*1j**rango*np.sqrt(np.pi*(2*rango+1))*jn1p[rango]*Auxiliar2

    extra0=np.zeros((L-Li+1))
    extra1=np.zeros((L-Li+1))
    extra2=np.zeros((L-Li+1))
    extra3=np.zeros((L-Li+1))
    Calderon=np.zeros((L-Li+1,2))
    ErrorSal=np.zeros((L-Li+1))
    ErrorResolucionSistema=np.zeros((L-Li+1))
    contador2=0
    for l in range(Li,L+1):
        Resolucion=ResolverTodo(omega,c,rs,ps,rho,l,regla)
        ErrorSal[l-Li]=ErrorSalto(Resolucion[1],Resolucion[0],Resolucion[2],k,aes,rs,ps,l,Lo)
        Calderon[l-Li]=ComprobacionCalderonDiscreto(Resolucion[1],Resolucion[0],l,n)
        ErrorResolucionSistema[l-Li]=np.linalg.norm(Resolucion[1]*Resolucion[0]-Resolucion[2].toarray()[:,0])
        Resolucion=Resolucion[0]
        contador =0
        for laux in range(0,l+1):
            for m in range(-laux,laux+1):
                if m!=0:
                    extra0[l-Li]=extra0[l-Li]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador])**2
                    extra1[l-Li]=extra1[l-Li]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador+(l+1)**2])**2
                    extra2[l-Li]=extra2[l-Li]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador+2*n*(l+1)**2])**2
                    extra3[l-Li]=extra3[l-Li]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador+(2*n+1)*(l+1)**2])**2
                else:
                    extra0[l-Li]=extra0[l-Li]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador]-lambdateo0[laux])**2
                    extra1[l-Li]=extra1[l-Li]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador+(l+1)**2]-lambdateo1[laux])**2
                    extra2[l-Li]=extra2[l-Li]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador+2*n*(l+1)**2]-lambdateo2[laux])**2
                    extra3[l-Li]=extra3[l-Li]+np.abs(Resolucion[2*(cual-1)*(l+1)**2+contador+(2*n+1)*(l+1)**2]-lambdateo3[laux])**2
                contador=contador+1
        extra0[l-Li]=extra0[l-Li]+np.linalg.norm(lambdateo0[np.asarray(range(l+1,LMas1))])**2      
        extra1[l-Li]=extra1[l-Li]+np.linalg.norm(lambdateo1[np.asarray(range(l+1,LMas1))])**2
        extra2[l-Li]=extra2[l-Li]+np.linalg.norm(lambdateo2[np.asarray(range(l+1,LMas1))])**2
        extra3[l-Li]=extra3[l-Li]+np.linalg.norm(lambdateo3[np.asarray(range(l+1,LMas1))])**2
        contador2=contador2+1
    extra0=np.sqrt(extra0)
    extra1=np.sqrt(extra1)
    extra2=np.sqrt(extra2)
    extra3=np.sqrt(extra3)
    return [[extra0,extra1,extra2,extra3],[extra0/np.linalg.norm(lambdateo0),extra1/np.linalg.norm(lambdateo1),extra2/np.linalg.norm(lambdateo2),extra3/np.linalg.norm(lambdateo3)],Calderon,ErrorSal,ErrorResolucionSistema]


def ConvergenciaCuadraturaCalderonSalto(L,omega,c,rs,ps,rho,topeRegla):
    [n,k,aes]=DefinicionDeAlgunosParametros(omega,c,rho)

    Calderon=np.zeros((topeRegla,2))
    ErrorSal=np.zeros((topeRegla+1))
    ErrorResolucionSistema=np.zeros((topeRegla))
    for regla in range(0,topeRegla):
        Resolucion=ResolverTodo(omega,c,rs,ps,rho,L,regla+1)
        ErrorSal[regla]=ErrorSalto(Resolucion[1],Resolucion[0],Resolucion[2],k,aes,rs,ps,L,L)
        Calderon[regla]=ComprobacionCalderonDiscreto(Resolucion[1],Resolucion[0],L,n)
        ErrorResolucionSistema[regla]=np.linalg.norm(Resolucion[1]*Resolucion[0]-Resolucion[2].toarray()[:,0])
        Resolucion=Resolucion[0]
    return [Calderon,ErrorSal,ErrorResolucionSistema]


def Dibujaru(dd,centro,ancho,alto,interancho,interlargo,Resultado,t,omega,c,listaParametros,rs,ps,L,ConCotas,maxi,mini):
    #Funci\'on para dibujar

    #Inputs: (falta terminar de anotarlos y anotarlos bien)
    #dd: corte paralelo al eje x(1),y(2),z(3) DEBE SER UN INT
    #centro: centro del dibujo DEBE SER UN ARRAY DE FLOAT DE LARGO 3
    #ancho: ancho del rectangulo a dibujar DEBE SER UN FLOAT
    #alto: alto del rectangulo a dibujar DEBE SER UN FLOAT
    #interancho: cantidad de los intervalos del ancho DEBE SER UN INT
    #interlargo: cantidad de los intervalos del alto DEBE SER UN INT
    #omega debe ser un float.
    #c debe ser una lista de floats.
    #L debe ser un int.
    #t debe ser un float.
    #listaParametros=DefinicionDeAlgunosParametros(omega,c0,c1,r1,p1,rho0,rho1)

    x1=np.linspace(-ancho/2,ancho/2,interancho)
    y1=np.linspace(-alto/2,alto/2,interlargo)
    XE,YE=np.meshgrid(x1,y1,sparse='True')
    pgraf=XE*1j*0.+YE*1j*0.


    #M\'as par\'ametros, pero a partir de los anteriores

    [k,aes,n]=listaParametros
    import matplotlib.pyplot as plt

    #Radio al cuadrado. Este es para ahorrar operaciones en el futuro.
    rcuad=rs**2

    #Radio por n\'umero de onda. Este es para ahorrar operaciones en el futuro.
    rk=np.zeros((n,2))

    rk[:,0]=rs*k[0]

    rango=np.arange(0,n)

    rk[rango,1]=rs[rango]*k[rango+1]


    #Comprobaci\'on de rk


    LMas1=L+1
    #Arreglos con 0 para rellenarlos en un pr\'oximo for. 
    jn=np.zeros((LMas1,n,2))  #Array de floats 64
    jnp=np.zeros((LMas1,n,2)) #Array de floats 64
    hn=np.zeros((LMas1,n,2),dtype=np.complex128)  #Array de complex 128
    hnp=np.zeros((LMas1,n,2),dtype=np.complex128) #Array de complex 128
    rango=np.asarray(range(0,LMas1))
    for i in range(0,2):
        for j in range(0,n):
            aux00=sci.spherical_jn(rango,rk[j,i])
            aux01=sci.spherical_jn(rango,rk[j,i],derivative=True)
            jn[:,j,i]=aux00 #Funciones de bessel de primer tipo en rk[j,i]
            jnp[:,j,i]=aux01 #Derivada de las funciones de bessel de primer tipo en rk[j,i]
            hn[:,j,i]=aux00+1j*sci.spherical_yn(rango,rk[j,i])  #Funciones de hankel(1) en rk[j,i]
            hnp[:,j,i]=aux01+1j*sci.spherical_yn(rango,rk[j,i],derivative=True) #Derivada de las funciones de hankel(1) en rk[j,i]

    del rk #Borrado de la variable que ya no se ocupa

    if dd==3:
        for ii in range(0,len(x1)):
            for jj in range(0,len(y1)):
                theta=0.0
                x=x1[ii]
                y=y1[jj]
                z=0.

                xvectorcart=np.asarray([x,y,z])+centro
                Esfera=0
                num=0


                while num<n:
                    aux=xvectorcart-ps[num]
                    raux=np.linalg.norm(aux)
                    if raux<rs[num]:
                        Esfera=num+1
                        num=n
                    num=num+1

                u=np.complex128(0.)

                if Esfera==0:
                    for num in range(0,n):
                        aux=xvectorcart-ps[num]
                        r=np.linalg.norm(aux)
                        x=aux[0]
                        y=aux[1]
                        z=aux[2]

                        theta=np.arccos(z/r)
                        phi=np.arctan2(y,x)

                        haux=sci.spherical_jn(rango,r*k[0])+1j*sci.spherical_yn(rango,r*k[0])
                        contador=0
                        utemp=np.complex128(0.)
                        for l in range(0,L+1):
                            temp=np.complex128(0.)
                            esfericostemp=sci.sph_harm(range(-l,l+1),l,phi,theta)
                            for esferico in esfericostemp:
                                temp=temp+(Resultado[contador+2*(L+1)**2*num]*k[0]*jnp[l,num,0]+Resultado[contador+(L+1)**2*(2*num+1)]*jn[l,num,0])*esferico
                                contador=contador+1
                            temp=haux[l]*temp
                            utemp=utemp+temp

                        utemp=rcuad[num]*utemp
                        u=u+utemp
                    #print u
                    pgraf[jj,ii]=np.exp(1j*k[0]*xvectorcart[2]-1j*omega*t)+1j*k[0]*u*np.exp(-1j*omega*t)
                else:                    
                    aux=xvectorcart-ps[Esfera-1]
                    r=np.linalg.norm(aux)
                    x=aux[0]
                    y=aux[1]
                    z=aux[2]

                    theta=0.0
                    if z>0:
                        theta=np.arctan(np.sqrt(x**2+y**2)/z)
                    elif z<0:
                        theta=np.pi+np.arctan(np.sqrt(x**2+y**2)/z)
                    else:
                        theta=np.pi/2.

                    phi=0.0
                    phi=np.arctan2(y,x)

                    jaux=np.asarray(sci.spherical_jn(rango,r*k[Esfera]))

                    contador=0
                    #print u
                    for l in range(0,L+1):

                        temp=np.complex128(0.)
                        #print 't '+str(temp)
                        esfericostemp=sci.sph_harm(range(-l,l+1),l,phi,theta)
                        #print esfericostemp
                        for esferico in esfericostemp:
                            temp=temp+(-Resultado[contador+2*(L+1)**2*(Esfera-1+n)]*k[Esfera]*hnp[l,Esfera-1,1]+Resultado[contador+(L+1)**2*(2*(Esfera-1)+1+2*n)]*hn[l,Esfera-1,1])*esferico
                            contador=contador+1

                        temp=jaux[l]*temp
                        #print temp
                        u=u+temp
                        #print u
                    pgraf[jj,ii]=1j*k[Esfera]*rcuad[Esfera-1]*u*np.exp(-1j*omega*t)
        plt.figure(facecolor='w')
        if ConCotas:
            plt.imshow(np.real(pgraf), vmin=mini, vmax=maxi, origin ='lower',extent=[-ancho/2+centro[0],ancho/2+centro[0],-alto/2+centro[1],alto/2+centro[1]])
        else:
            plt.imshow(np.real(pgraf), origin ='lower',extent=[-ancho/2+centro[0],ancho/2+centro[0],-alto/2+centro[1],alto/2+centro[1]])
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.colorbar()
    elif dd==2:
        for ii in range(0,len(x1)):
            for jj in range(0,len(y1)):
                theta=0.0
                x=x1[ii]
                y=0.
                z=y1[jj]
                xvectorcart=np.asarray([x,y,z])+centro

                Esfera=0
                num=0


                while num<n:
                    aux=xvectorcart-ps[num]
                    raux=np.linalg.norm(aux)
                    if raux<rs[num]:
                        Esfera=num+1
                        num=n
                    num=num+1

                u=np.complex128(0.)

                if Esfera==0:
                    for num in range(0,n):
                        aux=xvectorcart-ps[num]
                        r=np.linalg.norm(aux)
                        x=aux[0]
                        y=aux[1]
                        z=aux[2]

                        theta=0.0
                        if z>0:
                            theta=np.arctan(np.sqrt(x**2+y**2)/z)
                        elif z<0:
                            theta=np.pi+np.arctan(np.sqrt(x**2+y**2)/z)
                        else:
                            theta=np.pi/2.
                        phi=0.0
                        phi=np.arctan2(y,x)

                        haux=sci.spherical_jn(rango,r*k[0])+1j*sci.spherical_yn(rango,r*k[0])
                        contador=0
                        utemp=np.complex128(0.)
                        for l in range(0,L+1):
                            temp=np.complex128(0.)
                            esfericostemp=sci.sph_harm(range(-l,l+1),l,phi,theta)
                            for esferico in esfericostemp:
                                temp=temp+(Resultado[contador+2*(L+1)**2*num]*k[0]*jnp[l,num,0]+Resultado[contador+(L+1)**2*(2*num+1)]*jn[l,num,0])*esferico
                                contador=contador+1
                            temp=haux[l]*temp
                            utemp=utemp+temp

                        utemp=rcuad[num]*utemp
                        u=u+utemp
                    #print u
                    pgraf[jj,ii]=np.exp(1j*k[0]*xvectorcart[2]-1j*omega*t)+1j*k[0]*u*np.exp(-1j*omega*t)
                else:                    
                    aux=xvectorcart-ps[Esfera-1]
                    r=np.linalg.norm(aux)
                    x=aux[0]
                    y=aux[1]
                    z=aux[2]

                    theta=0.0
                    if z>0:
                        theta=np.arctan(np.sqrt(x**2+y**2)/z)
                    elif z<0:
                        theta=np.pi+np.arctan(np.sqrt(x**2+y**2)/z)
                    else:
                        theta=np.pi/2.

                    phi=0.0
                    phi=np.arctan2(y,x)

                    jaux=np.asarray(sci.spherical_jn(rango,r*k[Esfera]))

                    contador=0
                    #print u
                    for l in range(0,L+1):

                        temp=np.complex128(0.)
                        #print 't '+str(temp)
                        esfericostemp=sci.sph_harm(range(-l,l+1),l,phi,theta)
                        #print esfericostemp
                        for esferico in esfericostemp:
                            temp=temp+(-Resultado[contador+2*(L+1)**2*(Esfera-1+n)]*k[Esfera]*hnp[l,Esfera-1,1]+Resultado[contador+(L+1)**2*(2*(Esfera-1)+1+2*n)]*hn[l,Esfera-1,1])*esferico
                            contador=contador+1


                        temp=jaux[l]*temp
                        #print temp
                        u=u+temp
                        #print u
                    pgraf[jj,ii]=1j*k[Esfera]*rcuad[Esfera-1]*u*np.exp(-1j*omega*t)
        plt.figure(facecolor='w')
        if ConCotas:
            plt.imshow(np.real(pgraf), vmin=mini, vmax=maxi, origin ='lower',extent=[-ancho/2+centro[0],ancho/2+centro[0],-alto/2+centro[2],alto/2+centro[2]])
        else:
            plt.imshow(np.real(pgraf), origin ='lower',extent=[-ancho/2+centro[0],ancho/2+centro[0],-alto/2+centro[2],alto/2+centro[2]])
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.colorbar()
    else:
        for ii in range(0,len(x1)):
            for jj in range(0,len(y1)):
                theta=0.0
                x=0.
                y=x1[ii]
                z=y1[jj]
                xvectorcart=np.asarray([x,y,z])+centro

                Esfera=0
                num=0


                while num<n:
                    aux=xvectorcart-ps[num]
                    raux=np.linalg.norm(aux)
                    if raux<rs[num]:
                        Esfera=num+1
                        num=n
                    num=num+1

                u=np.complex128(0.)

                if Esfera==0:
                    for num in range(0,n):
                        aux=xvectorcart-ps[num]
                        r=np.linalg.norm(aux)
                        x=aux[0]
                        y=aux[1]
                        z=aux[2]

                        theta=0.0
                        if z>0:
                            theta=np.arctan(np.sqrt(x**2+y**2)/z)
                        elif z<0:
                            theta=np.pi+np.arctan(np.sqrt(x**2+y**2)/z)
                        else:
                            theta=np.pi/2.
                        phi=0.0
                        phi=np.arctan2(y,x)

                        haux=sci.spherical_jn(rango,r*k[0])+1j*sci.spherical_yn(rango,r*k[0])
                        contador=0
                        utemp=np.complex128(0.)
                        for l in range(0,L+1):
                            temp=np.complex128(0.)
                            esfericostemp=sci.sph_harm(range(-l,l+1),l,phi,theta)
                            for esferico in esfericostemp:
                                temp=temp+(Resultado[contador+2*(L+1)**2*num]*k[0]*jnp[l,num,0]+Resultado[contador+(L+1)**2*(2*num+1)]*jn[l,num,0])*esferico
                                contador=contador+1
                            temp=haux[l]*temp
                            utemp=utemp+temp

                        utemp=rcuad[num]*utemp
                        u=u+utemp
                    #print u
                    pgraf[jj,ii]=np.exp(1j*k[0]*xvectorcart[2]-1j*omega*t)+1j*k[0]*u*np.exp(-1j*omega*t)
                else:                    
                    aux=xvectorcart-ps[Esfera-1]
                    r=np.linalg.norm(aux)
                    x=aux[0]
                    y=aux[1]
                    z=aux[2]

                    theta=0.0
                    if z>0:
                        theta=np.arctan(np.sqrt(x**2+y**2)/z)
                    elif z<0:
                        theta=np.pi+np.arctan(np.sqrt(x**2+y**2)/z)
                    else:
                        theta=np.pi/2.

                    phi=0.0
                    phi=np.arctan2(y,x)

                    jaux=np.asarray(sci.spherical_jn(rango,r*k[Esfera]))

                    contador=0
                    #print u
                    for l in range(0,L+1):

                        temp=np.complex128(0.)
                        #print 't '+str(temp)
                        esfericostemp=sci.sph_harm(range(-l,l+1),l,phi,theta)
                        #print esfericostemp
                        for esferico in esfericostemp:
                            temp=temp+(-Resultado[contador+2*(L+1)**2*(Esfera-1+n)]*k[Esfera]*hnp[l,Esfera-1,1]+Resultado[contador+(L+1)**2*(2*(Esfera-1)+1+2*n)]*hn[l,Esfera-1,1])*esferico
                            contador=contador+1

                        temp=jaux[l]*temp
                        #print temp
                        u=u+temp
                        #print u
                    pgraf[jj,ii]=1j*k[Esfera]*rcuad[Esfera-1]*u*np.exp(-1j*omega*t)
        plt.figure(facecolor='w')
        if ConCotas:
            plt.imshow(np.real(pgraf), vmin=mini, vmax=maxi, origin ='lower',extent=[-ancho/2+centro[0],ancho/2+centro[0],-alto/2+centro[2],alto/2+centro[2]])
        else:
            plt.imshow(np.real(pgraf), origin ='lower',extent=[-ancho/2+centro[1],ancho/2+centro[1],-alto/2+centro[2],alto/2+centro[2]])
        plt.xlabel('y [m]')
        plt.ylabel('z [m]')
        plt.colorbar()

def ErrorSalto(k,aj,radio,punto,trazas0j,trazasj,L,Lo):
    Lmas1al2=len(trazas0j)/2
    trazas0j=np.transpose(np.matrix(trazas0j))
    trazasj=np.transpose(np.matrix(trazasj))
    X=np.asmatrix(np.diag(np.concatenate((np.ones(Lmas1al2),-aj*np.ones(Lmas1al2)))))
    VectorLadoIzquierdo=np.asarray(-X*trazas0j+trazasj)
    suma1 = 0.
    contador=0
    extra1=0.
    rango=np.arange(0,Lo+1)
    jn=sci.spherical_jn(rango,radio*k)
    for l in range(0,L+1):
        for m in range(-l,l+1):
            if m!=0:
                suma1=suma1+np.linalg.norm(VectorLadoIzquierdo[contador])**2
                extra1=extra1+np.linalg.norm(VectorLadoIzquierdo[contador])**2
            else:
                suma1=suma1+np.linalg.norm(VectorLadoIzquierdo[contador]-2*np.sqrt(np.pi)*np.exp(1j*k*punto[2])*1j**l*np.sqrt(2*l+1)*jn[l])**2
            contador=contador+1

    jnp=sci.spherical_jn(rango,radio*k,derivative=True)
    suma2 = 0.
    extra2=0.
    for l in range(0,L+1):
        for m in range(-l,l+1):
            if m!=0:
                suma2=suma2+np.linalg.norm(VectorLadoIzquierdo[contador])**2
                extra2=extra2+np.linalg.norm(VectorLadoIzquierdo[contador])**2
            else:
                suma2=suma2+np.linalg.norm(VectorLadoIzquierdo[contador]-2*-aj*-k*np.sqrt(np.pi)*np.exp(1j*k*punto[2])*1j**l*np.sqrt(2*l+1)*jnp[l])**2
            contador=contador+1

    return [np.sqrt(suma1)*radio,np.sqrt(suma2)*radio]
'''


# Reconstruccion funci\'on u
def ueuinc(xvectorcart, Resultado, t, omega, listaParametros, L):
    [k, aes, ps, rs, n] = listaParametros
    '''
    print '\n rs'
    print rs
    '''
    # Radio al cuadrado. Este es para ahorrar operaciones en el futuro.
    rcuad = rs ** 2
    '''
    print '\n rcuad'
    print rcuad
    '''

    # Radio por n\'umero de onda. Este es para ahorrar operaciones en el futuro.
    rk = np.zeros((n, 2))

    rk[:, 0] = rs * k[0]

    rango = np.arange(0, n)

    rk[rango, 1] = rs[rango] * k[rango + 1]

    # Comprobaci\'on de rk
    '''
    print '\n Valores de r por k'
    print 'r por k ' + str(rk)
    print 'r ' + str(rs)
    print 'k ' + str(k)
    '''
    # Funciones esfericas de Bessel y hankel precalculadas en rk hasta L
    jn = np.zeros((L + 1, n, 2), dtype=np.complex128)
    jnp = np.zeros((L + 1, n, 2), dtype=np.complex128)
    hn = np.zeros((L + 1, n, 2), dtype=np.complex128)
    hnp = np.zeros((L + 1, n, 2), dtype=np.complex128)

    for i in range(0, 2):
        for j in range(0, n):
            aux = sci.sph_jn(L, rk[j, i])
            aux00 = np.asarray(aux[0])
            aux01 = np.asarray(aux[1])
            del aux
            jn[:, j, i] = aux00  # Funciones de bessel en rk[j,i]
            jnp[:, j, i] = aux01  # Derivada de las funciones de bessel en rk[j,i]
            aux1 = sci.sph_yn(L, rk[j, i])
            hn[:, j, i] = aux00 + 1j * np.asarray(aux1[0])  # Funciones de hankel(1) en rk[j,i]
            hnp[:, j, i] = aux01 + 1j * np.asarray(aux1[1])  # Derivada de las funciones de hankel(1) en rk[j,i]
    del rk  # Borrado de la variable que ya no se ocupa

    Esfera = 0
    num = 0

    while num < n:
        aux = xvectorcart - ps[num]
        raux = np.linalg.norm(aux)
        if raux < rs[num]:
            Esfera = num + 1
            num = n
        num = num + 1

    u = np.complex128(0.)

    if Esfera == 0:
        for num in range(0, n):
            aux = xvectorcart - ps[num]
            r = np.linalg.norm(aux)
            x = aux[0]
            y = aux[1]
            z = aux[2]

            theta = 0.0
            if z > 0:
                theta = np.arctan(np.sqrt(x ** 2 + y ** 2) / z)
            elif z < 0:
                theta = np.pi + np.arctan(np.sqrt(x ** 2 + y ** 2) / z)
            else:
                theta = np.pi / 2.
            phi = np.arctan2(y, x)
            '''
            if x>0 and y>0:
                print '1'
                phi=np.arctan(y/x)
            elif x>0 and y<0:
                phi=np.pi*2.+np.arctan(y/x)
                print '2'
            elif x==0:
                print '3'
                phi=np.pi/2.*np.sign(y)
            else:
                print '4'
                phi=np.pi+np.arctan(y/x)
            '''

            haux = sci.sph_jn(L, r * k[0])[0] + 1j * sci.sph_yn(L, r * k[0])[0]
            contador = 0
            utemp = np.complex128(0.)
            for l in range(0, L + 1):
                temp = np.complex128(0.)
                esfericostemp = sci.sph_harm(range(-l, l + 1), l, phi, theta)
                for esferico in esfericostemp:
                    temp = temp + (Resultado[contador + 2 * (L + 1) ** 2 * num] * k[0] * jnp[l, num, 0] + Resultado[
                        contador + (L + 1) ** 2 * (2 * num + 1)] * jn[l, num, 0]) * esferico
                    contador = contador + 1
                temp = haux[l] * temp
                utemp = utemp + temp

            utemp = rcuad[num] * utemp
            u = u + utemp
        # print u
        return np.exp(1j * k[0] * xvectorcart[2] - 1j * omega * t) + 1j * k[0] * u * np.exp(-1j * omega * t)
    else:
        aux = xvectorcart - ps[Esfera - 1]
        r = np.linalg.norm(aux)
        x = aux[0]
        y = aux[1]
        z = aux[2]

        theta = 0.0
        if z > 0:
            theta = np.arctan(np.sqrt(x ** 2 + y ** 2) / z)
        elif z < 0:
            theta = np.pi + np.arctan(np.sqrt(x ** 2 + y ** 2) / z)
        else:
            theta = np.pi / 2.

        phi = 0.0
        phi = np.arctan2(y, x)
        '''
        if x>0 and y>0:
            print '1'
            phi=np.arctan(y/x)
        elif x>0 and y<0:
            phi=np.pi*2.+np.arctan(y/x)
            print '2'
        elif x==0:
            print '3'
            phi=np.pi/2.*np.sign(y)
        else:
            print '4'
            phi=np.pi+np.arctan(y/x)
        '''
        jaux = np.asarray(sci.sph_jn(L, r * k[Esfera])[0])

        contador = 0
        # print u
        for l in range(0, L + 1):

            temp = np.complex128(0.)
            # print 't '+str(temp)
            esfericostemp = sci.sph_harm(range(-l, l + 1), l, phi, theta)
            # print esfericostemp
            for esferico in esfericostemp:
                temp = temp + (-Resultado[contador + 2 * (L + 1) ** 2 * (Esfera - 1 + n)] * k[Esfera] * hnp[
                    l, Esfera - 1, 1] + Resultado[contador + (L + 1) ** 2 * (2 * (Esfera - 1) + 1 + 2 * n)] * hn[
                                   l, Esfera - 1, 1]) * esferico
                contador = contador + 1

            '''
            print hnp[l,Esfera-1,1]
            '''
            temp = jaux[l] * temp
            # print temp
            u = u + temp
            # print u
        return 1j * k[Esfera] * rcuad[Esfera - 1] * u * np.exp(-1j * omega * t)

