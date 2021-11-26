import numpy as np
import matplotlib.pyplot as plt

import pyshtools as psh


def drawing_cut_with_real_spherical_harmonics_one_sphere_laplace_kernel(dd, centro, ancho, alto, interancho, interlargo,
                                                                        coefficients, r, L, exterior):
    n = 1
    ps = [np.asarray([0., 0., 0.])]
    rs = np.asarray([r])
    # Inputs:
    ##dd: int. Indica si el dibujo se va a hacer en un corte paralelo al plano:
    # xy si dd==1, xz si dd==2, yz si dd=3. Indicates if the drawing is a
    # parallel cut of the plane: xy if dd==1, xz if dd==2, yz if dd==3.
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
    ## coefficients: coefficients of the spherical harmonic expansion
    ##L: int. Orden m\'aximo de los arm\'nicos esf\'ericos que se utilizan para
    # discretizar las trazas. (L+1)**2 es la cantidad total de arm\'onicos
    # esf\'ericos que se utilizan para discretizar una traza. (Maximum order of
    # spherical harmonics that are used to discretize the traces. (L+1)**2 is
    # the total number of spherical harmonics that are used to discretize a
    # trace).

    x1 = np.linspace(-ancho / 2, ancho / 2, interancho)
    y1 = np.linspace(-alto / 2, alto / 2, interlargo)
    pgraf = np.zeros((len(y1), len(x1)))

    LMas1 = L + 1
    eles = np.arange(0, L + 1)
    l2_1 = 2 * eles + 1
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

                u = 0.

                if Esfera == 0:
                    for num in np.arange(0, n):
                        aux = xvectorcart - ps[num]
                        r = np.linalg.norm(aux)
                        x = aux[0]
                        y = aux[1]
                        z = aux[2]

                        theta = np.arccos(z / r)
                        phi = np.arctan2(y, x)

                        utemp = 0.
                        spherical_harmonic = psh.expand.spharm(L, theta, phi, normalization='ortho', csphase=-1, degrees=False)
                        l2_1 = 0
                        for l in eles:
                            utemp = utemp + spherical_harmonic[0, l, 0]*((r*coefficients[(l2_1 + l) + (L + 1) ** 2 * (2 * num + 1)] + l*coefficients[(l2_1 + l) + 2 * (L + 1) ** 2 * num])/(2*l+1))
                            for m in np.arange(1, l+1):
                                utemp = utemp + spherical_harmonic[0, l, m] * ((r * coefficients[
                                    (l2_1 + l+m) + (L + 1) ** 2 * (2 * num + 1)] + l * coefficients[(l2_1 + l+m) + 2 * (
                                            L + 1) ** 2 * num]) / (2*l+1))
                                utemp = utemp + spherical_harmonic[1, l, m] * ((r * coefficients[
                                    (l2_1 + l - m) + (L + 1) ** 2 * (2 * num + 1)] + l * coefficients[
                                                                                   (l2_1 + l - m) + 2 * (
                                                                                           L + 1) ** 2 * num]) / (2*l+1))
                            l2_1=l2_1+ 2*l+1
                        del spherical_harmonic
                        u = u + utemp
                    pgraf[jj, ii] = u + exterior(np.asarray([x, y, z]))
                else:
                    aux = xvectorcart - ps[Esfera - 1]
                    r = np.linalg.norm(aux)
                    x = aux[0]
                    y = aux[1]
                    z = aux[2]

                    theta = np.arccos(z / r)

                    phi = np.arctan2(y, x)

                    utemp = 0.
                    spherical_harmonic = psh.expand.spharm(L, theta, phi, normalization='ortho', csphase=-1, degrees=False)
                    l2_1 = 0
                    for l in eles:
                        utemp = utemp + spherical_harmonic[0, l, 0] * ((r * coefficients[
                            (l2_1 + l) + (L + 1) ** 2 * (2 * (Esfera-1 + n) + 1)] + (l+1) * coefficients[
                                                                            (l2_1 + l) + 2 * (L + 1) ** 2 * (Esfera-1 + n)]) /
                                                                       (2*l+1))
                        for m in np.arange(1, l + 1):
                            utemp = utemp + spherical_harmonic[0, l, m] * ((r * coefficients[
                                (l2_1 + l + m) + (L + 1) ** 2 * (2 * (Esfera-1 + n) + 1)] + (l+1) * coefficients[(l2_1 + l + m) + 2 * (
                                    L + 1) ** 2 * (Esfera-1 + n)]) / (2*l+1))
                            utemp = utemp + spherical_harmonic[1, l, m] * ((r * coefficients[
                                (l2_1 + l - m) + (L + 1) ** 2 * (2 * (Esfera-1 + n) + 1)] + (l+1) * coefficients[
                                                                                (l2_1 + l - m) + 2 * (
                                                                                        L + 1) ** 2 * (Esfera-1 + n)]) / (2*l+1))
                        l2_1 = l2_1 + 2*l+1
                    del spherical_harmonic
                    u = u + utemp
                    pgraf[jj, ii] = u
        return pgraf
    elif dd == 2:
        for ii in np.arange(0, len(x1)):
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

                u = 0.

                if Esfera == 0:
                    for num in np.arange(0, n):
                        aux = xvectorcart - ps[num]
                        r = np.linalg.norm(aux)
                        x = aux[0]
                        y = aux[1]
                        z = aux[2]

                        theta = np.arccos(z / r)
                        phi = np.arctan2(y, x)

                        utemp = 0.
                        spherical_harmonic = psh.expand.spharm(L, theta, phi, normalization='ortho', csphase=-1, degrees=False)

                        l2_1 = 0
                        for l in eles:
                            utemp = utemp + spherical_harmonic[0, l, 0] * ((r * coefficients[
                                (l2_1 + l) + (L + 1) ** 2 * (2 * num + 1)] + l * coefficients[
                                                                                (l2_1 + l) + 2 * (L + 1) ** 2 * num]) /
                                                                           (2*l+1))
                            for m in np.arange(1, l + 1):
                                utemp = utemp + spherical_harmonic[0, l, m] * ((r * coefficients[
                                    (l2_1 + l + m) + (L + 1) ** 2 * (2 * num + 1)] + l * coefficients[
                                                                                    (l2_1 + l + m) + 2 * (
                                                                                            L + 1) ** 2 * num]) / (2*l+1))
                                utemp = utemp + spherical_harmonic[1, l, m] * ((r * coefficients[
                                    (l2_1 + l - m) + (L + 1) ** 2 * (2 * num + 1)] + l * coefficients[
                                                                                    (l2_1 + l - m) + 2 * (
                                                                                            L + 1) ** 2 * num]) / (2*l+1))
                            l2_1 = l2_1 +  2*l+1
                        del spherical_harmonic
                        u = u + utemp
                    pgraf[jj, ii] = u + exterior(np.asarray([x, y, z]))
                else:
                    aux = xvectorcart - ps[Esfera - 1]
                    r = np.linalg.norm(aux)
                    x = aux[0]
                    y = aux[1]
                    z = aux[2]

                    theta = np.arccos(z / r)

                    phi = np.arctan2(y, x)

                    utemp = 0.
                    spherical_harmonic = psh.expand.spharm(L, theta, phi, normalization='ortho', csphase=-1, degrees=False)

                    l2_1 = 0
                    for l in eles:
                        utemp = utemp + spherical_harmonic[0, l, 0]*((r*coefficients[(l2_1 + l) + (L + 1) ** 2 * (2 * (Esfera-1 + n) + 1)] + (l+1)*coefficients[(l2_1 + l) + 2 * (L + 1) ** 2 * (Esfera-1 + n)])/(2*l+1))
                        for m in np.arange(1, l+1):
                            utemp = utemp + spherical_harmonic[0, l, m] * ((r * coefficients[
                                (l2_1 + l+m) + (L + 1) ** 2 * (2 * (Esfera-1 + n) + 1)] + (l+1) * coefficients[(l2_1 + l+m) + 2 * (
                                        L + 1) ** 2 * (Esfera-1 + n)]) / (2*l+1))
                            utemp = utemp + spherical_harmonic[1, l, m] * ((r * coefficients[
                                (l2_1 + l - m) + (L + 1) ** 2 * (2 * (Esfera-1 + n) + 1)] + (l+1) * coefficients[
                                                                               (l2_1 + l - m) + 2 * (
                                                                                       L + 1) ** 2 * (Esfera-1 + n)]) / (2*l+1))
                        l2_1 = l2_1 +  2*l+1
                    del spherical_harmonic
                    u = u + utemp
                    pgraf[jj, ii] = u
        return pgraf
    else:
        for ii in np.arange(0, len(x1)):
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

                u = 0.

                if Esfera == 0:
                    for num in np.arange(0, n):
                        aux = xvectorcart - ps[num]
                        r = np.linalg.norm(aux)
                        x = aux[0]
                        y = aux[1]
                        z = aux[2]

                        theta = np.arccos(z / r)
                        phi = np.arctan2(y, x)

                        utemp = 0.
                        spherical_harmonic = psh.expand.spharm(L, theta, phi, normalization='ortho', csphase=-1, degrees=False)
                        l2_1 = 0
                        for l in eles:
                            utemp = utemp + spherical_harmonic[0, l, 0]*((r*coefficients[(l2_1 + l) + (L + 1) ** 2 * (2 * num + 1)] + l*coefficients[(l2_1 + l) + 2 * (L + 1) ** 2 * num])/(2*l+1))
                            for m in np.arange(1, l+1):
                                utemp = utemp + spherical_harmonic[0, l, m] * ((r * coefficients[
                                    (l2_1 + l+m) + (L + 1) ** 2 * (2 * num + 1)] + l * coefficients[(l2_1 + l+m) + 2 * (
                                            L + 1) ** 2 * num]) / (2*l+1))
                                utemp = utemp + spherical_harmonic[1, l, m] * ((r * coefficients[
                                    (l2_1 + l - m) + (L + 1) ** 2 * (2 * num + 1)] + l * coefficients[
                                                                                   (l2_1 + l - m) + 2 * (
                                                                                           L + 1) ** 2 * num]) / (2*l+1))
                            l2_1 = l2_1 + 2*l+1
                        del spherical_harmonic
                        u = u + utemp
                    pgraf[jj, ii] = u + exterior(np.asarray([x, y, z]))
                else:
                    aux = xvectorcart - ps[Esfera - 1]
                    r = np.linalg.norm(aux)
                    x = aux[0]
                    y = aux[1]
                    z = aux[2]

                    theta = np.arccos(z / r)

                    phi = np.arctan2(y, x)

                    utemp = 0.
                    spherical_harmonic = psh.expand.spharm(L, theta, phi, normalization='ortho', csphase=-1, degrees=False)
                    l2_1 = 0
                    for l in eles:
                        utemp = utemp + spherical_harmonic[0, l, 0] * ((r * coefficients[
                            (l2_1 + l) + (L + 1) ** 2 * (2 * (Esfera-1 + n) + 1)] + (l+1) * coefficients[
                                                                            (l2_1 + l) + 2 * (L + 1) ** 2 * (Esfera-1 + n)]) /
                                                                       (2*l+1))
                        for m in np.arange(1, l + 1):
                            utemp = utemp + spherical_harmonic[0, l, m] * ((r * coefficients[
                                (l2_1 + l + m) + (L + 1) ** 2 * (2 * (Esfera-1 + n) + 1)] + (l+1) * coefficients[(l2_1 + l + m) + 2 * (
                                    L + 1) ** 2 * (Esfera-1 + n)]) / (2*l+1))
                            utemp = utemp + spherical_harmonic[1, l, m] * ((r * coefficients[
                                (l2_1 + l - m) + (L + 1) ** 2 * (2 * (Esfera-1 + n) + 1)] + l * coefficients[
                                                                                (l2_1 + l - m) + 2 * (
                                                                                        L + 1) ** 2 * (Esfera-1 + n)]) / (2*l+1))
                        l2_1 = l2_1 + 2*l+1
                    del spherical_harmonic
                    u = u + utemp
                    pgraf[jj, ii] = u
        return pgraf
