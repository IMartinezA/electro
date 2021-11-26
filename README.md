# Sobre el final del nombre de los archivos  

Los que terminan con Basic1 solo necesitan los paquetes:  
- numpy
- scipy
- matplotlib
o quizás menos  

Los que terminan con Linux1 necesitan los mismos paquetes que Basic1 y el paquete:
- pyshtools (no funciona en windows)

# Archivos que solo tienen funciones
- ElectropermeabilizationEnvBasic1 (requiere el archivo LaplaceSpheresFunctionsEnvBasic1)
- LaplaceSpheresFunctionsEnvBasic1
- GraficosFigurasEnvLinux1
- HelmholtzSpheresFunctionsEnvLinux1

# Archivos que usan los archivos con funciones y hay algún tipo de output:

## ElectropermeabilizationTestEnvLinux
(requiere ElectropermeabilizationEnvBasic1)  
(Extremadamente lento, no recuerdo lo que hice con los parámetros así que puede que salgan dibujos extraños)  
Actualmente lo uso para testear la versión con tiempo lineal.

## LaplaceSpheresTestEnvLinux1
(requiere LaplaceSpheresFunctionsEnvBasic1 y HelmholtzSpheresFunctionsEnvLinux1)
- Lo usé para hacer algunos testeos considerando solo el MTF estático.

## PicturesEnvLinux1
(requiere GraficosFigurasEnvLinux1, LaplaceSpheresFunctionsEnvBasic1 y HelmholtzSpheresFunctionsEnvLinux1)  
Extremadamente lento, lo usé para ver los campos en el volumen (en un corte en una sección). Me sirvió para encontrar algunos detalles importantes.
