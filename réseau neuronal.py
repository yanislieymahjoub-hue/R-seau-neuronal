from math import exp

def réseau_neuronal(entrées,poids,biais):
    res = entrées[0]*poids[0] + entrées[1]*poids[1] + biais
    return 1 / (1 + exp(-res))

A = réseau_neuronal ([1.0,2.0],[8.0,9.0],7.0)
print(A)