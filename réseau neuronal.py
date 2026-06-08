from math import exp

def réseau_neuronal(entrées,poids,biais):
    res = entrées[0]*poids[0] + entrées[1]*poids[1] + biais
    return 1 / (1 + exp(-res))

A = réseau_neuronal ([1.0,2.0],[8.0,9.0],7.0)
print(A)


def couche(entrées,neurone):
    res = []
    for i,j in neurone :
        G = réseau_neuronal (entrées,i,j)
        res.append(G)
    return res


neurones = [
    ([8.0, 9.0], 7.0),
    ([1.0, -2.0], 0.5),
    ([-3.0, 4.0], -1.0),]


B = couche ([0.0,1.0],neurones)
print (B)
