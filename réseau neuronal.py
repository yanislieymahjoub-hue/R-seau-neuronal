from math import exp

def réseau_neuronal(entrées,poids,biais):
    res = entrées[0]*poids[0] + entrées[1]*poids[1] + biais
    return 1 / (1 + exp(-res))

A = réseau_neuronal ([1.0,2.0],[8.0,9.0],7.0)



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

neurones2 = [
    ([8.0, 7.0], 7.0),
    ([1.0, -2.0], 0.5),
    ([-3.0, 4.0], -1.0),]


def IA (entrées, couche1, couche2):
    fe = couche (entrées, couche1)
    fu = couche (fe, couche2)
    return fu

C = IA([0.0,1.0],neurones, neurones2)


def cout (prédiction, attendu):
    res = (prédiction - attendu)**2
    return res

D = cout (0.1,1)


def gradient (w,x,attendu,epsilon) : 
    prediction = w * x
    cout1 = cout(prediction, attendu)
    new_prediction = (w + epsilon) * x
    cout2 = cout(new_prediction, attendu)
    return (cout2 - cout1) / epsilon

E = gradient (0.5, 1.0, 1.0, 0.0001)
print (E)

def entrainement(w,x,attendu,epsilon,lr,nb_itérations):
    for i in range (nb_itérations) :
        grad = gradient (w,x,attendu,epsilon)
        w = w - (lr*grad)
    return w

F = entrainement (0.5,1.0,1.0,0.0001,0.1,5)
print (F)
