import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([0,1,1,0])

W1 = np.random.randn(2, 2)
b1 = np.zeros(2)
W2 = np.random.randn(1, 2)
b2 = np.zeros(1)

def sigmo(x):
    return 1 / (1 + np.exp(-x))

lr = 0.5

for i in range(10000):
    z1 = X @ W1.T + b1
    a1 = sigmo(z1)
    z2 = a1 @ W2.T + b2
    a2 = sigmo(z2)

    loss = np.mean((a2 - Y.reshape(4, 1))**2)

    dz2 = (a2 - Y.reshape(4,1)) * a2 * (1 - a2)
    dW2 = dz2.T @ a1
    db2 = np.sum(dz2, axis=0)

    dz1 = (dz2 @ W2) * a1 * (1 - a1)
    dW1 = dz1.T @ X
    db1 = np.sum(dz1, axis=0)

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if i % 1000 == 0:
        print(f"iter {i} — loss : {loss:.4f}")

print("\nEntraînement terminé !\n")

while True:
    print("Entre deux valeurs (0 ou 1) pour tester le réseau XOR.")
    a = int(input("Valeur 1 : "))
    b = int(input("Valeur 2 : "))

    entree = np.array([a, b])
    z1 = entree @ W1.T + b1
    a1 = sigmo(z1)
    z2 = a1 @ W2.T + b2
    a2 = sigmo(z2)
    prediction = int(np.round(a2[0]))

    print(f"\n{a} XOR {b} = {prediction}\n")

    continuer = input("Rejouer ? (o/n) : ")
    if continuer.lower() != "o":
        break