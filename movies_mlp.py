import csv
import numpy as np

#lendo o dataset
data = []
try:
    with open("movies_metadata.csv", encoding="utf-8") as archive:
        reader = csv.reader(archive)
        next(reader)
        for line in reader:
            data.append([float(line[0]), float(line[1]), float(line[2])])
except:
    print("Error opening file.")
    data = [
        [5.0, 90], #media dos usuarios e duração 
        [8.7, 115],
        [4.7, 120],
        [7.2, 130],
    ]

x = np.array(data)

#saída
y = np.array([[0], [1], [1], [1]])

#pesos
np.random.seed(0)
w1 = np.random.rand(3, 4)
w2 = np.random.rand(4, 1)

#função sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#função derivada
def derivative_sigmoid(x):
    return x * (1 - x)

#treinando
for epoch in range(100):
    z1 = np.dot(x, w1)
    a1 = sigmoid(z1)

    z2 = np.dot(a1, w2)
    exit = sigmoid(z2)

    error = y - exit
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Average error: {np.mean(np.abs(error)):.4f}")

# Backprop
d_exit = error * derivative_sigmoid(exit)
d_w2 = np.dot(a1.T, d_exit)

d_a1 = np.dot(d_exit, w2.T)
d_z1 = d_a1 * derivative_sigmoid(a1)
d_w1 = np.dot(x.T, d_z1)


w2 += d_w2 * 0.1
w1 += d_w1 * 0.1

# Previsãol
print("\nFinal network output: ")
print(exit)

