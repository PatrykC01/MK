import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Wczytanie i przygotowanie danych ---
try:
    df = pd.read_csv('war9.csv', sep=';', decimal=',')
except FileNotFoundError:
    print("Błąd: Upewnij się, że plik 'war9.csv' znajduje się w tym samym folderze co skrypt.")
    exit()

A = df[['x1', 'x2']].values 
b = df['y'].values

# --- Metoda 1 (Oryginalna): Demonstracja niestabilności ---
print("--- Metoda 1: Demonstracja niestabilności numerycznej ---")
U, S_diag, VT = np.linalg.svd(A, full_matrices=False)

print(f"Wartości osobliwe (S_diag): {S_diag}")
print("UWAGA: Jedna z wartości osobliwych jest bliska zeru. To powoduje błędy.\n")

# Ta linia powoduje błędy numeryczne, ponieważ próbujemy dzielić przez zero
# S_inv_unstable = np.linalg.inv(np.diag(S_diag)) 
# x_svd_unstable = VT.T @ S_inv_unstable @ U.T @ b
# print(f"Wyniki niestabilne: a={x_svd_unstable[0]}, b={x_svd_unstable[1]}\n")


# --- Metoda 2 (Poprawna): Użycie stabilnej funkcji pinv ---
print("--- Metoda 2: Użycie stabilnej funkcji np.linalg.pinv (wynik referencyjny) ---")
A_plus = np.linalg.pinv(A)
x_pinv = A_plus @ b
a_pinv, b_pinv = x_pinv
print(f"Obliczony współczynnik a = {a_pinv:.4f}")
print(f"Obliczony współczynnik b = {b_pinv:.4f}\n")


# --- Metoda 3: Ręczne SVD ze stabilizacją (poprawiona Metoda 1) ---
print("--- Metoda 3: Ręczne SVD ze stabilizacją ---")

threshold = np.finfo(float).eps * max(A.shape) * S_diag[0]

S_inv_diag_stable = np.array([1/s if s > threshold else 0 for s in S_diag])
S_inv_stable = np.diag(S_inv_diag_stable)

x_svd_stable = VT.T @ S_inv_stable @ U.T @ b
a_svd_stable, b_svd_stable = x_svd_stable

print(f"Obliczony współczynnik a = {a_svd_stable:.4f}")
print(f"Obliczony współczynnik b = {b_svd_stable:.4f}\n")

assert np.allclose(x_svd_stable, x_pinv)
print("Wyniki z metody stabilizowanej (3) i pinv (2) są zgodne!\n")


# --- Prezentacja finalnego wyniku i wizualizacja ---
print("--- Wynik końcowy ---")
print(f"Ostateczny wzór regresji: y = {a_pinv:.2f} * x1 + {b_pinv:.2f} * x2")

x1_surf, x2_surf = np.meshgrid(np.linspace(df.x1.min(), df.x1.max(), 10),
                               np.linspace(df.x2.min(), df.x2.max(), 10))
y_surf = a_pinv * x1_surf + b_pinv * x2_surf

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.x1, df.x2, df.y, color='red', marker='o', label='Dane pomiarowe')
ax.plot_surface(x1_surf, x2_surf, y_surf, color='blue', alpha=0.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Regresja wieloliniowa: Dopasowanie płaszczyzny do danych')
plt.show()
