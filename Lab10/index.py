import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Konfiguracja parametrów wykresów
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

# --- SEKCJA 1: WCZYTANIE I PRZYGOTOWANIE DANYCH ---

df_X = pd.read_csv('War9_X.csv', sep=';', decimal=',', header=None)
df_Xprime = pd.read_csv('War9_Xprime.csv', sep=';', decimal=',', header=None)

# Konwersja do macierzy numpy
X_raw = df_X.values
Xprime_raw = df_Xprime.values

if np.linalg.norm(X_raw - Xprime_raw) < 1e-10:
    print("UWAGA: Wykryto identyczne pliki X i X'. Stosowanie metody Time-Shift.")
    # Tworzymy X i X' z jednego zbioru danych (X_raw)
    # X to kolumny od 0 do przedostatniej
    X = X_raw[:, :-1]
    # X' to kolumny od 1 do ostatniej
    Xprime = X_raw[:, 1:]
else:
    X = X_raw
    Xprime = Xprime_raw

scale_factor = np.max(np.abs(X))
X_norm = X / scale_factor
Xprime_norm = Xprime / scale_factor

print(f"Wymiary macierzy do obliczeń: {X_norm.shape}")
print(f"Czynnik skalujący: {scale_factor:.2e}")

# --- SEKCJA 2: ALGORYTM DMD ---

def DMD(X, Xprime, r):
    # Krok 1: SVD
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    
    # Redukcja
    Ur = U[:, :r]
    Sigmar = np.diag(Sigma[:r])
    Vr = VT[:r, :].T
    
    # Krok 2: Macierz Atilde
    Atilde = Ur.conj().T @ Xprime @ Vr @ np.linalg.inv(Sigmar)
    
    # Krok 3: Wartości własne i wektory własne Atilde
    Lambda, W = np.linalg.eig(Atilde)
    
    # Krok 4: Mody DMD (Phi)
    Phi = Xprime @ Vr @ np.linalg.inv(Sigmar) @ W
    
    return Phi, Lambda

# --- SEKCJA 3: OBLICZENIA I WIZUALIZACJA ---

r = 2
Phi, Lambda = DMD(X_norm, Xprime_norm, r)

print("\n--- WYNIKI ---")
print(f"Obliczone wartości własne (Lambda): {Lambda}")

x0 = X_norm[:, 0]
b = np.linalg.pinv(Phi) @ x0  # Amplitudy modów

m = X_norm.shape[1]
time_steps = np.arange(m)
# Macierz Vandermonde'a (ewolucja czasu)
Vandermonde = np.power(Lambda[:, np.newaxis], time_steps)
# Model: X_dmd = Phi * b * Lambda^t
X_dmd_norm = Phi @ (np.diag(b) @ Vandermonde)

# Obliczenie błędu
error = np.linalg.norm(X_norm - X_dmd_norm, 'fro') / np.linalg.norm(X_norm, 'fro')
print(f"Błąd rekonstrukcji (względny): {error:.6e}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k--', label='|z|=1 (stabilność)')
plt.scatter(np.real(Lambda), np.imag(Lambda), c='r', s=100, label='Wartości własne')
plt.xlabel(r'Re($\lambda$)')
plt.ylabel(r'Im($\lambda$)')
plt.title('Widmo wartości własnych DMD')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.subplot(1, 2, 2)
# Wybieramy wiersz (czujnik), który ma największą wariancję
row_idx = np.argmax(np.std(X_norm, axis=1))
plt.plot(X_norm[row_idx, :], 'b-', label='Dane (znormalizowane)', linewidth=2)
plt.plot(np.real(X_dmd_norm[row_idx, :]), 'r--', label='Rekonstrukcja DMD', linewidth=2)
plt.title(f'Rekonstrukcja sygnału (Wiersz {row_idx})')
plt.xlabel('Krok czasowy')
plt.ylabel('Amplituda')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
