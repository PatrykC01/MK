import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.image import imread

# Ustawienie rozmiaru wykresów
plt.rcParams['figure.figsize'] = [16, 8]

# --- Krok 1: Wczytanie lokalnego obrazu i przygotowanie danych ---

A= imread('9.webp')

# Wyświetlenie oryginalnego kolorowego obrazu
plt.figure(1)
plt.imshow(A)
plt.axis('off')
plt.title('Oryginalny obraz (kolorowy)')
plt.show()

# --- Krok 2: Analiza wartości singularnych (na podstawie wersji szarej) ---

X_gray = np.mean(A, -1)

img = plt.imshow(X_gray)
img.set_cmap('gray')
plt.axis('off')
plt.title('Oryginalny obraz (w skali szarości)')
plt.show()

U_gray, S_gray, VT_gray = np.linalg.svd(X_gray, full_matrices=False)

# Obliczenie skumulowanej sumy "informacji" (energii)
cumulative_energy = np.cumsum(S_gray) / np.sum(S_gray)

# Znalezienie liczby wartości singularnych potrzebnych do zachowania 90% informacji
try:
    r = np.where(cumulative_energy >= 0.90)[0][0] + 1
    print(f"Liczba wartości singularnych (r) użyta dla każdego kanału (na podstawie 90% energii obrazu w skali szarości): {r}")
except IndexError:
    r = len(S_gray)
    print("Nie udało się osiągnąć 90% informacji, użyto wszystkich wartości.")

# Wykres skumulowanej sumy (oparty na wersji szarej)
plt.figure(2)
plt.plot(cumulative_energy)
plt.title('Skumulowana suma znormalizowanych wartości singularnych (z wersji szarej)')
plt.xlabel('Liczba wartości singularnych')
plt.ylabel('Skumulowana energia')
plt.axhline(y=0.9, color='r', linestyle='--')
plt.axvline(x=r, color='g', linestyle='--')
plt.legend(['Skumulowana energia', 'Próg 90%', f'Wybrana liczba wartości (r): {r}'])
plt.grid(True)
plt.show()

# --- Krok 3: Wykonanie SVD i rekonstrukcja dla każdego kanału kolorów ---

# Rozdzielamy obraz na kanały R, G, B
A_float = A.astype(float)
R = A_float[:,:,0]
G = A_float[:,:,1]
B = A_float[:,:,2]

# Wykonujemy SVD dla każdego kanału
U_r, S_r, VT_r = np.linalg.svd(R, full_matrices=False)
U_g, S_g, VT_g = np.linalg.svd(G, full_matrices=False)
U_b, S_b, VT_b = np.linalg.svd(B, full_matrices=False)

# Rekonstruujemy każdy kanał używając tej samej liczby wartości singularnych 'r'
R_approx = U_r[:, :r] @ np.diag(S_r[:r]) @ VT_r[:r, :]
G_approx = U_g[:, :r] @ np.diag(S_g[:r]) @ VT_g[:r, :]
B_approx = U_b[:, :r] @ np.diag(S_b[:r]) @ VT_b[:r, :]

# --- Krok 4: Złożenie kanałów w jeden obraz i wyświetlenie ---

# Tworzymy pustą macierz o wymiarach oryginalnego obrazu
A_reconstructed = np.zeros_like(A_float)

# Składamy zrekonstruowane kanały
A_reconstructed[:,:,0] = R_approx
A_reconstructed[:,:,1] = G_approx
A_reconstructed[:,:,2] = B_approx

A_reconstructed_clipped = np.clip(A_reconstructed, 0, 255)
A_final = A_reconstructed_clipped.astype(A.dtype)


# Wyświetlenie obrazu po kompresji
plt.figure(3)
plt.imshow(A_final)
plt.axis('off')
plt.title(f'Obraz kolorowy zrekonstruowany przy użyciu r={r} wartości singularnych na kanał')
plt.show()



