import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Krok 1: Wczytanie i przygotowanie obrazu
  
image_path = '9.webp'
    
# Wczytanie obrazu i konwersja do skali szarości
img = Image.open(image_path).convert('L')
    
# Konwersja obrazu do macierzy numpy i normalizacja
X = np.asarray(img, dtype=float) / 255.0
    
print(f"Obraz wczytany pomyślnie. Wymiary macierzy X: {X.shape}")

# Krok 2: Obliczenie macierzy korelacji

print("Obliczanie macierzy korelacji...")
# Korelacja kolumn
XTX = X.T @ X

# Korelacja wierszy
XXT = X @ X.T
print("Obliczenia zakończone.")

# Rozkład SVD
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Macierze zbudowane z wartości osobliwych
Sigma2 = np.diag(S**2)

# Sprawdzenie własności teoretycznych
check_columns = np.allclose(XTX, Vt.T @ Sigma2 @ Vt)
check_rows = np.allclose(XXT, U @ Sigma2 @ U.T)

print(f"Zgodność macierzy korelacji kolumn z teorią SVD: {check_columns}")
print(f"Zgodność macierzy korelacji wierszy z teorią SVD: {check_rows}")

# Krok 3: Wizualizacja wyników
plt.style.use('default')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Oryginalny obraz
axes[0].imshow(X, cmap='gray')
axes[0].set_title('Oryginalny obraz (Macierz X)')
axes[0].set_xlabel(f'Liczba kolumn: {X.shape[1]}')
axes[0].set_ylabel(f'Liczba wierszy: {X.shape[0]}')

# Macierz korelacji kolumn X^T * X
im1 = axes[1].imshow(XTX, cmap='viridis')
axes[1].set_title('Macierz korelacji kolumn (XᵀX)')
axes[1].set_xlabel(f'Kolumna {XTX.shape[1]}')
axes[1].set_ylabel(f'Kolumna {XTX.shape[0]}')
fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

# Macierz korelacji wierszy X * X^T
im2 = axes[2].imshow(XXT, cmap='viridis')
axes[2].set_title('Macierz korelacji wierszy (XXᵀ)')
axes[2].set_xlabel(f'Wiersz {XXT.shape[1]}')
axes[2].set_ylabel(f'Wiersz {XXT.shape[0]}')
fig.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)

plt.suptitle('Analiza korelacji obrazu z użyciem SVD', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
