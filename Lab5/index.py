import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io

plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams.update({'font.size': 12})

# 1. Wczytanie danych
mat_contents = scipy.io.loadmat('allFaces.mat')

faces = mat_contents['faces']

m = int(mat_contents['m'].item())
n = int(mat_contents['n'].item())
nfaces = np.ndarray.flatten(mat_contents['nfaces'])

# 2. Przygotowanie danych treningowych (pierwsze 36 osób)
trainingFaces = faces[:,:np.sum(nfaces[:36])]
avgFace = np.mean(trainingFaces, axis=1) 

# 3. Obliczenie SVD na danych z odjętą średnią
# X = U * S * V^T
X = trainingFaces - np.tile(avgFace, (trainingFaces.shape[1], 1)).T
U, S, VT = np.linalg.svd(X, full_matrices=0)

# 4. Obliczenia dla Wariantu 9 (k = 90%) [cite: 129-142]
k_target = 0.90
cumulative_energy = np.cumsum(S) / np.sum(S)
r_90 = np.argmax(cumulative_energy > k_target) + 1

print(f"--- WYNIKI DLA WARIANTU 9 (k=90%) ---")
print(f"Wymagana liczba eigenfaces (r): {r_90}")
print(f"Dokładna zachowana informacja: {cumulative_energy[r_90-1]*100:.4f}%")

# 5. Rekonstrukcja obrazu testowego
test_index = np.sum(nfaces[:36])
testFace = faces[:, test_index] 
testFaceMS = testFace - avgFace 

U_r = U[:, :r_90]
Coeffs = U_r.T @ testFaceMS 
reconFace = avgFace + U_r @ Coeffs 

# 6. Wizualizacja
fig, ax = plt.subplots(1, 2)

ax[0].imshow(np.reshape(testFace, (m, n)).T, cmap='gray')
ax[0].set_title('Oryginał (Osoba 37)')
ax[0].axis('off')

ax[1].imshow(np.reshape(reconFace, (m, n)).T, cmap='gray')
ax[1].set_title(f'Rekonstrukcja\n(k=90%, r={r_90})')
ax[1].axis('off')

plt.tight_layout()
plt.show()
