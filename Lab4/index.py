import matplotlib.pyplot as plt
import numpy as np
import sys

plt.rcParams['figure.figsize'] = [16, 8]

# --- KROK 1: Wczytywanie danych z pliku 9.csv ---
try:
    data_flat = np.loadtxt('9.csv', delimiter=',')
except FileNotFoundError:
    print("BŁĄD: Nie znaleziono pliku '9.csv'.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"BŁĄD podczas wczytywania pliku '9.csv': {e}", file=sys.stderr)
    sys.exit(1)

if data_flat.size % 2 != 0:
    print(f"BŁĄD: Plik zawiera nieparzystą liczbę wartości ({data_flat.size}), nie można utworzyć par 2D.", file=sys.stderr)
    sys.exit(1)

num_points = data_flat.size // 2
x_coords = data_flat[:num_points]
y_coords = data_flat[num_points:]
X = np.vstack((x_coords, y_coords))

nPoints = X.shape[1]
print(f"Wczytano pomyślnie {nPoints} punktów danych z pliku '9.csv'.")


# --- KROK 2: Wykres danych wejściowych ---
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(X[0,:], X[1,:], '.', color='k', markersize=2)
ax1.grid()
ax1.set_title(f'Dane wejściowe z 9.csv ({nPoints} punktów)')
ax1.set_xlabel('Współrzędna X')
ax1.set_ylabel('Współrzędna Y')
ax1.set_aspect('equal', 'box')


# --- KROK 3: Obliczenia PCA ---
Xavg = np.mean(X,axis=1)
B = X - np.tile(Xavg,(nPoints,1)).T
U, S, VT = np.linalg.svd(B/np.sqrt(nPoints),full_matrices=0)


# --- KROK 4: Wykres wyników PCA ---
ax2 = fig.add_subplot(122)
ax2.plot(X[0,:], X[1,:], '.', color='k', markersize=2, label='Dane')
ax2.grid()
ax2.set_title('Dane z nałożoną analizą PCA')
ax2.set_xlabel('Współrzędna X')
ax2.set_ylabel('Współrzędna Y')
ax2.set_aspect('equal', 'box')

theta_plot = 2 * np.pi * np.arange(0,1,0.01)
Xstd = U @ np.diag(S) @ np.array([np.cos(theta_plot),np.sin(theta_plot)])

ax2.plot(Xavg[0] + Xstd[0,:], Xavg[1] + Xstd[1,:],'-',color='r',linewidth=2, label='Elipsy ufności (1, 2, 3σ)')
ax2.plot(Xavg[0] + 2*Xstd[0,:], Xavg[1] + 2*Xstd[1,:],'-',color='r',linewidth=2)
ax2.plot(Xavg[0] + 3*Xstd[0,:], Xavg[1] + 3*Xstd[1,:],'-',color='r',linewidth=2)

ax2.plot(np.array([Xavg[0], Xavg[0]+U[0,0]*S[0]*3]),
         np.array([Xavg[1], Xavg[1]+U[1,0]*S[0]*3]),'-',color='cyan',linewidth=4, label='Pierwsza oś główna')
ax2.plot(np.array([Xavg[0], Xavg[0]+U[0,1]*S[1]*3]),
         np.array([Xavg[1], Xavg[1]+U[1,1]*S[1]*3]),'-',color='magenta',linewidth=4, label='Druga oś główna')

ax2.legend()


# --- KROK 5: Obliczenie i wydrukowanie wyników zadania ---
print("\n--- Wyniki analizy PCA dla wariantu 9 ---")

print(f"Środek danych (Xavg): [{Xavg[0]:.4f}, {Xavg[1]:.4f}]")

os_glowna_1 = U[:, 0]
os_glowna_2 = U[:, 1]
print(f"Pierwsza oś główna (wektor U[:,0]): [{os_glowna_1[0]:.4f}, {os_glowna_1[1]:.4f}]")
print(f"Druga oś główna (wektor U[:,1]):   [{os_glowna_2[0]:.4f}, {os_glowna_2[1]:.4f}]")

kat_radiany = np.arctan2(os_glowna_1[1], os_glowna_1[0])
kat_stopnie = np.degrees(kat_radiany)
print(f"Kąt obrotu (theta): {kat_stopnie:.2f} stopni ({kat_radiany:.4f} radianów)")

wariancje = S**2
calkowita_wariancja = np.sum(wariancje)
procent_wariancji = (wariancje / calkowita_wariancja) * 100
print(f"Oś 1 wyjaśnia {procent_wariancji[0]:.2f}% wariancji (S[0]={S[0]:.4f})")
print(f"Oś 2 wyjaśnia {procent_wariancji[1]:.2f}% wariancji (S[1]={S[1]:.4f})")

plt.tight_layout()
plt.show()
