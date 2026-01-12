import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import seaborn as sns

# Ustawienie stylu i ziarna losowości
sns.set_style('darkgrid')
np.random.seed(seed=1)

# ==========================================
# 1. Tworzenie zbioru danych (Wariant 9)
# ==========================================
# Wymagania:
# - 30 sekwencji
# - 20 kroków czasowych każda
# - Wartości wejściowe: rozkład jednostajny zaokrąglony do 0, 0.2, 0.4, 0.6, 0.8, 1
# - Cel t: suma liczb w sekwencji

nb_of_samples = 30
sequence_len = 20

# Inicjalizacja macierzy wejść
X = np.zeros((nb_of_samples, sequence_len))

# Generowanie sekwencji
for row_idx in range(nb_of_samples):
    # Generujemy liczby losowe [0, 1], mnożymy przez 5, zaokrąglamy do najbliższej całkowitej, dzielimy przez 5
    # Daje to wartości {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
    X[row_idx, :] = np.round(np.random.rand(sequence_len) * 5) / 5.0

# Tworzenie celów (targetów) dla każdej sekwencji (suma liczb)
t = np.sum(X, axis=1)

print(f"Wygenerowano dane dla wariantu 9:")
print(f"Kształt wejścia X: {X.shape} (oczekiwano 30, 20)")
print(f"Przykładowa sekwencja (pierwsza): {X[0]}")
print(f"Cel dla pierwszej sekwencji: {t[0]}")
print("-" * 30)

# ==========================================
# 2. Definicja funkcji modelu RNN (Forward)
# ==========================================

def update_state(xk, sk, wx, wRec):
    """
    Oblicza stan k na podstawie poprzedniego stanu (sk) i obecnego wejścia (xk).
    """
    return xk * wx + sk * wRec

def forward_states(X, wx, wRec):
    """
    Rozwija sieć w czasie i oblicza wszystkie stany.
    Zwraca macierz stanów S (ostatnia kolumna to wyjścia końcowe).
    """
    # Inicjalizacja macierzy stanów (stan początkowy s0 = 0)
    S = np.zeros((X.shape[0], X.shape[1]+1))
    
    # Aktualizacja stanów w czasie
    for k in range(0, X.shape[1]):
        S[:, k+1] = update_state(X[:, k], S[:, k], wx, wRec)
    return S

def loss(y, t): 
    """Błąd średniokwadratowy (MSE) między celem t a wyjściem y."""
    return np.mean((t - y)**2)

# ==========================================
# 3. Definicja funkcji wstecznych (Backward)
# ==========================================

def output_gradient(y, t):
    """
    Gradient funkcji straty MSE względem wyjścia y.
    """
    return 2. * (y - t)

def backward_gradient(X, S, grad_out, wRec):
    """
    Propagacja wsteczna gradientu przez sieć.
    Zwraca gradienty parametrów (wx_grad, wRec_grad) oraz gradienty stanów w czasie.
    """
    grad_over_time = np.zeros((X.shape[0], X.shape[1]+1))
    grad_over_time[:, -1] = grad_out
    
    wx_grad = 0
    wRec_grad = 0
    
    for k in range(X.shape[1], 0, -1):
        # Akumulacja gradientów parametrów
        wx_grad += np.sum(np.mean(grad_over_time[:, k] * X[:, k-1], axis=0))
        wRec_grad += np.sum(np.mean(grad_over_time[:, k] * S[:, k-1], axis=0))
        
        # Gradient na wyjściu poprzedniej warstwy
        grad_over_time[:, k-1] = grad_over_time[:, k] * wRec
        
    return (wx_grad, wRec_grad), grad_over_time

# ==========================================
# 4. Sprawdzenie gradientu (Gradient Checking)
# ==========================================

print("Rozpoczynanie sprawdzania gradientu...")
params = [1.2, 1.2]  # [wx, wRec]
eps = 1e-7

S = forward_states(X, params[0], params[1])
grad_out = output_gradient(S[:, -1], t)
backprop_grads, grad_over_time = backward_gradient(X, S, grad_out, params[1])

for p_idx, _ in enumerate(params):
    grad_backprop = backprop_grads[p_idx]
    
    # + eps
    params[p_idx] += eps
    plus_loss = loss(forward_states(X, params[0], params[1])[:, -1], t)
    
    # - eps
    params[p_idx] -= 2 * eps
    min_loss = loss(forward_states(X, params[0], params[1])[:, -1], t)
    
    # Reset parametru
    params[p_idx] += eps
    
    # Gradient numeryczny
    grad_num = (plus_loss - min_loss) / (2*eps)
    
    if not np.isclose(grad_num, grad_backprop):
        raise ValueError(
            f'Błąd! Gradient numeryczny {grad_num:.6f} różni się od '
            f'wstecznej propagacji {grad_backprop:.6f}!')

print('Gradienty poprawne (No gradient errors found).')

# ==========================================
# 5. Wizualizacja (Funkcje pomocnicze)
# ==========================================

def get_loss_surface(w1_low, w1_high, w2_low, w2_high, nb_of_ws, loss_func):
    """Generuje powierzchnię funkcji straty."""
    w1 = np.linspace(w1_low, w1_high, num=nb_of_ws)
    w2 = np.linspace(w2_low, w2_high, num=nb_of_ws)
    ws1, ws2 = np.meshgrid(w1, w2)
    loss_ws = np.zeros((nb_of_ws, nb_of_ws))
    for i in range(nb_of_ws):
        for j in range(nb_of_ws):
            loss_ws[i,j] = loss_func(ws1[i,j], ws2[i,j])
    return ws1, ws2, loss_ws

def plot_surface(ax, ws1, ws2, loss_ws):
    """Rysuje powierzchnię straty."""
    surf = ax.contourf(
        ws1, ws2, loss_ws, levels=np.logspace(-0.2, 8, 30), 
        cmap=cm.viridis, norm=LogNorm())
    ax.set_xlabel('$w_{in}$', fontsize=12)
    ax.set_ylabel('$w_{rec}$', fontsize=12)
    return surf

def plot_points(ax, points):
    """Rysuje punkty na wykresie."""
    for wx, wRec, c in points:
        ax.plot(wx, wRec, c+'o', linewidth=2)

def get_loss_surface_figure(loss_func, points):
    """Tworzy wykres powierzchni straty z punktami."""
    fig = plt.figure(figsize=(10, 4))   
    
    ax_1 = fig.add_subplot(1,2,1)
    ws1_1, ws2_1, loss_ws_1 = get_loss_surface(-3, 3, -3, 3, 50, loss_func)
    surf_1 = plot_surface(ax_1, ws1_1, ws2_1, loss_ws_1 + 1)
    plot_points(ax_1, points)
    ax_1.set_xlim(-3, 3)
    ax_1.set_ylim(-3, 3)
    ax_1.set_title("Przegląd powierzchni straty")

    ax_2 = fig.add_subplot(1,2,2)
    ws1_2, ws2_2, loss_ws_2 = get_loss_surface(-0.1, 2.1, -0.1, 2.1, 50, loss_func)
    surf_2 = plot_surface(ax_2, ws1_2, ws2_2, loss_ws_2 + 1)
    plot_points(ax_2, points)
    ax_2.set_xlim(-0.1, 2.1)
    ax_2.set_ylim(-0.1, 2.1)
    ax_2.set_title("Zbliżenie")

    return fig

def get_grad_over_time(wx, wRec):
    """Funkcja pomocnicza do pobierania gradientu w czasie dla danych wag."""
    S = forward_states(X, wx, wRec)
    grad_out = output_gradient(S[:, -1], t).sum()
    _, grad_over_time = backward_gradient(X, S, grad_out, wRec)
    return grad_over_time

def plot_gradient_over_time(points, get_grad_over_time):
    """Rysuje gradienty w czasie dla wybranych punktów."""
    fig = plt.figure(figsize=(7, 3))
    ax = plt.subplot(111)
    for wx, wRec, c in points:
        grad_over_time = get_grad_over_time(wx, wRec)
        x = np.arange(-grad_over_time.shape[1]+1, 1, 1)
        plt.plot(x, np.sum(grad_over_time, axis=0), c+'-', 
                 label=f'({wx}, {wRec})', linewidth=1, markersize=8)
    plt.xlim(0, -grad_over_time.shape[1]+1)
    plt.yscale('symlog')
    plt.xlabel('krok czasowy k', fontsize=12)
    plt.ylabel('Gradient', fontsize=12)
    plt.title('Niestabilność gradientu (wybuchający/zanikający)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig

def loss_func_viz(wx, wRec):
    return loss(forward_states(X, wx, wRec)[:, -1], t)

points = [(2,1,'r'), (1,2,'b'), (1,-2,'m'), (1,0,'c'), (1,0.5,'g'), (1,-0.5,'y')]
try:
    fig1 = get_loss_surface_figure(loss_func_viz, points)
    fig2 = plot_gradient_over_time(points, get_grad_over_time)
    plt.show()
except Exception as e:
    print("Nie można wygenerować wykresów (brak interfejsu graficznego).")
