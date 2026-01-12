import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Konfiguracja parametrów
learning_rate = 0.01
max_iterations = 1000
tolerance = 1e-6

def f(x, y):
    return np.sin((x + 3*y)**2)

def gradient(x, y):
    u = x + 3*y
    df_dx = 2 * u * np.cos(u**2)
    df_dy = 6 * u * np.cos(u**2)
    return np.array([df_dx, df_dy])

np.random.seed(42)
x = np.random.uniform(1, 3)
y = np.random.uniform(1, 3)

print(f"Punkt startowy: x0 = {x:.4f}, y0 = {y:.4f}")

history_x = [x]
history_y = [y]
history_f = [f(x, y)]

for iteration in range(max_iterations):
    grad = gradient(x, y)
    grad_norm = np.linalg.norm(grad)
    
    if grad_norm < tolerance:
        print(f"Zbieznosc osiagnieta w iteracji {iteration}")
        break
    
    x_new = x - learning_rate * grad[0]
    y_new = y - learning_rate * grad[1]
    
    # Ograniczenie do dziedziny [1, 3]
    x_new = np.clip(x_new, 1, 3)
    y_new = np.clip(y_new, 1, 3)
    
    x, y = x_new, y_new
    
    history_x.append(x)
    history_y.append(y)
    history_f.append(f(x, y))

print(f"Minimum: x={x:.4f}, y={y:.4f}, f={f(x,y):.4f}")

fig = plt.figure(figsize=(16, 5))

# Wykres 1: 3D
ax1 = fig.add_subplot(131, projection='3d')
x_range = np.linspace(1, 3, 50)
y_range = np.linspace(1, 3, 50)
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)
ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6, edgecolor='none')
ax1.plot(history_x, history_y, history_f, 'r.-', linewidth=2, markersize=4)
ax1.set_title('Powierzchnia funkcji z trajektoria')

# Wykres 2: Mapa konturowa
ax2 = fig.add_subplot(132)
contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.plot(history_x, history_y, 'r.-', linewidth=2, markersize=3)
ax2.scatter(history_x[0], history_y[0], color='green', s=100, label='Start')
ax2.scatter(history_x[-1], history_y[-1], color='red', s=100, label='Koniec')
ax2.set_title('Mapa konturowa')
ax2.legend()

# Wykres 3: Zbieżność
ax3 = fig.add_subplot(133)
ax3.plot(history_f, 'b-', linewidth=2)
ax3.set_xlabel('Iteracja')
ax3.set_ylabel('f(x,y)')
ax3.set_title('Wartosc funkcji celu')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
