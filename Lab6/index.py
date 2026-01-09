import numpy as np
import matplotlib.pyplot as plt

def relu_function(z):
    """
    Implementacja funkcji ReLU (Rectified Linear Unit).
    Wzór: f(z) = max(0, z)
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """
    Pochodna funkcji ReLU.
    Wzór: f'(z) = 0 dla z < 0, 1 dla z > 0
    """
    # Zwraca 1 gdzie z > 0, w przeciwnym razie 0
    return np.where(z > 0, 1.0, 0.0)

# 1. Przygotowanie danych
z = np.linspace(-10, 10, 400)

# 2. Obliczenie wartości funkcji i gradientu
y_relu = relu_function(z)
y_grad = relu_derivative(z)

# 3. Wizualizacja
plt.figure(figsize=(10, 6))

# Rysowanie funkcji ReLU
plt.plot(z, y_relu, label=r'Funkcja ReLU: $max(0, z)$', color='blue', linewidth=2)

# Rysowanie pochodnej
plt.plot(z, y_grad, label="Gradient (Pochodna)", color='red', linestyle='--', linewidth=2)

# Dodatki kosmetyczne wykresu
plt.title('Funkcja aktywacji ReLU oraz jej pochodna')
plt.xlabel('Wejście (z)')
plt.ylabel('Wartość')
plt.axhline(0, color='black', linewidth=0.5) # Oś X
plt.axvline(0, color='black', linewidth=0.5) # Oś Y
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()
