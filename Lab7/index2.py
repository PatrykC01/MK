import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definicje Funkcji i Architektury ---

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x >= 0, 1, alpha * np.exp(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def init_layers(nn_architecture, seed=42):
    np.random.seed(seed)
    params = {}
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        input_dim = layer["input_dim"]
        output_dim = layer["output_dim"]
        # Inicjalizacja He / Xavier dla lepszej zbieżności
        params[f'W{layer_idx}'] = np.random.randn(output_dim, input_dim) * np.sqrt(2.0/input_dim)
        params[f'b{layer_idx}'] = np.zeros((output_dim, 1))
    return params

nn_architecture = [
    {"input_dim": 2, "output_dim": 2, "activation": "elu"},
    {"input_dim": 2, "output_dim": 1, "activation": "tanh"}
]

# --- 2. Propagacja ---


def forward_propagation(X, params, nn_architecture):
    cache = {}
    A = X.reshape(-1, 1)
    cache['A0'] = A
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        W = params[f'W{layer_idx}']
        b = params[f'b{layer_idx}']
        Z = W @ A + b
        
        if layer["activation"] == "elu":
            A = elu(Z)
        elif layer["activation"] == "tanh":
            A = tanh(Z)
        
        cache[f'A{layer_idx}'] = A
        cache[f'Z{layer_idx}'] = Z
    return A, cache

def backward_propagation(Y, Y_hat, cache, params, nn_architecture):
    grads = {}
    m = Y.shape[0]
    # Gradient funkcji kosztu MSE
    dA = 2 * (Y_hat - Y.reshape(Y_hat.shape)) / m
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx = layer_idx_prev + 1
        A_prev = cache[f'A{layer_idx_prev}']
        Z = cache[f'Z{layer_idx}']
        
        if layer["activation"] == "elu":
            dZ = dA * elu_derivative(Z)
        elif layer["activation"] == "tanh":
            dZ = dA * tanh_derivative(Z)
            
        W = params[f'W{layer_idx}']
        
        # Obliczanie gradientów
        grads[f'dW{layer_idx}'] = dZ @ A_prev.T
        grads[f'db{layer_idx}'] = np.sum(dZ, axis=1, keepdims=True)
        
        # Propagacja błędu do poprzedniej warstwy
        dA = W.T @ dZ
        
    return grads

# --- 3. Uruchomienie i Wizualizacja ---

# Dane wejściowe
X = np.array([0.5, 0.8])
Y = np.array([0.3])

# Obliczenia
params = init_layers(nn_architecture)
Y_hat, cache = forward_propagation(X, params, nn_architecture)
grads = backward_propagation(Y, Y_hat, cache, params, nn_architecture)

print(f"Loss: {np.sum((Y_hat-Y)**2)/2:.6f}")

# --- WIZUALIZACJA ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Wykres 1: Architektura Sieci
ax1 = axes[0]
ax1.axis('off')
ax1.set_xlim(0, 4)
ax1.set_ylim(0, 5) # Ustalony zakres pionowy 0-5
ax1.set_title('Architektura sieci', fontweight='bold')

layers_sizes = [2, 2, 1]
layer_labels = ['Input\n(2)', 'Hidden\n(ELU)', 'Output\n(tanh)']

# Rysowanie połączeń i neuronów
for i, size in enumerate(layers_sizes):
    x = i + 1 # Pozycje X: 1, 2, 3
    
    y_center = 2.5
    spacing = 1.2
    start_y = y_center - ((size - 1) * spacing) / 2
    y_pos = [start_y + j * spacing for j in range(size)]
    
    # Rysowanie neuronów
    for y in y_pos:
        circle = plt.Circle((x, y), 0.25, color='skyblue', ec='black', zorder=10)
        ax1.add_patch(circle)
        
    # Podpis warstwy
    ax1.text(x, 0.5, layer_labels[i], ha='center', fontweight='bold')

    # Rysowanie linii do następnej warstwy
    if i < len(layers_sizes) - 1:
        next_size = layers_sizes[i+1]
        next_x = x + 1
        start_y_next = y_center - ((next_size - 1) * spacing) / 2
        y_pos_next = [start_y_next + j * spacing for j in range(next_size)]
        
        for y1 in y_pos:
            for y2 in y_pos_next:
                ax1.plot([x, next_x], [y1, y2], 'k-', alpha=0.3, linewidth=0.5, zorder=1)

# Wykres 2: Wartości Gradientów
ax2 = axes[1]
grad_vals, labels = [], []

# Zbieranie danych do wykresu w odpowiedniej kolejności
for i in range(1, 3):
    # Wagi
    dW = grads[f'dW{i}'].flatten()
    grad_vals.extend(list(dW))
    labels.extend([f'dW{i}_{j}' for j in range(len(dW))])
    
    # Biasy
    db = grads[f'db{i}'].flatten()
    grad_vals.extend(list(db))
    labels.extend([f'db{i}_{j}' for j in range(len(db))])

colors = ['blue' if 'dW' in l else 'red' for l in labels]
ax2.bar(labels, grad_vals, color=colors)
ax2.set_title('Wartości gradientów')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
