import numpy as np

# Tekst zadania (Wariant 9)
data = "General intelligence (the ability to solve an arbitrary problem) is among the field's long-term goals. To solve these problems, AI researchers have adapted and integrated a wide range of problem-solving techniques, including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, probability, and economics"

# Tworzenie słowników mapujących znaki na indeksy i odwrotnie
chars = list(set(data))
data_size, X_size = len(data), len(chars)
print(f"Dane mają {data_size} znaków, {X_size} unikalnych.")
char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for i,ch in enumerate(chars)}

H_size = 128        # Rozmiar warstwy ukrytej 
T_steps = 25        # Długość sekwencji (kroków czasowych)
learning_rate = 0.1
weight_sd = 0.1
z_size = H_size + X_size # Rozmiar wektora połączonego (H + X)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y * y

class Param:
    def __init__(self, name, value):
        self.name = name
        self.v = value                  # Wartość parametru
        self.d = np.zeros_like(value)   # Gradient
        self.m = np.zeros_like(value)   # Pamięć dla Adagrad

class Parameters:
    def __init__(self): 
        
        self.W_f = Param('W_f', np.random.randn(H_size, z_size) * weight_sd)
        self.b_f = Param('b_f', np.zeros((H_size, 1)))
        
        self.W_i = Param('W_i', np.random.randn(H_size, z_size) * weight_sd)
        self.b_i = Param('b_i', np.zeros((H_size, 1)))
        
        self.W_C = Param('W_C', np.random.randn(H_size, z_size) * weight_sd)
        self.b_C = Param('b_C', np.zeros((H_size, 1)))
        
        self.W_o = Param('W_o', np.random.randn(H_size, z_size) * weight_sd)
        self.b_o = Param('b_o', np.zeros((H_size, 1)))
        
        self.W_v = Param('W_v', np.random.randn(X_size, H_size) * weight_sd)
        self.b_v = Param('b_v', np.zeros((X_size, 1)))
        
    def all(self):
        return [self.W_f, self.W_i, self.W_C, self.W_o, self.W_v,
               self.b_f, self.b_i, self.b_C, self.b_o, self.b_v]

parameters = Parameters()

def forward(x, h_prev, C_prev, p=parameters):
    z = np.vstack((h_prev, x))
    f = sigmoid(np.dot(p.W_f.v, z) + p.b_f.v)
    i = sigmoid(np.dot(p.W_i.v, z) + p.b_i.v)
    C_bar = tanh(np.dot(p.W_C.v, z) + p.b_C.v)
    
    C = f * C_prev + i * C_bar
    o = sigmoid(np.dot(p.W_o.v, z) + p.b_o.v)
    h = o * tanh(C)
    
    v = np.dot(p.W_v.v, h) + p.b_v.v
    
    v_shift = v - np.max(v)
    y = np.exp(v_shift) / np.sum(np.exp(v_shift))
    
    return z, f, i, C_bar, C, o, h, v, y

def backward(target, dh_next, dC_next, C_prev, z, f, i, C_bar, C, o, h, v, y, p=parameters):
    dv = np.copy(y)
    dv[target] -= 1
    p.W_v.d += np.dot(dv, h.T)
    p.b_v.d += dv
    
    dh = np.dot(p.W_v.v.T, dv) + dh_next
    do = dh * tanh(C) * dsigmoid(o)
    p.W_o.d += np.dot(do, z.T)
    p.b_o.d += do
    
    dC = dh * o * dtanh(tanh(C)) + dC_next
    dC_bar = dC * i * dtanh(C_bar)
    p.W_C.d += np.dot(dC_bar, z.T)
    p.b_C.d += dC_bar
    
    di = dC * C_bar * dsigmoid(i)
    p.W_i.d += np.dot(di, z.T)
    p.b_i.d += di
    
    df = dC * C_prev * dsigmoid(f)
    p.W_f.d += np.dot(df, z.T)
    p.b_f.d += df
    
    dz = (np.dot(p.W_f.v.T, df) + np.dot(p.W_i.v.T, di) + 
          np.dot(p.W_C.v.T, dC_bar) + np.dot(p.W_o.v.T, do))
    
    dh_prev = dz[:H_size, :]
    dC_prev = f * dC
    return dh_prev, dC_prev

def forward_backward(inputs, targets, h_prev, C_prev):
    x_s, z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s, v_s, y_s = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    h_s[-1] = np.copy(h_prev)
    C_s[-1] = np.copy(C_prev)
    loss = 0
    
    # Forward pass
    for t in range(len(inputs)):
        x_s[t] = np.zeros((X_size, 1))
        x_s[t][inputs[t]] = 1
        (z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t], v_s[t], y_s[t]) = \
            forward(x_s[t], h_s[t-1], C_s[t-1])
        loss += -np.log(y_s[t][targets[t], 0])
        
    # Reset gradients
    for p in parameters.all():
        p.d.fill(0)
        
    # Backward pass
    dh_next = np.zeros_like(h_s[0])
    dC_next = np.zeros_like(C_s[0])
    for t in reversed(range(len(inputs))):
        dh_next, dC_next = backward(targets[t], dh_next, dC_next, C_s[t-1],
                                    z_s[t], f_s[t], i_s[t], C_bar_s[t], 
                                    C_s[t], o_s[t], h_s[t], v_s[t], y_s[t])
    
    for p in parameters.all():
        np.clip(p.d, -1, 1, out=p.d)
        
    return loss / len(inputs), h_s[len(inputs)-1], C_s[len(inputs)-1]

def update_parameters(params=parameters):
    for p in params.all():
        p.m += p.d * p.d
        p.v += -learning_rate * p.d / np.sqrt(p.m + 1e-8)

# Inicjalizacja zmiennych treningowych
smooth_loss = -np.log(1.0 / X_size) * T_steps
h_prev = np.zeros((H_size, 1))
C_prev = np.zeros((H_size, 1))
iter_num = 0
target_loss = 0.1

print("Rozpoczęcie uczenia...")
while smooth_loss > target_loss:
    if iter_num > 0:  # Przesuwanie okna
        start_idx = iter_num % (len(data) - T_steps)
    else:
        start_idx = 0

    if start_idx + T_steps + 1 > len(data):
        h_prev = np.zeros((H_size, 1))
        C_prev = np.zeros((H_size, 1))
        start_idx = 0

    inputs = [char_to_idx[ch] for ch in data[start_idx:start_idx+T_steps]]
    targets = [char_to_idx[ch] for ch in data[start_idx+1:start_idx+T_steps+1]]

    loss, h_prev, C_prev = forward_backward(inputs, targets, h_prev, C_prev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    update_parameters()

    if iter_num % 1000 == 0:
        print(f'Iteracja {iter_num}, Strata: {smooth_loss:.4f}')
    
    iter_num += 1

print(f'\nZakończono uczenie. Ostateczna strata: {smooth_loss:.4f}')

# ==========================================
# INTERAKTYWNE TESTOWANIE MODELU
# ==========================================

def predict(seed_text, length=100):

    h = np.zeros((H_size, 1))
    C = np.zeros((H_size, 1))
    
    last_idx = 0
    for char in seed_text:
        if char in char_to_idx:
            x = np.zeros((X_size, 1))
            idx = char_to_idx[char]
            x[idx] = 1
            _, _, _, _, C, _, h, _, y = forward(x, h, C)
            last_idx = np.argmax(y) # Przewidywany następny znak
        else:
            print(f"[Ostrzeżenie: znak '{char}' nieznany, pomijam]")

    full_text = seed_text
    print(f"\n--- Generowanie (start: '{seed_text}') ---")
    
    next_idx = last_idx 
    
    for _ in range(length):
        pred_char = idx_to_char[next_idx]
        full_text += pred_char
        
        # Nowe wejście to to, co sieć przed chwilą wymyśliła
        x = np.zeros((X_size, 1))
        x[next_idx] = 1
        
        _, _, _, _, C, _, h, _, y = forward(x, h, C)
        
        next_idx = np.argmax(y)
        
    return full_text

print("\n" + "="*50)
print("TRYB TESTOWY (Wpisz 'exit' aby zakończyć)")
print("Wpisz fragment tekstu z zadania, a sieć go dokończy.")
print("="*50)

while True:
    user_input = input("\nPodaj fragment tekstu: ")
    if user_input.lower() == 'exit':
        break
    
    if not user_input:
        continue
        
    try:
        generated = predict(user_input, length=150)
        print(f"Wynik: {generated}")
    except Exception as e:
        print(f"Błąd: {e}")
