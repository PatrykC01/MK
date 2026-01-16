import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Konfiguracja
NUM_BITS = 12  # Wariant 9: Liczby 12-bitowe
HIDDEN_SIZE = 16 
BATCH_SIZE = 64
EPOCHS = 2000
LEARNING_RATE = 0.01

# --- 1. Przygotowanie danych ---

def binary_repr(val, bits):
    return [int(b) for b in format(val, f'0{bits}b')[::-1]]

def generate_data(batch_size, bits):
    X = []
    Y = []
    
    for _ in range(batch_size):
        a = np.random.randint(0, 2**bits)
        b = np.random.randint(0, 2**bits)
        
        res = (a - b) % (2**bits)
        
        a_bits = binary_repr(a, bits)
        b_bits = binary_repr(b, bits)
        res_bits = binary_repr(res, bits)
        
        x_seq = [[a_b, b_b] for a_b, b_b in zip(a_bits, b_bits)]
        
        X.append(x_seq)
        Y.append([[r] for r in res_bits])
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# --- 2. Model RNN ---

class SubtractorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SubtractorRNN, self).__init__()
        
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.rnn(x) 
    
        out = self.fc(out)
        return self.sigmoid(out)

# --- 3. Trening ---

model = SubtractorRNN(input_size=2, hidden_size=HIDDEN_SIZE, output_size=1)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Rozpoczynanie treningu...")
for epoch in range(EPOCHS):
    X_train, Y_train = generate_data(BATCH_SIZE, NUM_BITS)
    
    # Reset gradientów
    optimizer.zero_grad()
    
    # Propagacja w przód
    outputs = model(X_train)
    
    # Obliczenie straty
    loss = criterion(outputs, Y_train)
    
    # Propagacja wsteczna (BPTT)
    loss.backward()
    
    # Aktualizacja wag
    optimizer.step()
    
    if epoch % 200 == 0:
        
        predicted = torch.round(outputs)
        accuracy = (predicted == Y_train).float().mean()
        print(f"Epoka {epoch}: Strata = {loss.item():.4f}, Dokładność bitowa = {accuracy.item():.4f}")

# --- 4. Weryfikacja (Interaktywna) ---

print("\n--- Rozpoczynanie testu manualnego ---")
model.eval() 

while True:
    print(f"\nPodaj dwie liczby z zakresu 0-{2**NUM_BITS - 1} (lub wpisz 'exit' aby wyjść):")
    try:
        user_in_a = input("Liczba A (odjemna): ")
        if user_in_a.lower() == 'exit': break
        val_a = int(user_in_a)
        
        user_in_b = input("Liczba B (odjemnik): ")
        if user_in_b.lower() == 'exit': break
        val_b = int(user_in_b)
        
        # Sprawdzenie zakresu
        if not (0 <= val_a < 2**NUM_BITS) or not (0 <= val_b < 2**NUM_BITS):
            print(f"Błąd: Liczby muszą być w zakresie 0-{2**NUM_BITS - 1}!")
            continue

        with torch.no_grad():
            # Obliczenie oczekiwanego wyniku
            val_res = (val_a - val_b) % (2**NUM_BITS)
            
            # Przygotowanie danych wejściowych
            a_bits = binary_repr(val_a, NUM_BITS)
            b_bits = binary_repr(val_b, NUM_BITS)
            X_test_seq = [[a, b] for a, b in zip(a_bits, b_bits)]
            X_test = torch.tensor([X_test_seq], dtype=torch.float32)
            
            # Predykcja sieci
            pred_raw = model(X_test)
            pred_bits = torch.round(pred_raw).int().squeeze().tolist()
            
            # Konwersja wyniku binarnego na liczbę dziesiętną
            pred_val = 0
            if isinstance(pred_bits, int): # Zabezpieczenie dla 1-bitowych sekwencji
                pred_bits = [pred_bits]
                
            for i, bit in enumerate(pred_bits):
                pred_val += bit * (2**i)
            
            print("-" * 30)
            print(f"Liczba A: {val_a} (binarnie: {a_bits[::-1]})")
            print(f"Liczba B: {val_b} (binarnie: {b_bits[::-1]})")
            print(f"Oczekiwany wynik: {val_res}")
            print(f"Wynik sieci:      {pred_val}")
            print(f"Bity sieci (LSB->): {pred_bits}")
            
            if val_res == pred_val:
                print(">> WYNIK POPRAWNY <<")
            else:
                print(">> BŁĄD SIECI <<")
                
    except ValueError:
        print("Błąd: Proszę wpisać poprawną liczbę całkowitą.")
