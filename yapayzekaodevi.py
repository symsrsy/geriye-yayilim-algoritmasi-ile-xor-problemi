import numpy as np
# Bu Geriye yayılım algoritmasında ilk katman 2 girişe ve 3 nörona sahitir.2.katman 3 girişe ve 1 çıkışa sahiptir. 
# Sigmoid aktivasyon fonksiyonu
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid aktivasyon fonksiyonunun türevi
#ikinci
def sigmoid_derivative(x):
    return x * (1 - x)

# XOR veri seti
training_inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
training_outputs = np.array([[0],[1],[1],[0]])

# Ağırlıkları ve eşik değerlerini kullanıcıdan alalım
# Katman 1
print("Katman 1 için ağırlıkları girin:")
synaptic_weights_0 = np.zeros((2, 3))
for i in range(2):
    for j in range(3):
        synaptic_weights_0[i][j] = float(input(f"Ağırlık {i+1}-{j+1}: "))

print("\nKatman 1 için eşik değerlerini girin:")
bias_0 = np.zeros(3)
for i in range(3):
    bias_0[i] = float(input(f"Eşik {i+1}: "))

# Katman 2
print("\nKatman 2 için ağırlıkları girin:")
synaptic_weights_1 = np.zeros((3, 1))
for i in range(3):
    synaptic_weights_1[i][0] = float(input(f"Ağırlık {i+1}-1: "))

print("\nKatman 2 için eşik değeri girin:")
bias_1 = np.zeros(1)
bias_1[0] = float(input("Eşik: "))

print("\nGirilen başlangıç ağırlıkları (Katman 1):")
print(synaptic_weights_0)
print("\nGirilen başlangıç eşik değerleri (Katman 1):")
print(bias_0)
print("\nGirilen başlangıç ağırlıkları (Katman 2):")
print(synaptic_weights_1)
print("\nGirilen başlangıç eşik değeri (Katman 2):")
print(bias_1)

# Eğitim iterasyonları
for iteration in range(60000):
    # İleri yayılım
    input_layer = training_inputs
    outputs_0 = sigmoid(np.dot(input_layer, synaptic_weights_0) + bias_0)
    outputs_1 = sigmoid(np.dot(outputs_0, synaptic_weights_1) + bias_1)

    # Hata hesaplama
    # error_1: İkinci katmandaki çıktıdaki hata.
    error_1 = training_outputs - outputs_1
    if iteration % 10000 == 0:
        print("Hata: ", str(np.mean(np.abs(error_1))))

    # Geriye yayılım
    # sigmoid_derivative(outputs_1) ifadesi
    # ikinci katmandaki çıktıların türevini hesaplar.
    # Bu türev, ağırlıkların güncellenmesi için geriye yayılımın bir parçası olarak kullanılır.
    # Eksi değerleri de alabilmek için
    # sigmoid fonksiyonunun türevi olarak x * (1 - x) kullanılır.
    delta_1 = error_1 * sigmoid_derivative(outputs_1)
    error_0 = delta_1.dot(synaptic_weights_1.T)
    delta_0 = error_0 * sigmoid_derivative(outputs_0)
    # delta_1: İkinci katmandaki hata ile çıktının sigmoid türevi arasındaki çarpım.
    # error_0: Gizli katmandaki hata.
    # delta_0: Gizli katmandaki hata ile girişlerin sigmoid türevi arasındaki çarpım.

    # Ağırlık ve eşik değeri güncelleme
    synaptic_weights_1 += outputs_0.T.dot(delta_1)
    synaptic_weights_0 += input_layer.T.dot(delta_0)
    bias_1 += np.sum(delta_1, axis=0)
    bias_0 += np.sum(delta_0, axis=0)



print("\nEğitim tamamlandı. Toplam epoch sayısı:", iteration)
print("\nEğitim sonrası ağırlıklar (Katman 1):")
print(synaptic_weights_0)
print("\nEğitim sonrası eşik değerleri (Katman 1):")
print(bias_0)
print("\nEğitim sonrası ağırlıklar (Katman 2):")
print(synaptic_weights_1)
print("\nEğitim sonrası eşik değeri (Katman 2):")
print(bias_1)

print("\nEğitim sonrası çıkışlar:")
print(outputs_1)


# Test verilerini kullanıcıdan alma
test_inputs = []
num_test_samples = int(input("Kaç test örneği girmek istersiniz? "))

for i in range(num_test_samples):
    print(f"Test örneği {i+1}:")
    test_input = []
    for j in range(2):
        test_input.append(float(input(f"Giriş {j+1}: ")))
    test_inputs.append(test_input)

test_inputs = np.array(test_inputs)

# Test verileri üzerinde ileri yayılım
test_outputs_0 = sigmoid(np.dot(test_inputs, synaptic_weights_0) + bias_0)
test_outputs_1 = sigmoid(np.dot(test_outputs_0, synaptic_weights_1) + bias_1)

# Tahminleri yazdırma
print("\nTest verileri için çıkış tahminleri:")
print(test_outputs_1)
