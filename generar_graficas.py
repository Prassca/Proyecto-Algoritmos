import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image

# 1. Configuración y Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ruta_proyecto = os.path.dirname(os.path.abspath(__file__))

# 2. Definición del Dataset y Carga
transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DatasetGrietas(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'Negative': 0, 'Positive': 1}
        for clase, etiqueta in self.class_to_idx.items():
            carpeta_clase = os.path.join(root_dir, clase)
            if os.path.exists(carpeta_clase):
                for archivo in os.listdir(carpeta_clase):
                    if archivo.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                        self.samples.append((os.path.join(carpeta_clase, archivo), etiqueta))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ruta_img, etiqueta = self.samples[index]
        imagen = Image.open(ruta_img).convert('RGB')
        if self.transform is not None:
            imagen = self.transform(imagen)
        return imagen, etiqueta

dataset_completo = DatasetGrietas(root_dir=ruta_proyecto, transform=transform_eval)
total_imagenes = len(dataset_completo)

train_size = int(0.70 * total_imagenes)
val_size = int(0.15 * total_imagenes)
test_size = total_imagenes - train_size - val_size

_, _, test_dataset = random_split(
    dataset_completo, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. Arquitectura del Modelo
class DetectorGrietas(nn.Module):
    def __init__(self):
        super(DetectorGrietas, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.feed_forward = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.feed_forward(self.features(x))

# 4. Cargar Pesos Guardados
modelo = DetectorGrietas().to(device)
ruta_pesos = os.path.join(ruta_proyecto, 'modelo_grietas.pth')
modelo.load_state_dict(torch.load(ruta_pesos, map_location=device))
modelo.eval()

# 5. Obtener Predicciones sobre Test
tn, fp, fn, tp = 0, 0, 0, 0

print("Evaluando el conjunto de prueba para generar métricas...", flush=True)
with torch.no_grad():
    for imagenes, etiquetas in test_loader:
        imagenes = imagenes.to(device)
        etiquetas = etiquetas.to(device).float().unsqueeze(1)
        salidas = modelo(imagenes)
        preds = (torch.sigmoid(salidas) >= 0.5).float()
        
        for p, r in zip(preds, etiquetas):
            if r == 0 and p == 0: tn += 1
            elif r == 0 and p == 1: fp += 1
            elif r == 1 and p == 0: fn += 1
            elif r == 1 and p == 1: tp += 1

# 6. Crear y Guardar la Matriz de Confusión
fig, ax = plt.subplots(figsize=(6, 5))
matriz = [[tn, fp], [fn, tp]]

cax = ax.matshow(matriz, cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Detección de Grietas', pad=20)
fig.colorbar(cax)

labels = ['Negative', 'Positive']
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(matriz[i][j]), va='center', ha='center', color='red', fontsize=14, fontweight='bold')

plt.xlabel('Predicción del Modelo')
plt.ylabel('Clase Real')
plt.tight_layout()

# Guardar la imagen en el proyecto
ruta_grafica = os.path.join(ruta_proyecto, 'matriz_confusion.png')
plt.savefig(ruta_grafica)
print(f"\n-> ¡Éxito! Gráfica guardada como: '{ruta_grafica}'", flush=True)

# Imprimir resumen de conteos
print(f"Verdaderos Negativos (Sin Grieta correctos): {tn}")
print(f"Falsos Positivos (Sin Grieta marcados como Grieta): {fp}")
print(f"Falsos Negativos (Con Grieta no detectados): {fn}")
print(f"Verdaderos Positivos (Con Grieta correctos): {tp}")