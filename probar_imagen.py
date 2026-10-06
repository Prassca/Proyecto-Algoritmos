import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ruta_proyecto = os.path.dirname(os.path.abspath(__file__))

# 2. Transformaciones estándar para imágenes de prueba
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Definición idéntica de la arquitectura CNN
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

# 4. Cargar pesos preentrenados
modelo = DetectorGrietas().to(device)
ruta_pesos = os.path.join(ruta_proyecto, 'modelo_grietas.pth')
modelo.load_state_dict(torch.load(ruta_pesos, map_location=device))
modelo.eval()

# 5. Función de predicción
def evaluar_imagen(ruta_imagen):
    if not os.path.exists(ruta_imagen):
        print(f"Error: No se encontró la imagen en {ruta_imagen}")
        return

    # Cargar y preprocesar imagen
    imagen = Image.open(ruta_imagen).convert('RGB')
    tensor_img = transform_test(imagen).unsqueeze(0).to(device)

    with torch.no_grad():
        salida = modelo(tensor_img)
        probabilidad = torch.sigmoid(salida).item()

    
    if probabilidad >= 0.5:
        diagnostico = "GRIETA DETECTADA (Positive)"
        confianza = probabilidad * 100
    else:
        diagnostico = "SIN GRIETA (Negative)"
        confianza = (1 - probabilidad) * 100

    print("\n==========================================")
    print(f" Archivo: {os.path.basename(ruta_imagen)}")
    print(f" Resultado: {diagnostico}")
    print(f" Confianza: {confianza:.2f}%")
    print("==========================================")

#### Probar

mi_foto = os.path.join(ruta_proyecto, "Positive", "foto2.jpeg") 

evaluar_imagen("tu_ruta_de_imagen_aqui.jpggggggggggggg") 
# Comentario de prueba
print("Prueba completada.")
 