import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance

# 1. Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo en uso: {device}")

# 2. Arquitectura CNN del modelo (idéntica a la del entrenamiento)
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

# 3. Cargar el modelo entrenado
ruta_pesos = 'modelo_grietas.pth'
modelo = DetectorGrietas().to(device)

if os.path.exists(ruta_pesos):
    modelo.load_state_dict(torch.load(ruta_pesos, map_location=device))
    print(f"-> Modelo cargado exitosamente desde: {ruta_pesos}")
else:
    print(f"-> ADVERTENCIA: No se encontró el archivo de pesos '{ruta_pesos}'.")

modelo.eval()

# 4. Transformaciones de prueba
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Función para procesar y evaluar la imagen seleccionada
def evaluar_imagen(ruta_imagen):
    if not os.path.exists(ruta_imagen):
        print(f"\n[Error] La imagen '{ruta_imagen}' no existe.")
        return

    img = Image.open(ruta_imagen).convert('RGB')
    
    # Realce de contraste previo para destacar microfisuras
    enhancer = ImageEnhance.Contrast(img)
    img_contrast = enhancer.enhance(1.8)

    tensor_img = transform_test(img_contrast).unsqueeze(0).to(device)

    with torch.no_grad():
        salida = modelo(tensor_img)
        probabilidad = torch.sigmoid(salida).item()

    tiene_grieta = probabilidad >= 0.5
    confianza = probabilidad * 100 if tiene_grieta else (1 - probabilidad) * 100

    print("\n" + "="*45)
    print(f" ARCHIVO ANALIZADO: {ruta_imagen}")
    print("="*45)
    print(f" Diagnóstico : {'GRIETA DETECTADA' if tiene_grieta else 'SIN GRIETA'}")
    print(f" Confianza   : {confianza:.2f}%")
    print("="*45 + "\n")

if __name__ == '__main__':
    print("\n=== SELECTOR INTERACTIVO DE PRUEBAS ===")
    print("1. Carpeta 'Positive' (Con grietas)")
    print("2. Carpeta 'Negative' (Sin grietas)")
    
    opcion = input("\nElige una opción (1 o 2): ").strip()
    
    if opcion == '1':
        carpeta = "Positive"
    elif opcion == '2':
        carpeta = "Negative"
    else:
        print("Opción no válida.")
        exit()

    if not os.path.exists(carpeta):
        print(f"\n[Error] No se encontró la carpeta '{carpeta}' en el directorio del proyecto.")
        exit()

    # Listar las primeras imágenes de la carpeta seleccionada
    archivos = [f for f in os.listdir(carpeta) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    if not archivos:
        print(f"\nLa carpeta '{carpeta}' no contiene imágenes.")
        exit()

    print(f"\nMostrando algunas imágenes disponibles en '{carpeta}':")
    for i, archivo in enumerate(archivos[:10]):  # Muestra hasta 10 ejemplos
        print(f"  [{i}] {archivo}")

    nombre_imagen = input(f"\nEscribe el nombre exacto de la imagen que quieres probar (o presiona Enter para usar la primera): ").strip()
    
    if not nombre_imagen:
        imagen_elegida = os.path.join(carpeta, archivos[0])
    else:
        imagen_elegida = os.path.join(carpeta, nombre_imagen)

    evaluar_imagen(imagen_elegida)