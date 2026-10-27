import os
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageEnhance, ImageOps
import io
import base64

app = Flask(__name__)

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Arquitectura CNN del modelo
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

# Cargar el modelo
modelo = DetectorGrietas().to(device)
ruta_pesos = 'modelo_grietas.pth'

if os.path.exists(ruta_pesos):
    modelo.load_state_dict(torch.load(ruta_pesos, map_location=device))
    print("-> Modelo cargado correctamente para la web.")
else:
    print("-> Advertencia: No se encontró 'modelo_grietas.pth'.")

modelo.eval()

# Transformaciones
from torchvision import transforms
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generar_mapa_termico(img_original, probabilidad):
    """Genera una visualización tipo mapa térmico (JET/Ironbow) superpuesta según la severidad"""
    img_rgb = img_original.convert("RGB").resize((400, 400))
    
    # Escala de grises para detectar bordes/fisuras
    gray = ImageOps.grayscale(img_rgb)
    np_gray = np.array(gray, dtype=float)
    
    # Normalizar matriz
    if np_gray.max() > np_gray.min():
        np_gray = (np_gray - np_gray.min()) / (np_gray.max() - np_gray.min())
    else:
      np_gray = np.zeros_like(np_gray)

    # Crear máscara de calor basada en probabilidad y texturas
    heatmap = Image.fromarray(np.uint8(np_gray * 255), mode='L')
    heatmap_colored = ImageOps.colorize(heatmap, black="blue", white="red")
    
    # Combinar imagen original con el mapa térmico translúcido
    blend = Image.blend(img_rgb, heatmap_colored, alpha=0.4 if probabilidad > 0.5 else 0.15)
    
    buffered = io.BytesIO()
    blend.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró ninguna imagen'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Archivo vacio'})

    try:
        img = Image.open(file.stream).convert('RGB')
        
        # Realce de contraste para microfisuras
        enhancer = ImageEnhance.Contrast(img)
        img_contrast = enhancer.enhance(1.8)

        # Inferencia con PyTorch
        tensor_img = transform_test(img_contrast).unsqueeze(0).to(device)
        with torch.no_grad():
            salida = modelo(tensor_img)
            probabilidad = torch.sigmoid(salida).item()

        tiene_grieta = probabilidad >= 0.5
        confianza = (probabilidad * 100) if tiene_grieta else ((1 - probabilidad) * 100)
        
        # Generar imagen con mapeo térmico
        img_map_base64 = generar_mapa_termico(img, probabilidad)

        return jsonify({
            'success': True,
            'diagnostic': 'GRIETA DETECTADA' if tiene_grieta else 'ESTRUCTURA SANA (SIN GRIETAS)',
            'confidence': f"{confianza:.2f}%",
            'is_crack': bool(tiene_grieta),
            'image_mapped': img_map_base64
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)