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

def procesar_imagen_preview(img_original):
    """Convierte la imagen a escala con contraste realzado para mejor visualización limpia"""
    img_rgb = img_original.convert("RGB").resize((400, 400))
    enhancer = ImageEnhance.Contrast(img_rgb)
    img_enhanced = enhancer.enhance(1.5)
    buffered = io.BytesIO()
    img_enhanced.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analizar_grosor_fisura(img_original):
    """Calcula el porcentaje aproximado de área afectada por la fisura"""
    gray = ImageOps.grayscale(img_original.resize((200, 200)))
    np_gray = np.array(gray, dtype=float)
    umbral = np_gray.mean() * 0.65
    pixeles_fisura = np.sum(np_gray < umbral)
    total_pixeles = np_gray.size
    return (pixeles_fisura / total_pixeles) * 100

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
        
        # Inferencia con la imagen original
        tensor_img = transform_test(img).unsqueeze(0).to(device)
        with torch.no_grad():
            salida = modelo(tensor_img)
            probabilidad = torch.sigmoid(salida).item()

        tiene_grieta = probabilidad >= 0.75
        confianza = (probabilidad * 100) if tiene_grieta else ((1 - probabilidad) * 100)
        
        # Determinar peligrosidad combinando probabilidad y área/grosor de la fisura con recomendaciones limpias
        if not tiene_grieta:
            es_peligrosa = False
            diagnostico = "ESTRUCTURA SANA"
            nivel_peligro = "Sin Riesgo"
            recomendacion = "La estructura analizada se encuentra en óptimas condiciones. No se observan discontinuidades, fisuras ni fracturas superficiales o estructurales. Los elementos mantienen su integridad física íntegra sin evidencia de tensiones mecánicas."
        else:
            porcentaje_area = analizar_grosor_fisura(img)
            if probabilidad >= 0.92 and porcentaje_area > 12.0:
                es_peligrosa = True
                diagnostico = "GRIETA SEVERA DETECTADA"
                nivel_peligro = "PELIGROSA (Riesgo Estructural Alto)"
                recomendacion = f"Se ha detectado una grieta severa con un nivel de certeza del {confianza:.1f}% y una afectación del {porcentaje_area:.1f}% en la zona evaluada.\n\n• Posible Causa: Sobrecarga mecánica, asentamiento diferencial del suelo o fatiga estructural.\n• Implicación: Riesgo potencial para la estabilidad y capacidad portante.\n• Acción: Suspender cargas cercanas y solicitar peritaje urgente por un ingeniero civil."
            else:
                es_peligrosa = False
                diagnostico = "GRIETA LEVE DETECTADA"
                nivel_peligro = "NO PELIGROSA (Superficial / Estética)"
                recomendacion = f"Se ha identificado una fisura menor o superficial (certeza del {confianza:.1f}%, extensión del {porcentaje_area:.1f}%).\n\n• Posible Causa: Contracción por fraguado del material, cambios térmicos o asentamiento menor de acabados.\n• Implicación: Daño meramente estético, sin compromiso estructural.\n• Acción: Limpieza de la fisura, sellado elástico superficial y monitoreo periódico cada 6 meses."

        img_preview_base64 = procesar_imagen_preview(img)

        return jsonify({
            'success': True,
            'diagnostic': diagnostico,
            'danger_level': nivel_peligro,
            'is_dangerous': bool(es_peligrosa),
            'confidence': f"{confianza:.2f}%",
            'recommendation': recomendacion,
            'is_crack': bool(tiene_grieta),
            'image_mapped': img_preview_base64
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)