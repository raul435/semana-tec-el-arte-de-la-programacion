import cv2
import pytesseract
import numpy as np

# Cargar la imagen
imagen = cv2.imread('/Users/joseraul/Library/Mobile Documents/com~apple~CloudDocs/escuela/semanatec arte de la programacion/placa_4.jpg')
if imagen is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta del archivo.")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplicar filtro bilateral para reducir el ruido
filtro = cv2.bilateralFilter(gris, 11, 17, 17)

# Detectar bordes con Canny
bordes = cv2.Canny(filtro, 30, 200)

# Encontrar contornos
contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]

print(f"Contornos detectados: {len(contornos)}")

placa = None
roi = None

# Buscar un contorno con forma de rectángulo (aproximación de 4 lados)
for c in contornos:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        placa = approx
        x, y, w, h = cv2.boundingRect(placa)
        roi = gris[y:y + h, x:x + w]
        break

# Verificar si se detectó una región válida
if roi is None or roi.size == 0:
    print("Error: No se detectó una región válida para la placa.")
    exit()

# Reconocimiento de texto con Tesseract
try:
    texto = pytesseract.image_to_string(roi, config='--psm 8')
    print("Texto detectado en la placa:", texto.strip())
except Exception as e:
    print(f"Error al procesar el texto con Tesseract: {e}")
    exit()

# Dibujar el contorno de la placa en la imagen original
if placa is not None:
    cv2.drawContours(imagen, [placa], -1, (0, 255, 0), 3)

# Mostrar la imagen y la región de interés
cv2.imshow("Imagen", imagen)
cv2.imshow("ROI o imagen gris", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()