import cv2
import pytesseract
import numpy as np

imagen = cv2.imread('placa_4.jpg')
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
filtro = cv2.bilateralFilter(gris, 11, 17, 17)
bordes = cv2.Canny(filtro, 30, 200)

contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]

placa = None
roi = gris
for c in contornos:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        placa = approx
        x, y, w, h = cv2.boundingRect(placa)
        roi = gris[y:y + h, x:x + w]
        break

texto = pytesseract.image_to_string(roi, config='--psm 8')
print("Texto detectado en la placa:", texto.strip())

if placa is not None:
    cv2.drawContours(imagen, [placa], -1, (0, 255, 0), 3)

cv2.imshow("Imagen", imagen)
cv2.imshow("ROI o imagen gris", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
