import cv2
import numpy as np

# Создаем черное цветное изображение размером 100x100 пикселей
image = np.zeros((100, 100, 3), dtype=np.uint8)

# Рисуем зеленый круг на изображении
cv2.circle(image, (50, 50), 25, (0, 255, 0), -1)

# Отображаем изображение
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()