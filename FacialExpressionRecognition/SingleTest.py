# importar las librerias necesarias
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# inicializar el detector facial de dlib y crear un predictor de marcas faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# cargar la imagen correspondiente, redimensionarla y pasarla a escala de grises
image = cv2.imread("./ConjPrueba/p16.jpg") #IMAGEN A LEER
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detectar caras en la imagen
rects = detector(gray, 1)

#declarar variables para guardar los puntos de interes
pointsM = []
points2 = []
points3 = []
pointsRE = []
pointsLE = []
mouth = "mouth"
RE = "right_eyebrow"
LE = "left_eyebrow"

# informar en caso de que no se detecte ningun rostro
if(not len(rects)):
        print("NO FACE DETECTED")
        
# iterar sobre las caras detectadas
for (i, rect) in enumerate(rects):
        
        # detectar las marcas faciales y convertirlas en array NumPy
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # clonar la imagen para dibujar sobre ella
        clone = image.copy()

        # elegir las marcas faciales de interes (boca y cejas)
        (i, j) = face_utils.FACIAL_LANDMARKS_IDXS[mouth]
        (a, b) = face_utils.FACIAL_LANDMARKS_IDXS[RE]
        (c, d) = face_utils.FACIAL_LANDMARKS_IDXS[LE]

        #guardar todos los puntos de la boca en la lista pointsM
        for (x, y) in shape[i:j]:
                pointsM.append([x,y])
                
        #guardar todos los puntos de las cejas en la lista points2 y points3
        for (n, m) in shape[a:b]:
                points2.append([n, m])
                
        for (n, m) in shape[c:d]:
                points3.append([n, m])

        #guardar los puntos de interes de ambas cejas en pointsRE y pointsLE
        for k in range(0, len(points2), 2):
                pointsRE.append(points2[k])
                (n, m) = points2[k]
                cv2.circle(clone, (n, m), 3, (255, 0, 0), -1)

        for k in range(0, len(points3), 2):
                pointsLE.append(points3[k])
                (n, m) = points3[k]
                cv2.circle(clone, (n, m), 3, (255, 0, 0), -1)
        
        #guardar los puntos de interes de la boca
        (mx0, my0) = pointsM[0] #comisura derecha
        cv2.circle(clone, (mx0, my0), 3, (255, 0, 0), -1)
        (mx1, my1) = pointsM[6] #comisura izquierda (no se usa)
        cv2.circle(clone, (mx1, my1), 3, (255, 0, 0), -1)
        (mx2, my2) = pointsM[14] #centro labio superior
        cv2.circle(clone, (mx2, my2), 3, (255, 0, 0), -1)
        (mx3, my3) = pointsM[18] #centro labio inferior
        cv2.circle(clone, (mx3, my3), 3, (255, 0, 0), -1)

        #guardar los puntos de ambas cejas en variables, el orden de izquerda a derecha
        (rex0, rey0) = pointsRE[0] #extremo externo ceja derecha (no se usa)
        (rex1, rey1) = pointsRE[1] #medio
        (rex2, rey2) = pointsRE[2] #extremo interno

        (lex0, ley0) = pointsLE[0] #extremo interno ceja izquierda
        (lex1, ley1) = pointsLE[1] #medio
        (lex2, ley2) = pointsLE[2] #extremo externo (no se usa)

        # calcular las pendientes de la rectas formadas por los puntos de interes

        #pendiente boca (comisura derecha y centro labio superior)
        pendMup1 = (my0 - my2)/(mx2 - mx0)

        #pendiente boca (comisura derecha y centro labio inferior)
        pendMdown1 = (my0 - my3)/(mx3 - mx0)

        #pendiente cejas (extremo interno y centro de la respectiva ceja)
        pendEBR = (rey1 - rey2)/(rex2 - rex1)
        pendEBL = (ley0 - ley1)/(lex1 - lex0)
        
        #calcular la apertura de la boca
        apertura = abs(my2 - my3)

        # calcular los niveles de cada emocion a partir de las pendientes sacadas
        # utilizando las funciones de regresion lineal obtenidas en el entrenamiento
        Happy = (-3.6945 * pendMup1) + (-2.1962 * pendMdown1) + 2.157
        Sad = (3.4623 * pendMup1) + (-1.9049 * pendMdown1) + (-0.0864 * apertura) + (-2.5612 * pendEBL) + 3.6316
        Surprised = (6.7817 * pendMup1) + (-4.0118 * pendMdown1) + (-0.0583 * apertura) + (-2.1956 * pendEBR) + 1.2609
        Angry = (3.2488 * pendMup1) + (-0.0497 * apertura) + (5.1385 * pendEBL) + 1.6319

        # imprimir los valores numericos de cada emocion
        print("Happy: "+ str(Happy))
        print("Sad: "+ str(Sad))
        print("Surprised: "+ str(Surprised))
        print("Angry: "+ str(Angry))

        #ARBOL DE DECISION
        if Surprised <= 2.94083:
                if Sad <= 2.968146:
                        if Angry <= 2.668535:
                                if Surprised <= 2.185779:
                                        if Happy <= 2.413626:
                                                solucion="ANGRY"
                                        else:
                                                solucion="HAPPY"
                                else:
                                        if Angry <= 2.108887:
                                                solucion="SAD"
                                        else:
                                                solucion="ANGRY"
                        else:
                                if Sad <= 1.488629:
                                        solucion="SURPRISED"
                                else:
                                        solucion="ANGRY"
                else:
                        if Angry <= 2.690646:
                                solucion="SAD"
                        else:
                                if Surprised <= 1.90996:
                                        solucion="SAD"
                                else:
                                        solucion="ANGRY"
        else:
                solucion="SURPRISED"

        # escribir la emocion detectada en la imagen y mostrarla con los puntos clave 
        cv2.putText(clone, solucion, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Shapes", clone)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
