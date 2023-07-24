# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os

# abrir los ficheros con los valores de cada emocion de cada imagen del cjto de entrenamiento
hap = open("valuesHap.txt", "r")
sad = open("valuesSad.txt", "r")
sur = open("valuesSur.txt", "r")
ang = open("valuesAng.txt", "r")

# crear 4 ficheros .arff para escribir las caracteristicas extraidas
# con los cuales se entrenara la IA
writeHap = open("trainHap.arff", "w")
writeSad = open("trainSad.arff", "w")
writeSur = open("trainSur.arff", "w")
writeAng = open("trainAng.arff", "w")

# escribir el formato de los ficheros para que WEKA pueda entenderlos
writeHap.write("@RELATION emotions\n\n@ATTRIBUTE "
               "pendMup1 NUMERIC\n@ATTRIBUTE pendMdown1 NUMERIC\n@ATTRIBUTE "
               "apertura NUMERIC\n@ATTRIBUTE pendMEBL NUMERIC\n@ATTRIBUTE "
               "pendEBR NUMERIC\n\n%Atributo objetivo: Happy \n@ATTRIBUTE "
               "Happy NUMERIC\n\n@DATA\n")
writeSad.write("@RELATION emotions\n\n@ATTRIBUTE "
               "pendMup1 NUMERIC\n@ATTRIBUTE pendMdown1 NUMERIC\n@ATTRIBUTE "
               "apertura NUMERIC\n@ATTRIBUTE pendMEBL NUMERIC\n@ATTRIBUTE "
               "pendEBR NUMERIC\n\n%Atributo objetivo: Sad \n@ATTRIBUTE "
               "Sad NUMERIC\n\n@DATA\n")
writeSur.write("@RELATION emotions\n\n@ATTRIBUTE "
               "pendMup1 NUMERIC\n@ATTRIBUTE pendMdown1 NUMERIC\n@ATTRIBUTE "
               "apertura NUMERIC\n@ATTRIBUTE pendMEBL NUMERIC\n@ATTRIBUTE "
               "pendEBR NUMERIC\n\n%Atributo objetivo: Surprised \n@ATTRIBUTE "
               "Surprised NUMERIC\n\n@DATA\n")
writeAng.write("@RELATION emotions\n\n@ATTRIBUTE "
               "pendMup1 NUMERIC\n@ATTRIBUTE pendMdown1 NUMERIC\n@ATTRIBUTE "
               "apertura NUMERIC\n@ATTRIBUTE pendMEBL NUMERIC\n@ATTRIBUTE "
               "pendEBR NUMERIC\n\n%Atributo objetivo: Angry \n@ATTRIBUTE "
               "Angry NUMERIC\n\n@DATA\n")

# inicializar el detector facial de dlib y crear un predictor de marcas faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# iterar sobre todo el conjunto de entrenamiento
for filename in os.listdir("./ConjEntrenamiento"):

    # leer los valores de las 4 emociones de una respectiva imagen
    lineHap = hap.readline()
    lineSad = sad.readline()
    lineSur = sur.readline()
    lineAng = ang.readline()

    # cargar la imagen correspondiente, redimensionarla y pasarla a escala de grises
    image = cv2.imread("ConjEntrenamiento/"+ filename)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detectar caras en la imagen
    rects = detector(gray, 1)

    # declarar variables para guardar los puntos de interes
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
        print("NO FACE DETECTED ON "+filename)
        
    # iterar sobre las caras detectadas (en el cjto solo hay 1 cara por imagen)
    for (i, rect) in enumerate(rects):
        
            # detectar las marcas faciales y convertirlas en array NumPy
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # elegir las marcas faciales de interes (boca y cejas)
            (i, j) = face_utils.FACIAL_LANDMARKS_IDXS[mouth]
            (a, b) = face_utils.FACIAL_LANDMARKS_IDXS[RE]
            (c, d) = face_utils.FACIAL_LANDMARKS_IDXS[LE]

            # guardar todos los puntos de la boca en la lista points
            for (x, y) in shape[i:j]:
                    pointsM.append([x, y])

            # guardar todos los puntos de las cejas en la lista points2 y points3
            for (n, m) in shape[a:b]:
                    points2.append([n, m])

            for (n, m) in shape[c:d]:
                    points3.append([n, m])

            #guardar los puntos de interes de ambas cejas (right & left)
            for k in range(0, len(points2), 2):
                    pointsRE.append(points2[k])
                    
            for k in range(0, len(points3), 2):
                    pointsLE.append(points3[k])
            
            #guardar los puntos de interes de la boca
            (mx0, my0) = pointsM[0] #comisura derecha
            (mx1, my1) = pointsM[6] #comisura izquierda (no se usa)
            (mx2, my2) = pointsM[14] #centro labio superior
            (mx3, my3) = pointsM[18] #centro labio inferior

            #guardar los puntos de ambas cejas en variables, el orden de izquerda a derecha
            (rex0, rey0) = pointsRE[0] #extremo externo ceja derecha (no se usa)
            (rex1, rey1) = pointsRE[1] #medio
            (rex2, rey2) = pointsRE[2] #extremo interno

            (lex0, ley0) = pointsLE[0] #extremo externo ceja izquierda (en foto)
            (lex1, ley1) = pointsLE[1] #medio
            (lex2, ley2) = pointsLE[2] #extremo interno

            #calcular la pendiente de la recta formada por los dos puntos de interes
            #pendiente boca (comisura derecha y centro labio superior)
            pendMup1 = (my0 - my2)/(mx2 - mx0)
            #pendiente boca (comisura derecha y centro labio inferior)
            pendMdown1 = (my0 - my3)/(mx3 - mx0)
            
            #pendiente cejas (extremo interno y centro de la respectiva ceja)
            pendEBR = (rey1 - rey2)/(rex2 - rex1)
            pendEBL = (ley0 - ley1)/(lex1 - lex0)
            
            #calcular la apertura de la boca
            apertura = abs(my2 - my3)

            #escribir en los ficheros las pendientes junto con el respectivo valor de cada emocion
            writeStr = str(pendMup1)+","+str(pendMdown1)+","+str(apertura)+","+str(pendEBL)+","+str(pendEBR)+","
            writeHap.write(writeStr+lineHap)
            writeSad.write(writeStr+lineSad)
            writeSur.write(writeStr+lineSur)
            writeAng.write(writeStr+lineAng)
print("FICHEROS DE ENTRENAMIENTO CREADOS CON EXITO")
#cerrar todos los ficheros
hap.close()
sad.close()
sur.close()
ang.close()
writeHap.close()
writeSad.close()
writeSur.close()
writeAng.close()
