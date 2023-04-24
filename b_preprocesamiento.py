
import numpy as np

import cv2 ### para leer imagenes jpg
from matplotlib import pyplot as plt ## para gráfciar imágnes
import a_funciones as fn #### funciones personalizadas, carga de imágenes
import joblib ### para descargar array
    
############################################
##### ver ejemplo de imágenes cargadas ######
#############################################

img1=cv2.imread(r'C:\Users\juanm\Downloads\Operations-Healthness_ADA_G2-main\Operations-Healthness_ADA_G2-main\colon_image_sets\colon_n\colonn990.jpeg')
img2 = cv2.imread(r'C:\Users\juanm\Downloads\Operations-Healthness_ADA_G2-main\Operations-Healthness_ADA_G2-main\colon_image_sets\colon_aca\colonca98.jpeg')


############################################
##### ver ejemplo de imágenes cargadas ######
#############################################

plt.imshow(img1)
plt.title('aca')
plt.show()

plt.imshow(img2)
plt.title('normal')
plt.show()

###### representación numérica de imágenes ####

img1.shape ### tamaño de imágenes
img1.max() ### máximo valor de intensidad en un pixel
img1.min() ### mínimo valor de intensidad en un pixel

np.prod(img1.shape) ### 5 millones de observaciones cada imágen

#### dado que se necesitarían muchas observaciones (imágenes para entrenar)
#### un modelo con tantas observaciones y no tenemos, vamos a reescalar las imágenes

img1 = cv2.resize(img1 ,(100,100))
plt.imshow(img1)
plt.title('Normal')
plt.show()

################################################################
######## Código para cargar todas las imágenes #############
####### reducir su tamaño y convertir en array ################
################################################################


width = 100 #tamaño para reescalar imágen
num_classes = 2 #clases variable respuesta
trainpath = 'C:\\Users\\juanm\\Downloads\\Operations-Healthness_ADA_G2-main\\Operations-Healthness_ADA_G2-main\\colon_image_sets\\'
#testpath = 'C:\\Users\\juanm\\Downloads\\Operations-Healthness_ADA_G2-main\\Operations-Healthness_ADA_G2-main\\colon_image_sets\\colon_n'

x_train, y_train= fn.img2data(trainpath) #Run in train
#x_test, y_test = fn.img2data(testpath) #Run in test


#### convertir salidas a numpy array ####
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train.shape

x_test.shape

####### salidas del preprocesamiento bases listas ######

joblib.dump(x_train, "x_train.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(x_test, "x_test.pkl")
joblib.dump(y_test, "y_test.pkl")