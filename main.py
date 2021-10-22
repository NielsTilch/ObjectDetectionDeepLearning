#Importing necessary package
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.regularizers import l2
from keras.utils.vis_utils import plot_model
from keras.datasets import mnist
from keras.utils.vis_utils import plot_model
import cv2
import seaborn as sns
from random import sample, randint

#Link to all of the dataset example :
# mnist : https://deepai.org/dataset/mnist
# cifar10 : https://www.tensorflow.org/datasets/catalog/cifar10
# cifar100 : https://paperswithcode.com/dataset/cifar-100

#Function use to test the build neural network on the mnist dataset 
#Entry variable : Number of epochs (number of time learning the same set)

def mnisttest(Nombre_epochs):
  
  (train_images_mnist, train_labels_mnist), (test_images_mnist, test_labels_mnist) = mnist.load_data()


  #On normalise les valeurs des pixels pour les mettre entre 0 et 1
  train_images_mnist = train_images_mnist/255
  test_images_mnist = test_images_mnist/255



  #On créé le réseau de neurones
  #----------------------
  #The neural network
  #----------------------
  modelCNN = tf.keras.Sequential([
      tf.keras.layers.Conv2D(64,3,padding="same",input_shape=(28,28,1),activation="relu",kernel_regularizer=l2(0.0005)),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(48,3,padding="same",activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(32,3,padding="same",activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(31,3,padding="same",activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64,activation="relu"),
      tf.keras.layers.Dense(64,activation="relu"),
      tf.keras.layers.Dense(10,activation="softmax")
  ])




  #On compile le réseau
  modelCNN.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics = ["accuracy"]
  )

  #On augmente la dimensions des matrices d'entrainement et de test pour pouvoir les faire rentrer dans le réseau
  train_images_mnist = np.expand_dims(train_images_mnist,3)
  test_images_mnist = np.expand_dims(test_images_mnist,3)
  list_images = list(test_images_mnist)
  random_image = sample(list_images, 1)
  random_image = np.array(random_image, dtype='float')

  #On entraine le modèle
  history = modelCNN.fit(
      train_images_mnist,
      train_labels_mnist,
      validation_data=(test_images_mnist, test_labels_mnist),
      epochs = Nombre_epochs
  )
  train_images_mnist = np.expand_dims(train_images_mnist,4)
  predictions = modelCNN.predict(test_images_mnist)

  labels = [0,1,2,3,4,5,6,7,8,9]
  i=0;
  pred = np.zeros(len(predictions))
  for x in predictions :
    pred[i] = np.argmax(x)
    i=i+1


  #On créé la matrice de confusion pour observer les positifs et les négatifs
  cm = tf.math.confusion_matrix(test_labels_mnist, pred)
 
  plt.figure(figsize=(9,9))
  sns.heatmap(cm, cbar=False, xticklabels=labels, yticklabels=labels, fmt='d', annot=True, cmap=plt.cm.Blues)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.show()

  #Showing the results
  x_axis = np.linspace(2,Nombre_epochs+1,1)
  plt.plot(history.history['accuracy'],label = "Précision de l'entrainement")
  plt.plot(history.history['val_accuracy'],label="Précision des tests")
  plt.title("Valeur de la précision de l'entrainement et de l'évaluation en fonction des epochs")
  plt.xlabel("Nombre d'epochs")
  plt.ylabel("Taux de précision")
  plt.legend()
  plt.show()

  #Evaluation du model avec les données tests
  modelCNN.evaluate(test_images_mnist, test_labels_mnist)

  #Boucle pour les tests de reconnaissance
  while (True):
    random_image = sample(list_images, 1)
    random_image = np.array(random_image, dtype='float')

    predict_random = modelCNN.predict(random_image)

    random_pred = str(np.argmax(predict_random))

    random_image = random_image.reshape((28, 28))
    plt.imshow(random_image)
    plt.title("résultat prédit : " + random_pred)
    plt.show()

    if input("Afficher un autre exemple ? oui/non : ") == "non":
      break




###################################################################################################

#Function use to test the build neural network on the cifar dataset 
#For this function, we pre-process the image in black and white to see
#how the black white preprocessing affect the accuracy.

#Entry variable : Number of epochs (number of time learning the same set)

def cifar10testgris(Nombre_epochs):
  
  
  (train_images_cifar10, train_labels_cifar10), (test_images_cifar10, test_labels_cifar10) = cifar10.load_data()

  #Passage au niveau de gris
  train_images_cifar10 = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in train_images_cifar10])
  test_images_cifar10 = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in test_images_cifar10])



  #On normalise les valeurs des pixels pour les mettre entre 0 et 1
  train_images_cifar10 = train_images_cifar10/255
  test_images_cifar10 = test_images_cifar10/255


  #On créé le réseau de neurones
  #----------------------
  #The neural network
  #----------------------
  modelCNN = tf.keras.Sequential([
      tf.keras.layers.Conv2D(128,3,padding="same",input_shape=(32,32,1),activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(64,3,padding="same",activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(32,3,padding="same",activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(16,3,padding="same",activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128,activation="relu"),
      tf.keras.layers.Dense(64,activation="relu"),
      tf.keras.layers.Dense(10,activation="softmax")
  ])

  #On augmente la dimensions des matrices d'entrainement et de test pour pouvoir les faire rentrer dans le réseau
  train_images_cifar10 = np.expand_dims(train_images_cifar10,3)
  test_images_cifar10 = np.expand_dims(test_images_cifar10,3)

  #On compile le réseau
  modelCNN.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics = ["accuracy"]
  )

  #On entraine le modèle
  history = modelCNN.fit(
      train_images_cifar10,
      train_labels_cifar10,
      validation_data=(test_images_cifar10, test_labels_cifar10),
      epochs = Nombre_epochs
  )

  modelCNN.evaluate(test_images_cifar10, test_labels_cifar10)

  x_axis = np.linspace(2,Nombre_epochs+1,1)
  plt.plot(history.history['accuracy'],label = "Précision de l'entrainement")
  plt.plot(history.history['val_accuracy'],label="Précision des tests")
  plt.title("Valeur de la précision de l'entrainement et de l'évaluation en fonction des epochs")
  plt.xlabel("Nombre d'epochs")
  plt.ylabel("Taux de précision")
  plt.legend()
  plt.show()


###################################################################################################

#Function use to test the build neural network on the cifar10 dataset 
#Entry variable : Number of epochs (number of time learning the same set)

def cifar10test(Nombre_epochs):
  
  
  (train_images_cifar10, train_labels_cifar10), (test_images_cifar10, test_labels_cifar10) = cifar10.load_data()

  #On normalise les valeurs des pixels pour les mettre entre 0 et 1
  train_images_cifar10 = train_images_cifar10/255
  test_images_cifar10 = test_images_cifar10/255



  #On créé le réseau de neurones
  #----------------------
  #The neural network
  #----------------------
  modelCNN = tf.keras.Sequential([
      tf.keras.layers.Conv2D(128,3,padding="same",input_shape=(32,32,3),activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(64,3,padding="same",activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(32,3,padding="same",activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(16,3,padding="same",activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128,activation="relu"),
      tf.keras.layers.Dense(64,activation="relu"),
      tf.keras.layers.Dense(10,activation="softmax")
  ])


  #On compile le réseau
  modelCNN.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics = ["accuracy"]
  )

  #On entraine le modèle
  history = modelCNN.fit(
      train_images_cifar10,
      train_labels_cifar10,
      validation_data = (test_images_cifar10, test_labels_cifar10),
      epochs = Nombre_epochs
  )


  

  #Showing the results
  x_axis = np.linspace(2,Nombre_epochs+1,1)
  plt.plot(history.history['accuracy'],label = "Précision de l'entrainement")
  plt.plot(history.history['val_accuracy'],label="Précision des tests")
  plt.title("Valeur de la précision de l'entrainement et de l'évaluation en fonction des epochs")
  plt.xlabel("Nombre d'epochs")
  plt.ylabel("Taux de précision")
  plt.legend()
  plt.show()



###################################################################################################


#Function use to test the build neural network on the cifar100 dataset 
#For this function, we pre-process the image in black and white to see
#how the black white preprocessing affect the accuracy.

#Entry variable : Number of epochs (number of time learning the same set)

def cifar100testgris(Nombre_epochs):


  (train_images_cifar100, train_labels_cifar100), (test_images_cifar100, test_labels_cifar100) = cifar10.load_data()

  train_images_cifar100 = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in train_images_cifar100])
  test_images_cifar100 = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in test_images_cifar100])

  #On normalise les valeurs des pixels pour les mettre entre 0 et 1
  train_images_cifar100 = train_images_cifar100/255
  test_images_cifar100 = test_images_cifar100/255



  #On créé le réseau de neurones
  #----------------------
  #The neural network
  #----------------------
  modelCNN = tf.keras.Sequential([
      tf.keras.layers.Conv2D(64,3,padding="same",input_shape=(32,32,1),activation="relu",kernel_regularizer=l2(0.0005)),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(48,3,padding="same",activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(32,3,padding="same",activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(16,3,padding="same",activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64,activation="relu"),
      tf.keras.layers.Dense(64,activation="relu"),
      tf.keras.layers.Dense(100,activation="softmax")
  ])


  train_images_cifar100 = np.expand_dims(train_images_cifar100,3)
  test_images_cifar100 = np.expand_dims(test_images_cifar100,3)

  #On compile le réseau
  modelCNN.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics = ["accuracy"]
  )

  #On augmente la dimensions des matrices d'entrainement et de test pour pouvoir les faire rentrer dans le réseau
  train_images_cifar100 = np.expand_dims(train_images_cifar100,3)
  test_images_cifar100 = np.expand_dims(test_images_cifar100,3)

  #On entraine le modèle
  history = modelCNN.fit(
      train_images_cifar100,
      train_labels_cifar100,
      validation_data = (test_images_cifar100, test_labels_cifar100),
      epochs = Nombre_epochs
  )

  #Showing results
  x_axis = np.linspace(2,Nombre_epochs+1,1)
  plt.plot(history.history['accuracy'],label = "Précision de l'entrainement")
  plt.plot(history.history['val_accuracy'],label="Précision des tests")
  plt.title("Valeur de la précision de l'entrainement et de l'évaluation en fonction des epochs")
  plt.xlabel("Nombre d'epochs")
  plt.ylabel("Taux de précision")
  plt.legend()
  plt.show()


  labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'];
  list_images = list(test_images)
  while (True):
    random_image = sample(list_images, 1)
    random_image = np.array(random_image, dtype='float')

    predict_random = modelCNN.predict(random_image)

    random_pred = np.argmax(predict_random)

    random_image = random_image.reshape((32, 32))
    plt.imshow(random_image)
    plt.title("résultat prédit : " + str(labels[random_pred]))
    plt.show()

    if input("Afficher un autre exemple ? oui/non : ") == "non":
      break



###################################################################################################

#Function use to test the build neural network on the cifar100 dataset 
#Entry variable : Number of epochs (number of time learning the same set)

def cifar100test(Nombre_epochs):


  (train_images_cifar100, train_labels_cifar100), (test_images_cifar100, test_labels_cifar100) = cifar100.load_data()

  #On normalise les valeurs des pixels pour les mettre entre 0 et 1
  train_images_cifar100 = train_images_cifar100/255
  test_images_cifar100 = test_images_cifar100/255



  #On créé le réseau de neurones
  #----------------------
  #The neural network
  #----------------------
  modelCNN = tf.keras.Sequential([
      tf.keras.layers.Conv2D(64,3,padding="same",input_shape=(32,32,3),activation="relu",kernel_regularizer=l2(0.0005)),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(48,3,padding="same",activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(32,3,padding="same",activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(16,3,padding="same",activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=(2,2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64,activation="relu"),
      tf.keras.layers.Dense(64,activation="relu"),
      tf.keras.layers.Dense(100,activation="softmax")
  ])



  #On compile le modèle
  modelCNN.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics = ["accuracy"]
  )

  #On entraine le modèle
  history = modelCNN.fit(
      train_images_cifar100,
      train_labels_cifar100,
      validation_data = (test_images_cifar100, test_labels_cifar100),
      epochs = Nombre_epochs
  )

  #Showing the results
  x_axis = np.linspace(2,Nombre_epochs+1,1)
  plt.plot(history.history['accuracy'],label = "Précision de l'entrainement")
  plt.plot(history.history['val_accuracy'],label="Précision des tests")
  plt.title("Valeur de la précision de l'entrainement et de l'évaluation en fonction des epochs")
  plt.xlabel("Nombre d'epochs")
  plt.ylabel("Taux de précision")
  plt.legend()
  plt.show()



###################################################################################################
##################################################################################################
###################################################################################################

print("Bienvenue dans l'apprentissage / Welcome to the learning object program \n\n\n")



print("Voici les possibilités/ Tests possibilities:\n")
print("1 : Pour la librairie Cifar10 / Cifar10 library")
print("2 : Pour la librairie mnist (avec matrice de confusion) / Mnist Library")
print("3 : Pour la librairie Cifar100 / Cifar100 Library\n")

while(1):
  
  i = int(input("Choix / Choice : "))

  j = int(input("Choisir le nombre d'epoch / Number of epochs : "))

  if(i != 2):
    k = str(input("Prétraitement gris 'oui' ou 'non' / Gray pre-processing 'yes' or 'no': "))

  if (j >0) :
    if(i==1):
      if (k == "oui" or k == "yes"):
        cifar10testgris(j)
      elif (k=="non" or k == "no"):
        cifar10test(j)
      else:
        print("/!\ Problème prétraitement / Problem with pre-processing input choice /!\ ")
    
    elif(i==2):
      mnisttest(j)
      

    elif(i==3):
      if (k == "oui" or k == "yes"):
        cifar100testgris(j)
      elif (k=="non" or k == "no"):
        cifar100test(j)
      else:
        print("/!\ Problème prétraitement / Problem with pre-processing input choice /!\ ")
        
    
    else:
        print("\n\n\n/!\ Entrée non reconnu / Unrecognized input /!\ \n\n\n")
  else:
    print("\n\n\n/!\ Nombre d'epoch nul ou négatif ! / Epochs input invalid /!\ \n\n\n")