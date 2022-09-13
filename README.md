# Clasificaci-n_Vinos_Blanco

PROBLEMA:

En este proyecto de machine learning vamos a intentar sacar si un vino blanco cumple los requisitios de una bodega antes de llevarlo a producción.
Para ellos vamos a analizar las propiedades de algunas variables fisico-químicas de los mismos.
Los modelos utilizados son de tipo clasificación y supervisados.

OBTENCIÓN DE DATOS:

Los datos han sido sacados de https://www.kaggle.com/code/taha07/wine-quality-prediction-data-analysis/data

PROCESAMIENTO DEL DATASET:

1- Los datos obtenidos son tanto de vino tinto, como de blanco, por lo que lo primero que haremos es quedarnos solo con estos últimos.
   Las características fisico-químicas de ambos son muy diferente, por lo que tenemos que dividirlos.

2- Hemos quitado los valores nulos.

3- El dataset original dividía los vinos en 7 categorías, pero al tener un desbalanceo muy grande lo hemos dejado en 2. 'Aptos' y 'No aptos'.

4- Creamos una nueva feature (bound sulfur dioxide). Es la resta de 'total sulfure dioxide' y 'free sulfur dioxide'.
   Es una carácteristica de cualquier vino, por lo que su obtención ante nuevos vinos es muy fácil y nuestro modelo entrena mejor con ella   que con la de 'total sulfure dioxide'. 

5- Quitamos la variable 'density'. Depende de 'residual sugar' y del 'alcohol', por lo que nos es irrelevante al tener las otras dos.

6- Quitamos algunos outliers.

7- Balanceamos la muestra para su entrenamiento. Anteriormente teníamos un 69% para 'aptos' y un 31% para los 'no aptos'.

MODELO:

Entrenamos el modelo aplicándole un Pipeline y un GridSearch.
El mejor modelo lo hemos obtenido con un GradientBoostingClassifier.
La métrica utilizada ha sido la 'Accurency' al querer clasificar los vinos entre 'Aptos' y 'No aptos' y ser los dos igual de importantes.

Para futuros modelos sería interesante tener otro tipo de variables, como el tipo de uva.
