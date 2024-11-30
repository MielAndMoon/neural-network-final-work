import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

class PredictorInmuebles:
    def __init__(self, datos):
        """
        Inicializa el predictor de inmuebles con los datos proporcionados

        Parámetros:
        - datos: DataFrame con columnas de características del inmueble
        """
        self.datos = datos
        self.caracteristicas = None
        self.target = None
        self.modelo = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def preparar_datos(self, columnas_caracteristicas, columna_precio):
        """
        Prepara los datos para entrenamiento dividiendo en características y target

        Parámetros:
        - columnas_caracteristicas: Lista de columnas para predecir
        - columna_precio: Columna con el precio del inmueble
        """
        self.caracteristicas = self.datos[columnas_caracteristicas]
        self.target = self.datos[columna_precio]

        # Escalar características y target
        X_scaled = self.scaler_x.fit_transform(self.caracteristicas)
        y_scaled = self.scaler_y.fit_transform(self.target.values.reshape(-1, 1))

        # Dividir datos en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )

    def entrenar_perceptron(self, capas_ocultas=(10, 5), max_iter=500):
        """
        Entrena un modelo de Perceptrón Multicapa

        Parámetros:
        - capas_ocultas: Tupla con número de neuronas por capa oculta
        - max_iter: Máximo número de iteraciones de entrenamiento
        """
        self.modelo = MLPRegressor(
            hidden_layer_sizes=capas_ocultas,
            activation='relu',
            solver='adam',
            max_iter=max_iter,
            random_state=42
        )

        self.modelo.fit(self.X_train, self.y_train.ravel())

    def evaluar_modelo(self):
        """
        Evalúa el rendimiento del modelo

        Retorna:
        - Score de entrenamiento
        - Score de prueba
        """
        score_train = self.modelo.score(self.X_train, self.y_train)
        score_test = self.modelo.score(self.X_test, self.y_test)

        print(f"Score de entrenamiento: {score_train:.4f}")
        print(f"Score de prueba: {score_test:.4f}")

    def predecir(self, nuevos_datos):
        """
        Realiza predicciones para nuevos datos

        Parámetros:
        - nuevos_datos: DataFrame con características de nuevos inmuebles

        Retorna:
        - Predicciones de precio desnormalizadas
        """
        # Escalar nuevos datos
        nuevos_datos_scaled = self.scaler_x.transform(nuevos_datos)

        # Predecir y desnormalizar
        predicciones_scaled = self.modelo.predict(nuevos_datos_scaled)
        predicciones = self.scaler_y.inverse_transform(predicciones_scaled.reshape(-1, 1))

        return predicciones

def main():
    # Crear un conjunto de datos sintético más grande
    np.random.seed(42)
    n_samples = 100  # Número aumentado de muestras
    datos = pd.DataFrame({
        'area': np.random.uniform(80, 300, n_samples),
        'banos': np.random.randint(1, 5, n_samples),
        'habitaciones': np.random.randint(1, 6, n_samples),
        'edad': np.random.uniform(0, 30, n_samples),
        'precio': np.zeros(n_samples)  # Se llenará basado en características
    })

    # Crear precios sintéticos basados en características (con algo de ruido)
    datos['precio'] = (
        datos['area'] * 1000 +
        datos['banos'] * 50000 +
        datos['habitaciones'] * 40000 -
        datos['edad'] * 2000 +
        np.random.normal(0, 20000, n_samples)  # Añadir ruido aleatorio
    )

    predictor = PredictorInmuebles(datos)
    predictor.preparar_datos(
        columnas_caracteristicas=['area', 'banos', 'habitaciones', 'edad'],
        columna_precio='precio'
    )
    predictor.entrenar_perceptron(capas_ocultas=(10, 5), max_iter=1000)
    predictor.evaluar_modelo()

    # Ejemplo de predicción
    nuevos_inmuebles = pd.DataFrame({
        'area': [160],
        'banos': [2],
        'habitaciones': [3],
        'edad': [5]
    })
    prediccion = predictor.predecir(nuevos_inmuebles)
    print(f"Precio predicho: ${prediccion[0][0]:,.2f}")

if __name__ == "__main__":
    main()
