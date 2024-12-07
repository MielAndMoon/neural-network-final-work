import numpy as np
import pandas as pd
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import matplotlib.pyplot as plt

class PredictorInmueblesH2O:
    def __init__(self, datos):
        """
        Inicializa el predictor de inmuebles con los datos proporcionados

        Parámetros:
        - datos: DataFrame con columnas de características del inmueble
        """
        self.datos = datos
        self.h2o_data = h2o.H2OFrame(datos)
        self.modelo = None

    def preparar_datos(self, columnas_caracteristicas, columna_precio):
        """
        Prepara los datos para entrenamiento dividiendo en características y target

        Parámetros:
        - columnas_caracteristicas: Lista de columnas para predecir
        - columna_precio: Columna con el precio del inmueble
        """
        self.caracteristicas = columnas_caracteristicas
        self.target = columna_precio

        # Dividir los datos en conjuntos de entrenamiento y prueba
        self.train, self.test = self.h2o_data.split_frame(ratios=[0.8], seed=42)

    def entrenar_perceptron(self, capas_ocultas=[10, 5], epochs=100):
        """
        Entrena un modelo de Perceptrón Multicapa con H2O

        Parámetros:
        - capas_ocultas: Lista con el número de neuronas por capa oculta
        - epochs: Número de épocas para el entrenamiento
        """
        self.modelo = H2ODeepLearningEstimator(
            hidden=capas_ocultas,
            epochs=epochs,
            activation="Rectifier",
            seed=42
        )
        self.modelo.train(
            x=self.caracteristicas, y=self.target, training_frame=self.train
        )

    def evaluar_modelo(self):
        """
        Evalúa el rendimiento del modelo en el conjunto de prueba

        Retorna:
        - Métricas de evaluación
        """
        rendimiento = self.modelo.model_performance(test_data=self.test)
        print(rendimiento)

    def predecir(self, nuevos_datos):
        """
        Realiza predicciones para nuevos datos

        Parámetros:
        - nuevos_datos: DataFrame con características de nuevos inmuebles

        Retorna:
        - Predicciones de precio
        """
        nuevos_datos_h2o = h2o.H2OFrame(nuevos_datos)
        predicciones = self.modelo.predict(nuevos_datos_h2o)
        return predicciones.as_data_frame()

    def graficar_resultados(self):
        """
        Genera un gráfico comparando las predicciones y los valores reales
        """
        predicciones = self.modelo.predict(self.test).as_data_frame()
        reales = self.test[self.target].as_data_frame()

        plt.figure(figsize=(10, 6))
        plt.title("Predicciones vs. Valores Reales")
        plt.plot(reales, label="Valores Reales", marker='o')
        plt.plot(predicciones, label="Predicciones", marker='x')
        plt.legend()
        plt.xlabel("Índice")
        plt.ylabel("Precio")
        plt.grid(True)
        plt.show()

def main():
    # Inicializar H2O
    h2o.init()

    # Crear un conjunto de datos sintético
    np.random.seed(42)
    n_samples = 100
    datos = pd.DataFrame({
        'area': np.random.uniform(80, 300, n_samples),
        'banos': np.random.randint(1, 5, n_samples),
        'habitaciones': np.random.randint(1, 6, n_samples),
        'edad': np.random.uniform(0, 30, n_samples),
        'precio': np.zeros(n_samples)
    })

    datos['precio'] = (
        datos['area'] * 1000 +
        datos['banos'] * 50000 +
        datos['habitaciones'] * 40000 -
        datos['edad'] * 2000 +
        np.random.normal(0, 20000, n_samples)
    )

    predictor = PredictorInmueblesH2O(datos)
    predictor.preparar_datos(
        columnas_caracteristicas=['area', 'banos', 'habitaciones', 'edad'],
        columna_precio='precio'
    )
    predictor.entrenar_perceptron(capas_ocultas=[10, 5], epochs=100)
    predictor.evaluar_modelo()

    # Ejemplo de predicción
    nuevos_inmuebles = pd.DataFrame({
        'area': [160],
        'banos': [2],
        'habitaciones': [3],
        'edad': [5]
    })
    prediccion = predictor.predecir(nuevos_inmuebles)
    print(f"\n\n\nPredicción del precio: {prediccion}\n\n\n")

    # Graficar resultados
    predictor.graficar_resultados()

    # Finalizar H2O
    h2o.shutdown(prompt=False)

if __name__ == "__main__":
    main()
