# Predicción del precio mensual de la gasolina — Regresión Lineal

## Contenido
| Archivo | Descripción |
|---|---|
| `Regresion_Precio_Gasolina.ipynb` | Notebook completo (ya ejecutado): EDA, limpieza, atípicos, normalización, selección de características, regresión lineal, métricas, intervalos 95 % y exportación del modelo |
| `app_gasolina.py` | Aplicativo web en Streamlit + Plotly para ingresar datos y obtener predicciones |
| `modelo_gasolina.pkl` | Modelo entrenado guardado con joblib (lo genera el notebook) |
| `dataset_precio_gasolina 1 5.csv` | Datos de entrada |
| `requirements.txt` | Dependencias de la app (Streamlit Cloud lo instala automáticamente) |
| `requirements-notebook.txt` | Dependencias extra para ejecutar el notebook |

## Uso
```bash
pip install -r requirements-notebook.txt
jupyter notebook Regresion_Precio_Gasolina.ipynb   # ejecutar todas las celdas
streamlit run app_gasolina.py                       # abre http://localhost:8501
```

## Resultados (partición 65 % entrenamiento / 35 % prueba)
| Métrica | Entrenamiento | Prueba (semilla 42) |
|---|---|---|
| R² | 0.926 | 0.843 |
| MAE (USD/galón) | 0.168 | 0.257 |
| MSE | 0.046 | 0.096 |

- R² ajustado de entrenamiento: 0.912.
- R² de prueba en 500 particiones aleatorias: mediana 0.872.
- Validación cruzada K-Fold (5 particiones): R² medio 0.871.

**Variables seleccionadas** por eliminación hacia atrás con OLS (p < 0.05): precio del petróleo, tipo de cambio, demanda, inflación, PIB, impuestos a combustibles y eventos geopolíticos.
**Variable excluida:** producción OPEP.

## Librerías usadas
pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn, ipywidgets, joblib, Streamlit y Plotly.
