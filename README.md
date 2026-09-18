# Spaceship Titanic — TC3009C-RI

Repositorio del reto **Spaceship Titanic** (Kaggle) para la materia
**Inteligencia Artificial Avanzada para la Ciencia de Datos**, Tecnológico
de Monterrey.

**Equipo:** Ana Paula Moreno · David Tinoco Romero · Emilio Páez de la Mora ·
Alejandro Vázquez · Emilio Torres Paul

---

## 🎯 El problema

Predecir si un pasajero de la nave *Spaceship Titanic* fue transportado
(`Transported`, booleano) a otra dimensión durante una colisión con una
anomalía espacial, a partir de sus datos demográficos, de reserva y de
consumo a bordo. Es un problema de **clasificación binaria** sobre un
dataset prácticamente balanceado (~50.4% / 49.6%).

## 📁 Estructura del repositorio

```
TC3009C-RI/
├── Notebooks/              # Los tres avances del reto, en orden cronológico
│   ├── Reto_Primer_Avance_Spaceship_Titanic.ipynb
│   ├── Reto_Segundo_Avance.ipynb
│   └── Reto_Tercer_Avance_Spaceship_Titanic.ipynb
│
├── Reto_Full/               # Entregable consolidado + interfaz de uso del modelo
│   ├── Spaceship_Titanic_Solucion.ipynb
│   ├── preprocessing.py
│   └── dashboard/
│       ├── app.py
│       └── templates/
│           └── index.html
│
├── data/                    # Datos crudos, intermedios y modelos exportados
│   ├── train.csv / test.csv / sample_submission.csv
│   ├── spaceship_train_procesado.csv
│   ├── X_train.csv / X_test.csv / y_train.csv / y_test.csv
│   ├── modelo_*.pkl         # Modelos individuales de la Sección 4 del Avance 3
│   └── modelos_full/        # Modelos finales exportados desde Reto_Full
│       ├── preprocesador.pkl
│       ├── modelo_rf_final.pkl
│       └── modelo_svc_final.pkl
│
└── REORDENPROPUESTA.txt
```

## 📓 Los tres avances (`Notebooks/`)

| Avance | Contenido |
|---|---|
| **1 — Primer Avance** | Carga y comprensión del dataset, EDA (univariable, bivariable), selección/descarte de variables, manejo de faltantes (imputación lógica CryoSleep↔gasto, mediana, moda), feature engineering (`Deck`, `CabinNum`, `Side`, `CabinNumBin`, `group_size`, `HasSpent`), codificación (One-Hot) y escalado (`log1p` + `RobustScaler`). Dataset final: `(8693, 27)`, sin nulos. |
| **2 — Segundo Avance** | Selección de métricas (Accuracy, Precision, Recall, F1) y baseline (~0.5036). Entrenamiento de dos modelos de familias distintas (Regresión Logística vs. Random Forest) y elección de **Random Forest** como mejor modelo. Exploración individual de hiperparámetros, una subsección por integrante. |
| **3 — Tercer Avance** | Diagnóstico del modelo base (curvas de aprendizaje y de validación), tuning con `RandomizedSearchCV` + `StratifiedKFold`, evaluación final (F1-score, ROC-AUC, matriz de confusión). Componente individual: cada integrante tuneó un modelo de familia distinta (Regresión Logística, SVC, KNN, Naive Bayes, Red Neuronal). Exportación de modelos y conclusión final del equipo. |

## 📦 El entregable consolidado (`Reto_Full/`)

`Spaceship_Titanic_Solucion.ipynb` reúne, en un solo notebook, las
decisiones y resultados más importantes de los tres avances —
preprocesamiento, selección de modelo, diagnóstico/tuning, resultados
individuales y modelo final— con sus justificaciones, sin repetir el
detalle exhaustivo de cada avance (que se cita cuando aplica). Incluye
además dos apéndices con anotaciones del profesor:

- **Apéndice A** — `Deck` como variable ordinal (en vez de One-Hot),
  ordenada empíricamente por tasa de `Transported`.
- **Apéndice B** — Verificación estadística (ANOVA + Chi-cuadrado) de que
  `group_size` aporta información real y no es una variable redundante.

**`preprocessing.py`** contiene la clase `SpaceshipPreprocessor`
(`fit`/`transform`, estilo scikit-learn) con todo el feature engineering
del Avance 1 encapsulado como objeto reutilizable — tanto el notebook como
el dashboard la importan desde aquí, para no duplicar la lógica de
preprocesamiento.

### Modelo final: estrategia dual

El equipo no eligió un único modelo "ganador", sino una estrategia dual
basada en el contexto de uso:

| Modelo | F1-score | ROC-AUC | Rol |
|---|---|---|---|
| **SVC** (kernel `rbf`) | **0.8106** | 0.8945 | Implementación principal — mejor Recall, detecta más verdaderos positivos |
| **Random Forest tuneado** | 0.8050 | **0.8984** | Respaldo interpretable — reglas de decisión explícitas, mejor ordenamiento probabilístico |

## 🖥️ Dashboard (`Reto_Full/dashboard/`)

Interfaz local en Flask para usar los modelos ya entrenados, sin tener que
volver a correr ningún notebook.

**Predicción individual:** formulario con los datos de un pasajero y un
selector para elegir entre SVC o Random Forest.

**Predicción masiva:** carga un archivo `.csv` o `.xlsx` con varios
pasajeros (mismo formato que `test.csv`) y muestra un resumen (total,
transportados, probabilidad promedio) más una tabla de resultados
descargable.

### Cómo correrlo

```bash
pip install flask pandas scikit-learn joblib openpyxl
cd Reto_Full/dashboard
python app.py
```

Abre `http://127.0.0.1:5000` en el navegador. El servidor carga
`preprocesador.pkl`, `modelo_rf_final.pkl` y `modelo_svc_final.pkl` desde
`data/modelos_full/` al arrancar.

## 📄 `REORDENPROPUESTA.txt`

> *(Pendiente: pega aquí el contenido de este archivo o dime en una frase
> qué documenta, y completo esta sección — no pude leerlo directo desde
> GitHub.)*

## 🚀 Cómo reproducir el proyecto completo

1. Clona el repositorio y colócate en la raíz.
2. Corre los notebooks de `Notebooks/` en orden (1 → 2 → 3) si quieres ver
   el proceso completo avance por avance, **o** ve directo a
   `Reto_Full/Spaceship_Titanic_Solucion.ipynb` para el resumen
   consolidado con los modelos finales.
3. Corre el notebook de `Reto_Full/` de principio a fin para regenerar
   `preprocessing.py`-compatible `.pkl` en `data/modelos_full/`.
4. Levanta el dashboard (ver sección anterior) para predecir con los
   modelos ya entrenados.
