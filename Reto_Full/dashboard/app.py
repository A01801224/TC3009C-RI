"""
Dependencias necesarias para correr este archivo. Instálalas en la terminal
antes de ejecutar `python app.py`:

    pip install flask pandas scikit-learn joblib openpyxl

- flask        -> servidor web y rutas (/, /predict, /predict_batch)
- pandas       -> manejo de dataframes y lectura de .csv/.xlsx
- scikit-learn -> necesario para que joblib pueda deserializar SpaceshipPreprocessor,
                  el RandomForestClassifier y el Pipeline+SVC
- joblib       -> cargar los modelos .pkl ya entrenados
- openpyxl     -> motor que pandas usa para leer archivos .xlsx (carga masiva)

pip install flask pandas scikit-learn joblib openpyxl

"""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # para poder importar preprocessing.py
from preprocessing import SpaceshipPreprocessor  # noqa: F401 (necesario para que joblib pueda deserializar)

app = Flask(__name__)

MODELOS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "modelos_full"

preprocesador = joblib.load(MODELOS_DIR / "preprocesador.pkl")
modelos = {
    "rf": joblib.load(MODELOS_DIR / "modelo_rf_final.pkl"),
    "svc": joblib.load(MODELOS_DIR / "modelo_svc_final.pkl"),
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    datos = request.get_json()
    modelo_elegido = datos.pop("modelo", "svc")  # "rf" o "svc"

    if modelo_elegido not in modelos:
        return jsonify({"error": f"Modelo '{modelo_elegido}' no reconocido"}), 400

    # Construir un DataFrame de una sola fila con el formato de test.csv
    fila = pd.DataFrame([datos])
    fila_procesada = preprocesador.transform(fila)

    modelo = modelos[modelo_elegido]
    prediccion = bool(modelo.predict(fila_procesada)[0])
    probabilidad = float(modelo.predict_proba(fila_procesada)[0][1])

    return jsonify({
        "modelo_usado": modelo_elegido,
        "transported": prediccion,
        "probabilidad": round(probabilidad, 4),
    })


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    archivo = request.files.get("archivo")
    modelo_elegido = request.form.get("modelo", "svc")

    if archivo is None:
        return jsonify({"error": "No se recibió ningún archivo"}), 400
    if modelo_elegido not in modelos:
        return jsonify({"error": f"Modelo '{modelo_elegido}' no reconocido"}), 400

    nombre = archivo.filename.lower()
    try:
        if nombre.endswith(".csv"):
            df = pd.read_csv(archivo)
        elif nombre.endswith(".xlsx"):
            df = pd.read_excel(archivo)
        else:
            return jsonify({"error": "Formato no soportado. Usa .csv o .xlsx"}), 400
    except Exception as e:
        return jsonify({"error": f"No se pudo leer el archivo: {e}"}), 400

    # Si no trae PassengerId (necesario para group_size), se genera uno
    # donde cada pasajero viaja solo (group_size=1) — mismo supuesto que en /predict
    if "PassengerId" not in df.columns:
        df["PassengerId"] = [f"{i:04d}_01" for i in range(1, len(df) + 1)]

    try:
        df_procesado = preprocesador.transform(df)
    except Exception as e:
        return jsonify({"error": f"Error al preprocesar: {e}"}), 400

    modelo = modelos[modelo_elegido]
    predicciones = modelo.predict(df_procesado)
    probabilidades = modelo.predict_proba(df_procesado)[:, 1]

    resultados = pd.DataFrame({
        "PassengerId": df["PassengerId"].values,
        "Transported": predicciones.astype(bool),
        "Probabilidad": probabilidades.round(4),
    })

    resumen = {
        "modelo_usado": modelo_elegido,
        "total": len(resultados),
        "transportados": int(resultados["Transported"].sum()),
        "no_transportados": int((~resultados["Transported"]).sum()),
        "probabilidad_promedio": round(float(resultados["Probabilidad"].mean()), 4),
        "resultados": resultados.to_dict(orient="records"),
    }
    return jsonify(resumen)

if __name__ == "__main__":
    app.run(debug=True)