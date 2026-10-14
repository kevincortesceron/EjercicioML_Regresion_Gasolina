"""
Aplicativo de predicción del precio mensual de la gasolina (USD/galón).
Librerías: Streamlit (interfaz), Plotly (gráficos), statsmodels (modelo e intervalos),
scikit-learn (normalización), pandas/numpy (datos), joblib (carga del modelo).

Ejecutar:  streamlit run app_gasolina.py
Requiere el archivo modelo_gasolina.pkl generado por el notebook.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="Predictor precio gasolina", page_icon="⛽", layout="wide")

ETIQUETAS = {
    "Precio_Petroleo_USD_Barril": ("Precio del petróleo", "USD/barril"),
    "Demanda_miles_barriles_dia": ("Demanda", "miles de barriles/día"),
    "Inflacion_Porcentaje": ("Inflación", "%"),
    "PIB_Porcentaje": ("Crecimiento del PIB", "%"),
    "Impuestos_Combustibles_Porcentaje": ("Impuestos a combustibles", "%"),
    "Eventos_Geopoliticos": ("Evento geopolítico en el mes", "0 = no, 1 = sí"),
    "Tipo_Cambio_USD_COP": ("Tipo de cambio", "COP por USD"),
    "Produccion_OPEP_millones_dia": ("Producción OPEP", "millones barriles/día"),
}


@st.cache_resource
def cargar_modelo(ruta="modelo_gasolina.pkl"):
    return joblib.load(Path(__file__).parent / ruta)


def predecir(df_entrada: pd.DataFrame, b: dict, alpha: float) -> pd.DataFrame:
    """Aplica winsorización, normalización y devuelve predicción + intervalos."""
    x = df_entrada[b["features"]].astype(float).copy()
    for c, (lo, hi) in b["limites_clip"].items():
        x[c] = x[c].clip(lo, hi)
    xs = pd.DataFrame(b["scaler"].transform(x), columns=b["features"])
    fr = b["modelo_ols"].get_prediction(sm.add_constant(xs, has_constant="add")).summary_frame(alpha=alpha)
    return pd.DataFrame({
        "Predicción_USD_galón": fr["mean"].values,
        "IC_media_inf": fr["mean_ci_lower"].values,
        "IC_media_sup": fr["mean_ci_upper"].values,
        "IP_inf": fr["obs_ci_lower"].values,
        "IP_sup": fr["obs_ci_upper"].values,
    }, index=df_entrada.index)


try:
    b = cargar_modelo()
except FileNotFoundError:
    st.error("No se encontró **modelo_gasolina.pkl**. Ejecute primero el notebook para generarlo.")
    st.stop()

FEATURES = b["features"]

# ---------------- Barra lateral: entradas ----------------
st.sidebar.header("Variables del mes")
st.sidebar.caption("Ingrese las condiciones económicas, energéticas y geopolíticas.")
entrada = {}
for c in FEATURES:
    nombre, unidad = ETIQUETAS.get(c, (c, ""))
    r = b["rangos"][c]
    if c == "Eventos_Geopoliticos":
        entrada[c] = 1 if st.sidebar.toggle(nombre, value=False, help=unidad) else 0
    elif c == "Impuestos_Combustibles_Porcentaje":
        entrada[c] = st.sidebar.slider(f"{nombre} ({unidad})", 0, 40, int(r["mediana"]))
    else:
        paso = 1.0 if r["max"] > 1000 else 0.1
        entrada[c] = st.sidebar.number_input(f"{nombre} ({unidad})", value=round(r["mediana"], 2), step=paso,
                                             help=f"Rango histórico: {r['min']:.2f} – {r['max']:.2f}")
confianza = st.sidebar.select_slider("Nivel de confianza", options=[0.80, 0.90, 0.95, 0.99], value=0.95,
                                     format_func=lambda v: f"{v:.0%}")
alpha = 1 - confianza

fuera = [ETIQUETAS[c][0] for c in FEATURES
         if not (b["rangos"][c]["min"] <= entrada[c] <= b["rangos"][c]["max"])]

# ---------------- Encabezado ----------------
st.title("⛽ Predicción del precio mensual de la gasolina")
st.caption("Modelo de regresión lineal entrenado con datos mensuales 2018–2023. "
           "Apoyo a decisiones de política energética y estrategia comercial.")

tab_pred, tab_esc, tab_lote, tab_modelo = st.tabs(
    ["Predicción", "Análisis de escenarios", "Predicción por lote (CSV)", "Sobre el modelo"])

# ---------------- Pestaña 1: predicción individual ----------------
with tab_pred:
    p = predecir(pd.DataFrame([entrada]), b, alpha).iloc[0]
    if fuera:
        st.warning("Valores fuera del rango histórico (la predicción es una extrapolación): " + ", ".join(fuera))

    c1, c2, c3 = st.columns(3)
    c1.metric("Precio estimado", f"{p['Predicción_USD_galón']:.2f} USD/galón")
    c2.metric(f"Intervalo de predicción {confianza:.0%}", f"{p['IP_inf']:.2f} – {p['IP_sup']:.2f}",
              help="Rango donde caería el precio de un mes concreto con esas condiciones.")
    c3.metric(f"IC {confianza:.0%} de la media", f"{p['IC_media_inf']:.2f} – {p['IC_media_sup']:.2f}",
              help="Rango del precio promedio esperado para esas condiciones.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p["IP_inf"], p["IP_sup"]], y=["Precio", "Precio"], mode="lines",
                             line=dict(width=14, color="rgba(42,122,176,0.25)"),
                             name=f"Intervalo de predicción {confianza:.0%}"))
    fig.add_trace(go.Scatter(x=[p["IC_media_inf"], p["IC_media_sup"]], y=["Precio", "Precio"], mode="lines",
                             line=dict(width=14, color="rgba(42,122,176,0.7)"), name=f"IC {confianza:.0%} media"))
    fig.add_trace(go.Scatter(x=[p["Predicción_USD_galón"]], y=["Precio"], mode="markers",
                             marker=dict(size=18, color="#d1495b", symbol="diamond"), name="Predicción"))
    hist = pd.Series(b["historico"]["Precio_Gasolina_USD_Galon"])
    fig.add_vrect(x0=hist.min(), x1=hist.max(), fillcolor="grey", opacity=0.07, line_width=0,
                  annotation_text="rango histórico", annotation_position="top left")
    fig.update_layout(height=230, xaxis_title="USD/galón", yaxis_visible=False,
                      margin=dict(l=10, r=10, t=30, b=40), legend=dict(orientation="h", y=-0.5))
    st.plotly_chart(fig, width="stretch")

    st.subheader("¿Qué aporta cada variable?")
    xs = (pd.DataFrame([entrada])[FEATURES].astype(float).clip(
        lower=pd.Series({c: b["limites_clip"].get(c, (-np.inf, np.inf))[0] for c in FEATURES}),
        upper=pd.Series({c: b["limites_clip"].get(c, (-np.inf, np.inf))[1] for c in FEATURES}), axis=1))
    xs = pd.Series(b["scaler"].transform(xs)[0], index=FEATURES)
    aporte = pd.Series({c: b["coeficientes"][c] * xs[c] for c in FEATURES}).sort_values()
    fig_a = go.Figure(go.Bar(x=aporte.values, y=[ETIQUETAS[c][0] for c in aporte.index], orientation="h",
                             marker_color=["#d1495b" if v > 0 else "#2a7ab0" for v in aporte.values]))
    fig_a.update_layout(height=300, xaxis_title=f"USD/galón respecto al precio base ({b['coeficientes']['const']:.2f})",
                        margin=dict(l=10, r=10, t=10, b=40))
    st.plotly_chart(fig_a, width="stretch")
    st.caption("Rojo: empuja el precio hacia arriba frente a un mes promedio. Azul: lo empuja hacia abajo.")

# ---------------- Pestaña 2: escenarios ----------------
with tab_esc:
    st.markdown("Varíe una variable manteniendo las demás fijas en los valores de la barra lateral.")
    opciones = [c for c in FEATURES if c != "Eventos_Geopoliticos"]
    var = st.selectbox("Variable a analizar", opciones, format_func=lambda c: ETIQUETAS[c][0])
    r = b["rangos"][var]
    grid = np.linspace(r["min"], r["max"], 60)
    esc = pd.DataFrame([entrada] * len(grid)); esc[var] = grid
    pe = predecir(esc, b, alpha)
    fig_e = go.Figure([
        go.Scatter(x=np.r_[grid, grid[::-1]], y=np.r_[pe["IP_sup"], pe["IP_inf"][::-1]], fill="toself",
                   fillcolor="rgba(42,122,176,0.18)", line=dict(width=0), name=f"Intervalo predicción {confianza:.0%}"),
        go.Scatter(x=grid, y=pe["Predicción_USD_galón"], line=dict(color="#d1495b", width=3), name="Predicción"),
        go.Scatter(x=[entrada[var]], y=[predecir(pd.DataFrame([entrada]), b, alpha).iloc[0, 0]], mode="markers",
                   marker=dict(size=12, color="black"), name="Valor actual"),
    ])
    fig_e.update_layout(height=420, xaxis_title=f"{ETIQUETAS[var][0]} ({ETIQUETAS[var][1]})",
                        yaxis_title="Precio gasolina (USD/galón)", margin=dict(t=20))
    st.plotly_chart(fig_e, width="stretch")

    con = predecir(pd.DataFrame([{**entrada, "Eventos_Geopoliticos": 1}]), b, alpha).iloc[0, 0]
    sin = predecir(pd.DataFrame([{**entrada, "Eventos_Geopoliticos": 0}]), b, alpha).iloc[0, 0]
    st.info(f"Con las condiciones actuales, un **evento geopolítico** cambia el precio estimado de "
            f"{sin:.2f} a {con:.2f} USD/galón (**{con - sin:+.2f}**).")

# ---------------- Pestaña 3: lote ----------------
with tab_lote:
    st.markdown("Suba un CSV con las columnas: " + ", ".join(f"`{c}`" for c in FEATURES)
                + ". Puede incluir otras columnas (p. ej. `Fecha`); se conservan en el resultado.")
    plantilla = pd.DataFrame([{c: b["rangos"][c]["mediana"] for c in FEATURES}])
    st.download_button("Descargar plantilla CSV", plantilla.to_csv(index=False), "plantilla_gasolina.csv", "text/csv")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        datos = pd.read_csv(archivo)
        faltan = [c for c in FEATURES if c not in datos.columns]
        if faltan:
            st.error("Faltan columnas: " + ", ".join(faltan))
        else:
            n_nulos = int(datos[FEATURES].isna().sum().sum())
            if n_nulos:
                st.warning(f"{n_nulos} valores faltantes imputados con la mediana histórica.")
                datos[FEATURES] = datos[FEATURES].fillna({c: b["rangos"][c]["mediana"] for c in FEATURES})
            resultado = pd.concat([datos, predecir(datos, b, alpha).round(3)], axis=1)
            st.dataframe(resultado, width="stretch")
            if "Precio_Gasolina_USD_Galon" in resultado:
                err = resultado["Precio_Gasolina_USD_Galon"] - resultado["Predicción_USD_galón"]
                m1, m2, m3 = st.columns(3)
                ss = ((resultado["Precio_Gasolina_USD_Galon"] - resultado["Precio_Gasolina_USD_Galon"].mean()) ** 2).sum()
                m1.metric("R²", f"{1 - (err ** 2).sum() / ss:.3f}")
                m2.metric("MAE", f"{err.abs().mean():.3f}")
                m3.metric("MSE", f"{(err ** 2).mean():.3f}")
            st.download_button("Descargar predicciones", resultado.to_csv(index=False),
                               "predicciones_gasolina.csv", "text/csv")

# ---------------- Pestaña 4: modelo ----------------
with tab_modelo:
    st.subheader("Desempeño")
    mp, mc = b["metricas_prueba"], b["metricas_cv"]
    me = b.get("metricas_entrenamiento", {})
    st.table(pd.DataFrame({"Entrenamiento (65 %)": me, "Prueba (35 %)": mp,
                           "Validación cruzada K-Fold (media)": mc}).dropna(axis=1, how="all").T
             .rename(columns={"R2": "R²"}).style.format("{:.3f}"))
    if "r2_ajustado" in b:
        st.caption(f"R² ajustado de entrenamiento: {b['r2_ajustado']:.3f}")
    st.subheader("Coeficientes estandarizados")
    coefs = pd.Series({ETIQUETAS[c][0]: b["coeficientes"][c] for c in FEATURES}).sort_values()
    st.plotly_chart(go.Figure(go.Bar(x=coefs.values, y=coefs.index, orientation="h", marker_color="#2a7ab0"))
                    .update_layout(height=300, xaxis_title="USD/galón por +1 desviación estándar",
                                   margin=dict(t=10)), width="stretch")
    st.subheader("Histórico del precio")
    h = pd.DataFrame(b["historico"])
    st.plotly_chart(go.Figure(go.Scatter(x=h["Fecha"], y=h["Precio_Gasolina_USD_Galon"], mode="lines+markers"))
                    .update_layout(height=300, yaxis_title="USD/galón", margin=dict(t=10)),
                    width="stretch")
    st.caption("Variable excluida por no ser estadísticamente significativa (eliminación hacia atrás, p > 0.05): producción OPEP.")
