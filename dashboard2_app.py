import pandas as pd
import yfinance as yf
from hmmlearn import hmm
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import datetime as dt
from sklearn.preprocessing import StandardScaler # Import StandardScaler
import logging # Para un mejor manejo de logs/errores

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Funciones del Modelo HMM ---

def load_sp500_vix_data(start_date, end_date): # Función espera 'end_date'
    """
    Descarga datos del S&P 500 y del VIX para un rango de fechas dado,
    calcula sus retornos logarítmicos y cambios porcentuales, y los combina.
    """
    try:
        logging.info(f"Descargando datos desde {start_date} hasta {end_date} para ^SPX y ^VIX.")
        sp500 = yf.download('^SPX', start=start_date, end=end_date)
        vix = yf.download('^VIX', start=start_date, end=end_date)
        
        if sp500.empty or vix.empty:
            logging.warning("Uno o ambos DataFrames de Yahoo Finance están vacíos para el rango seleccionado.")
            return pd.DataFrame() # Retorna DataFrame vacío si no hay datos

        # Calcular retornos logarítmicos del S&P 500
        # Asegurarse de que 'Close' existe antes de intentar calcular
        if 'Close' not in sp500.columns:
            logging.error("Columna 'Close' no encontrada en los datos del S&P 500.")
            return pd.DataFrame()
        sp500['Log_Return_SPX'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
        
        # Calcular cambios porcentuales del VIX
        if 'Close' not in vix.columns:
            logging.error("Columna 'Close' no encontrada en los datos del VIX.")
            return pd.DataFrame()
        vix['Pct_Change_VIX'] = vix['Close'].pct_change()

        # Unir los datos en un solo DataFrame y eliminar NAs
        data = pd.concat([sp500['Log_Return_SPX'], vix['Pct_Change_VIX']], axis=1)
        data.dropna(inplace=True)
        
        if data.empty:
            logging.warning("DataFrame de características vacío después de calcular retornos y eliminar NAs.")
            return pd.DataFrame()

        logging.info(f"Datos de características cargados y procesados. Filas: {len(data)}")
        return data
    except Exception as e:
        logging.error(f"Error al cargar y procesar datos del S&P 500 y VIX: {e}")
        return pd.DataFrame() # Retorna DataFrame vacío en caso de error

def train_hmm_model(data, n_components=3, random_state=42):
    """
    Entrena un Hidden Markov Model (HMM) Gaussiano con los datos proporcionados.
    Normaliza los datos antes del entrenamiento.
    """
    if data.empty:
        logging.warning("No hay datos para entrenar el HMM. Saltando entrenamiento.")
        return None, None, None

    X = data[['Log_Return_SPX', 'Pct_Change_VIX']].values 

    # Normalizar los datos
    scaler = StandardScaler()
    try:
        if X.shape[0] == 0: # Doble comprobación por si data no estaba vacío pero X sí
            logging.warning("Input data for HMM training is empty after values extraction. Cannot train model.")
            return None, None, None
        X_scaled = scaler.fit_transform(X)
    except ValueError as e:
        logging.error(f"Error al escalar los datos: {e}. Asegúrese de que X no esté vacío o contenga NaN/inf.")
        return None, None, None

    # Entrenar el modelo
    # 'full' covariance_type es apropiado para múltiples características con posible correlación
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000, random_state=random_state)
    try:
        model.fit(X_scaled) # Entrenar con datos escalados
        logging.info(f"HMM Model trained successfully with {n_components} components.")
        return model, X_scaled, scaler 
    except Exception as e:
        logging.error(f"Error training HMM: {e}")
        return None, None, None

def predict_hmm_states(model, X_data_scaled):
    """
    Predice los estados ocultos (regímenes) utilizando el modelo HMM entrenado.
    """
    if model is None or X_data_scaled is None or X_data_scaled.shape[0] == 0:
        logging.warning("Modelo HMM no entrenado o datos escalados vacíos. No se pueden predecir estados.")
        return None, None
    try:
        log_prob, hidden_states = model.decode(X_data_scaled, algorithm="viterbi")
        probs = model.predict_proba(X_data_scaled)
        logging.info(f"Estados HMM predichos para {len(hidden_states)} muestras.")
        return hidden_states, probs
    except Exception as e:
        logging.error(f"Error al predecir los estados HMM: {e}")
        return None, None

# --- 2. Inicialización del Dashboard Dash ---

app = dash.Dash(__name__)

# --- 3. Cargar y Entrenar Datos Iniciales ---
# Estos datos se usan para inicializar el dashboard y establecer el rango de fechas
try:
    initial_start_date = '2020-01-01'
    initial_end_date = dt.date.today()
    logging.info(f"Cargando datos iniciales para el dashboard desde {initial_start_date} hasta {initial_end_date}")

    # **CORRECCIÓN APLICADA AQUÍ: 'end' cambiado a 'end_date'**
    initial_data_features = load_sp500_vix_data(start_date=initial_start_date, end_date=initial_end_date)
    
    # Cargar solo los precios de cierre del S&P 500 para el gráfico inicial
    sp500_raw_close = yf.download('^SPX', start=initial_start_date, end=initial_end_date)['Close']
    
    # Inicializar initial_sp500_close como un DataFrame vacío
    initial_sp500_close = pd.DataFrame(columns=['Close', 'Regime']) 
    
    # Alinear índices y convertir a DataFrame para añadir la columna 'Regime'
    if not initial_data_features.empty and not sp500_raw_close.empty:
        common_index = sp500_raw_close.index.intersection(initial_data_features.index)
        if not common_index.empty:
            # Convertir Series a DataFrame, asegurando la columna 'Close'
            initial_sp500_close = pd.DataFrame(sp500_raw_close.loc[common_index])
            initial_sp500_close.columns = ['Close'] # Asegurar el nombre de la columna
        else:
            logging.warning("No hay fechas comunes entre los datos de cierre del S&P500 y las características HMM iniciales.")
    else:
        logging.warning("No se pudieron cargar datos iniciales del S&P 500 para el gráfico o características HMM.")


    initial_model, initial_X_scaled, initial_scaler = train_hmm_model(initial_data_features, n_components=3)
    initial_states, initial_probs = predict_hmm_states(initial_model, initial_X_scaled)

    # Asignar etiquetas a los estados
    if initial_model and initial_states is not None and not initial_sp500_close.empty:
        means_spx = initial_model.means_[:, 0] 
        sorted_means_indices = np.argsort(means_spx)
        
        state_labels = {
            sorted_means_indices[0]: 'Bajista', # Media del SPX más baja
            sorted_means_indices[1]: 'Lateral/Transición',
            sorted_means_indices[2]: 'Alcista' # Media del SPX más alta
        }
        
        # Asegurarse de que la longitud de los estados coincide con los datos de cierre
        if len(initial_states) == len(initial_sp500_close):
            # Usar pd.Categorical con categorías explícitas para evitar errores si un estado no aparece en los datos actuales
            initial_sp500_close['Regime'] = pd.Categorical(initial_states, categories=sorted(state_labels.keys())).map(state_labels)
        else:
            logging.warning(f"Longitud de estados predichos ({len(initial_states)}) no coincide con la de los datos de cierre ({len(initial_sp500_close)}). Asignando 'N/A'.")
            initial_sp500_close['Regime'] = 'N/A'

        latest_initial_probs = initial_probs[-1] if initial_probs is not None and len(initial_probs) > 0 else np.array([0,0,0])
    else:
        initial_sp500_close['Regime'] = 'N/A'
        latest_initial_probs = np.array([0,0,0])
        logging.info("No se pudieron determinar los regímenes iniciales (modelo no entrenado o sin datos).")

    # Determinar el rango de fechas para el DatePickerRange
    min_date_allowed = initial_sp500_close.index.min() if not initial_sp500_close.empty else dt.date(2020, 1, 1)
    max_date_allowed = initial_sp500_close.index.max() if not initial_sp500_close.empty else dt.date.today()
    start_date_picker_default = max_date_allowed - pd.Timedelta(days=365*2) 
    if start_date_picker_default < min_date_allowed:
        start_date_picker_default = min_date_allowed

except Exception as e:
    logging.critical(f"Error crítico al cargar datos iniciales o entrenar modelo inicial: {e}")
    # Fallback si falla la carga inicial
    initial_sp500_close = pd.DataFrame(columns=['Close', 'Regime'])
    min_date_allowed = dt.date(2020, 1, 1)
    max_date_allowed = dt.date.today()
    start_date_picker_default = min_date_allowed
    latest_initial_probs = np.array([0,0,0])


# --- 4. Layout del Dashboard ---
app.layout = html.Div([
    html.H1("Dashboard de Regímenes de Mercado S&P 500 con VIX usando HMM", style={'textAlign': 'center'}),

    html.Div([
        dcc.DatePickerRange(
            id='date-range-picker',
            min_date_allowed=min_date_allowed,
            max_date_allowed=max_date_allowed,
            start_date=start_date_picker_default,
            end_date=max_date_allowed,
            display_format='DD/MM/YYYY'
        ),
        html.Button('Actualizar Datos', id='update-button', n_clicks=0, style={'marginLeft': '20px'})
    ], style={'textAlign': 'center', 'margin': '20px'}),

    dcc.Graph(id='sp500-regimes-graph', style={'height': '600px'}),

    html.Div(id='regime-probabilities', style={'textAlign': 'center', 'marginTop': '20px'}),

    html.Hr(),
    html.Footer([
        html.P("Desarrollado con Modelo Oculto de Markov (HMM)", style={'fontSize': '0.8em', 'color': 'gray'})
    ], style={'textAlign': 'center', 'marginTop': '20px'})
])

# --- 5. Callbacks para la Interactividad ---

@app.callback(
    [Output('sp500-regimes-graph', 'figure'),
     Output('regime-probabilities', 'children')],
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('update-button', 'n_clicks')]
)
def update_graph(start_date, end_date, n_clicks):
    if start_date is None or end_date is None:
        logging.warning("Fechas no seleccionadas. Usando rango predeterminado para la actualización.")
        start_date = initial_start_date
        end_date = initial_end_date

    # Convertir fechas a formato datetime.date si vienen como string
    if isinstance(start_date, str):
        # Usar fromisoformat para manejar 'T00:00:00' si está presente
        start_date = dt.datetime.fromisoformat(start_date).date()
    if isinstance(end_date, str):
        end_date = dt.datetime.fromisoformat(end_date).date()

    logging.info(f"Actualizando gráfico para el rango: {start_date} a {end_date}")

    # Cargar los datos del S&P 500 y VIX para las características del HMM
    # **CORRECCIÓN APLICADA AQUÍ: 'end' cambiado a 'end_date'**
    data_features = load_sp500_vix_data(start_date=start_date, end_date=end_date)
    
    # Cargar solo los precios de cierre del S&P 500 para el gráfico
    sp500_raw_close_series = yf.download('^SPX', start=start_date, end=end_date)['Close']
    
    # Inicializar sp500_close_for_graph como un DataFrame vacío
    sp500_close_for_graph = pd.DataFrame(columns=['Close', 'Regime']) 

    # Asegurarse de que el DataFrame de cierre esté alineado con los datos de las características
    if not data_features.empty and not sp500_raw_close_series.empty:
        # Alinear los índices: solo tomamos las fechas que están en ambos DataFrames
        common_index = sp500_raw_close_series.index.intersection(data_features.index)
        if not common_index.empty:
            # Convertir Series a DataFrame, asegurando la columna 'Close'
            sp500_close_for_graph = pd.DataFrame(sp500_raw_close_series.loc[common_index])
            sp500_close_for_graph.columns = ['Close'] # Asegurar el nombre de la columna
        else:
            logging.warning("No hay fechas comunes entre los datos de cierre del S&P500 y las características HMM para el rango seleccionado.")
    
    model, X_scaled, scaler = train_hmm_model(data_features, n_components=3) # Re-entrenar (en prod se cargaría)
    states, probs = predict_hmm_states(model, X_scaled) # Usar datos escalados para predicción

    latest_regime_prob_text = html.P("No se pudo entrenar el modelo o predecir estados (datos insuficientes o error de entrenamiento).")
    
    if model and states is not None and not sp500_close_for_graph.empty:
        # Re-asignar etiquetas de estado si el modelo se entrena de nuevo
        means_spx = model.means_[:, 0] 
        sorted_means_indices = np.argsort(means_spx)
        state_labels_map = {
            sorted_means_indices[0]: 'Bajista',
            sorted_means_indices[1]: 'Lateral/Transición',
            sorted_means_indices[2]: 'Alcista'
        }
        
        # CORRECCIÓN: Crear mapeo inverso de etiquetas a índices
        inverse_state_labels_map = {v: k for k, v in state_labels_map.items()}
        
        # Asignar el régimen al DataFrame usado para el gráfico
        if len(states) == len(sp500_close_for_graph):
            sp500_close_for_graph['Regime'] = pd.Categorical(states, categories=sorted(state_labels_map.keys())).map(state_labels_map)
        else:
            logging.warning(f"Longitud de estados predichos ({len(states)}) no coincide con la de los datos de cierre ({len(sp500_close_for_graph)}). Asignando 'N/A'.")
            sp500_close_for_graph['Regime'] = 'N/A'
        
        # Para las probabilidades del último día visible en el gráfico
        latest_probs = probs[-1] if probs is not None and len(probs) > 0 else np.array([0,0,0])

        # CORRECCIÓN: Usar el mapeo inverso correcto
        alcista_idx = inverse_state_labels_map.get('Alcista', -1)
        lateral_idx = inverse_state_labels_map.get('Lateral/Transición', -1)
        bajista_idx = inverse_state_labels_map.get('Bajista', -1)

        # CORRECCIÓN: Validar que los índices sean válidos y estén dentro del rango
        alcista_prob = latest_probs[alcista_idx] if alcista_idx != -1 and 0 <= alcista_idx < len(latest_probs) else 0.0
        lateral_prob = latest_probs[lateral_idx] if lateral_idx != -1 and 0 <= lateral_idx < len(latest_probs) else 0.0
        bajista_prob = latest_probs[bajista_idx] if bajista_idx != -1 and 0 <= bajista_idx < len(latest_probs) else 0.0

        # AÑADIDO: Log de debug para verificar valores
        logging.info(f"Debug - Mapeo de estados: {state_labels_map}")
        logging.info(f"Debug - Mapeo inverso: {inverse_state_labels_map}")
        logging.info(f"Debug - Probabilidades más recientes: {latest_probs}")
        logging.info(f"Debug - Índices: alcista={alcista_idx}, lateral={lateral_idx}, bajista={bajista_idx}")
        logging.info(f"Debug - Probabilidades finales: alcista={alcista_prob}, lateral={lateral_prob}, bajista={bajista_prob}")

        latest_regime_prob_text = html.Div([
            html.H3("Probabilidades de Régimen Actual:", style={'marginBottom': '10px'}),
            html.P(f"Alcista: {alcista_prob*100:.2f}%", style={'color': 'green'}),
            html.P(f"Lateral/Transición: {lateral_prob*100:.2f}%", style={'color': 'orange'}),
            html.P(f"Bajista: {bajista_prob*100:.2f}%", style={'color': 'red'})
        ])
    else:
        sp500_close_for_graph['Regime'] = 'N/A' # Si no hay modelo/estados, asignar 'N/A'
        
    fig = go.Figure()

    if not sp500_close_for_graph.empty:
        # Añadir el gráfico de precios del S&P 500
        fig.add_trace(go.Scatter(
            x=sp500_close_for_graph.index,
            y=sp500_close_for_graph['Close'],
            mode='lines',
            name='S&P 500 Cierre', # El nombre se mantiene para el 'hover'
            line=dict(color='blue', width=2),
            showlegend=False # <--- Añadido para ocultar de la leyenda
        ))

        # Añadir las áreas de color para los regímenes
        colors = {'Alcista': 'rgba(0,255,0,0.2)', 'Bajista': 'rgba(255,0,0,0.2)', 'Lateral/Transición': 'rgba(255,255,0,0.2)', 'N/A': 'rgba(128,128,128,0.1)'}
        
        current_regime = None
        start_date_regime = None
        shapes = []

        # Iterar sobre el DataFrame de cierre del S&P 500 para superponer los regímenes
        for i, (date, row) in enumerate(sp500_close_for_graph.iterrows()):
            regime = row['Regime']
            if regime != current_regime:
                if current_regime is not None:
                    shapes.append(
                        dict(
                            type='rect',
                            xref='x',
                            yref='paper',
                            x0=start_date_regime,
                            y0=0,
                            x1=sp500_close_for_graph.index[i-1], # Fecha final del período anterior
                            y1=1,
                            fillcolor=colors.get(current_regime, 'rgba(0,0,0,0)'),
                            layer='below',
                            line_width=0
                        )
                    )
                current_regime = regime
                start_date_regime = date
        
        # Añadir el último período de régimen
        if current_regime is not None:
            shapes.append(
                dict(
                    type='rect',
                    xref='x',
                    yref='paper',
                    x0=start_date_regime,
                    y0=0,
                    x1=sp500_close_for_graph.index[-1],
                    y1=1,
                    fillcolor=colors.get(current_regime, 'rgba(0,0,0,0)'),
                    layer='below',
                    line_width=0
                )
            )
        fig.update_layout(shapes=shapes)
    else:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            text="No hay datos disponibles para el rango de fechas seleccionado o no se pudo procesar.",
            showarrow=False,
            font=dict(size=16, color="red")
        )
        logging.warning("No hay datos para mostrar en el gráfico.")


    fig.update_layout(
        title='S&P 500 con Regímenes de Mercado HMM',
        xaxis_title='Fecha',
        yaxis_title='Precio de Cierre',
        hovermode='x unified',
        showlegend=True, # Mantener en True para que se muestre la leyenda de los regímenes
        xaxis_rangeslider_visible=True,
    )

    return fig, latest_regime_prob_text

# --- 6. Ejecutar el Dashboard ---
if __name__ == '__main__':
    app.run(debug=True)