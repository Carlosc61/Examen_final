# ================================
# 1. IMPORTACIONES
# ================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


df = pd.read_csv('../datasets/vehiculos_eda_columnas.csv')# Crear variable objetivo: 1 si tiene destino, 0 si no
df['Tiene_Destino'] = df['destino'].notnull().astype(int)



# Reemplazar nulos por texto
df['origen'] = df['origen'].fillna("Desconocido")
df['unidad'] = df['unidad'].fillna("Desconocido")
# Codificar variables categóricas
le_origen = LabelEncoder()
le_unidad = LabelEncoder()

df['origen_enc'] = le_origen.fit_transform(df['origen'])
df['unidad_enc'] = le_unidad.fit_transform(df['unidad'])

# ================================
# 3. MODELO
# ================================
X = df[['origen_enc', 'unidad_enc']]
y = df['Tiene_Destino']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print("Accuracy del modelo:", acc)

# ==============================
# ================================
# 4. DASH APP
# ================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Predicción Transporte"

app.layout = dbc.Container([

    html.H1("Motor Predictivo - Transporte",
            className="text-center mt-4"),

    html.Hr(),

    dbc.Row([

        # CONTROLES
        dbc.Col([

            html.Label("Selecciona Origen"),
            dcc.Dropdown(
                id='origen-input',
                options=[{'label': o, 'value': o}
                         for o in df['origen'].unique()],
                value=df['origen'].unique()[0]
            ),

            html.Br(),

            html.Label("Selecciona Unidad"),
            dcc.Dropdown(
                id='unidad-input',
                options=[{'label': u, 'value': u}
                         for u in df['unidad'].unique()],
                value=df['unidad'].unique()[0]
            ),

        ], md=4),

        # RESULTADO
        dbc.Col([
            dcc.Graph(id='gauge-chart')
        ], md=8)

    ])

], fluid=True)
# ================================
# 5. CALLBACK
# ================================
@app.callback(
    Output('gauge-chart', 'figure'),
    [Input('origen-input', 'value'),
     Input('unidad-input', 'value')]
)
def update_prediction(origen, unidad):

    origen_val = le_origen.transform([origen])[0]
    unidad_val = le_unidad.transform([unidad])[0]

    input_data = pd.DataFrame(
        [[origen_val, unidad_val]],
        columns=X.columns
    )

    prob = model.predict_proba(input_data)[0][1] * 100

    color = "red" if prob < 50 else "green"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Probabilidad de Tener Destino"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "#f8d7da"},
                {'range': [50, 100], 'color': "#d4edda"}
            ],
        }
    ))

    fig.update_layout(height=400)

    return fig

# ================================
# 6. RUN
# ================================
if __name__ == '__main__':
    app.run(debug=True, port=8061)