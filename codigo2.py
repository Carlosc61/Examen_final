from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
try:
    df = pd.read_csv('../datasets/vehiculos_eda_columnas.csv')
except:
    url = "https://raw.githubusercontent.com/Carlosc61/Examen_final/main/vehiculos_eda_columnas.csv"
    df = pd.read_csv(url)

# Limpieza
df = df.dropna()

# Ajuste solo si existe la columna
if 'median_house_value' in df.columns:
    df = df[df['median_house_value'] < 500000]
    # ==========================================
# 3. GRÁFICOS
# ==========================================

# Conteo por origen
origen_counts = df['origen'].value_counts().reset_index()
origen_counts.columns = ['origen', 'cantidad']

fig_origen = px.bar(
    origen_counts,
    x='origen',
    y='cantidad',
    title='Rutas por Origen',
    color='cantidad'
)

# Conteo por destino
destino_counts = df['destino'].value_counts().reset_index()
destino_counts.columns = ['destino', 'cantidad']

fig_destino = px.bar(
    destino_counts,
    x='destino',
    y='cantidad',
    title='Rutas por Destino',
    color='cantidad'
)

# Distribución por unidad
fig_unidad = px.pie(
    df,
    names='unidad',
    title='Distribución por Unidad'
)
# ==========================================
# 4. APP
# ==========================================
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Análisis de Rutas"

app.layout = dbc.Container([

    dbc.Row([
        dbc.Col(html.H1(
            "Dashboard de Análisis de Rutas",
            className="text-center text-primary my-4"
        ), width=12),

        dbc.Col(html.P(
            "Exploración interactiva del dataset de rutas y transporte.",
            className="text-center text-muted"
        ), width=12),
    ]),

    html.Hr(),

    dbc.Tabs([

        # TAB 1 - ORIGEN
        dbc.Tab(label="1. Origen", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_origen), md=12)
            ], className="mt-4")
        ]),

        # TAB 2 - DESTINO
        dbc.Tab(label="2. Destino", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_destino), md=12)
            ], className="mt-4")
        ]),

        # TAB 3 - UNIDAD
        dbc.Tab(label="3. Unidad", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_unidad), md=12)
            ], className="mt-4")
        ]),

        # TAB 4 - BUSCADOR
        dbc.Tab(label="4. Buscador", children=[

            dbc.Row([
                dbc.Col([
                    html.Label("Buscar por nombre:"),
                    dcc.Input(
                        id='input-busqueda',
                        type='text',
                        placeholder='Escribe un nombre...',
                        className='form-control'
                    )
                ], md=6)
            ], className="mt-4"),

            dbc.Row([
                dbc.Col(html.Div(id='tabla-resultados'), md=12)
            ], className="mt-4")

        ])

    ])

], fluid=True)

# ==========================================
# 5. CALLBACK
# ==========================================
@app.callback(
    Output('tabla-resultados', 'children'),
    Input('input-busqueda', 'value')
)
def filtrar_datos(texto):

    if not texto:
        df_filtrado = df.head(10)
    else:
        df_filtrado = df[df['nombre'].str.contains(texto, case=False, na=False)]

    return dbc.Table.from_dataframe(
        df_filtrado,
        striped=True,
        bordered=True,
        hover=True
    )
# ==========================================
# 6. RUN
# ==========================================
if __name__ == '__main__':
    app.run(debug=True, port=8060)