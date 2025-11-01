# src/dashboard.py
"""
Dashboard orientado para profissionais da saúde:
'Sistema Preditivo de Surtos — Mpox (Brasil)'

Objetivos do layout:
- Apresentar indicadores e previsões com linguagem acessível.
- Mostrar evidência técnica (métricas, gráfico de clusters) em abas separadas.
- Permitir exportar dados e previsões em CSV.
"""

import io
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Paths (mesmo padrão do seu projeto)
AGG_PATH = Path("data/processed/aggregated")
MODEL_PATH = Path("data/models")

def safe_load_latest_agg():
    arquivos = sorted(AGG_PATH.glob("casos_por_estado_mes_*.csv"), reverse=True)
    if not arquivos:
        return None
    df = pd.read_csv(arquivos[0])
    df['ano_mes'] = pd.to_datetime(df['ano_mes'], format='%Y-%m')
    df['ano'] = df['ano_mes'].dt.year
    df['mes'] = df['ano_mes'].dt.month
    return df

# Carregar dados e modelos (com proteção)
df = safe_load_latest_agg()

modelo = None
estado_map = {}
try:
    modelo = joblib.load(MODEL_PATH / "mpox_casos_rf.joblib")
    estado_map = joblib.load(MODEL_PATH / "estado_map.joblib")
except Exception as e:
    print("[WARN] Modelo RF não carregado:", e)

# Funções utilitárias (mantidas do pipeline)
def gerar_features_serie_por_estado(df, janela=6):
    estados = []
    feats = []
    for estado, g in df.groupby('sg_uf_not'):
        s = g.sort_values('ano_mes')['casos']
        s_roll = s.rolling(window=janela, min_periods=1).mean().fillna(0)
        media = s_roll.mean()
        var = s_roll.var() if not np.isnan(s_roll.var()) else 0.0
        idx = np.arange(len(s_roll))
        slope = float(np.polyfit(idx, s_roll, 1)[0]) if len(s_roll) > 1 else 0.0
        feats.append([media, var, slope])
        estados.append(estado)
    df_feats = pd.DataFrame(feats, columns=['media_6m','var_6m','tendencia_6m'])
    df_feats['sg_uf_not'] = estados
    df_feats = df_feats[['sg_uf_not','media_6m','var_6m','tendencia_6m']]
    return df_feats

def criar_features_para_prever(df, estado_map, modelo, n_lags=3, meses_prev=3, estados=None):
    if modelo is None:
        return pd.DataFrame()
    previsoes = []
    estados_iter = estados if estados is not None else df['sg_uf_not'].unique()
    for estado in estados_iter:
        dados_estado = df[df['sg_uf_not']==estado].sort_values("ano_mes")
        if len(dados_estado) < n_lags:
            continue
        lags = list(dados_estado['casos'].values)[-n_lags:]
        ultima_data = dados_estado['ano_mes'].max()
        for i in range(meses_prev):
            proxima_data = (ultima_data + pd.DateOffset(months=1)).replace(day=1)
            entrada = {
                'sg_uf_not': estado_map.get(estado, -1),
                'ano': proxima_data.year,
                'mes': proxima_data.month,
                **{f'lag_{j+1}': lags[-(j+1)] for j in range(n_lags)}
            }
            pred = modelo.predict(pd.DataFrame([entrada]))[0]
            previsoes.append({
                'sg_uf_not': estado,
                'ano_mes': proxima_data,
                'casos': float(pred),
                'tipo': 'Previsto'
            })
            lags.append(pred)
            lags = lags[-n_lags:]
            ultima_data = proxima_data
    return pd.DataFrame(previsoes)

# Preparar df_total (histórico + previsões básicas)
if df is not None and modelo is not None:
    try:
        df_prev = criar_features_para_prever(df, estado_map, modelo, meses_prev=3)
        df_prev['ano'] = df_prev['ano_mes'].dt.year
        df_prev['mes'] = df_prev['ano_mes'].dt.month
        df_hist = df.copy()
        df_hist['tipo'] = 'Histórico'
        df_total = pd.concat([df_hist, df_prev], ignore_index=True)
    except Exception:
        df_total = df.copy()
else:
    df_total = df.copy() if df is not None else pd.DataFrame()

# Inicializar Dash
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, title="Mpox - Painel de Saúde")
server = app.server

# Layout (linguagem simples, orientada a ação)
app.layout = html.Div([
    html.Header([
        html.H1("Painel de Monitoramento e Previsão — Mpox (Brasil)"),
        html.P("Ferramenta para profissionais de saúde: monitore a situação atual, veja previsões e identifique grupos de risco. (Explicações simples em cada aba.)")
    ], style={'textAlign':'center','padding':'12px','backgroundColor':'#f4f6f8'}),

    html.Div(id="kpis", style={'display':'flex','gap':'12px','padding':'12px','justifyContent':'center','flexWrap':'wrap'}),

    dcc.Tabs([
        dcc.Tab(label="Visão Geral (Resumo)", children=[
            html.Div([
                html.H3("Resumo rápido"),
                html.P("Mapa interativo com intensidade de casos e ranking dos estados com mais registros."),
                html.Div([
                    html.Div([
                        html.Label("Escolha o ano:"),
                        dcc.Dropdown(
                            id='vg-ano',
                            options=[{'label':str(a),'value':a} for a in sorted(df['ano'].unique())] if df is not None else [],
                            value=df['ano'].max() if df is not None else None,
                            style={'width':'200px'}
                        ),
                        dcc.Graph(id='vg-mapa')
                    ], style={'flex':'1','padding':'10px'}),

                    html.Div([
                        html.H4("Ranking por Estado"),
                        html.P("Os 10 estados com mais casos no ano selecionado:"),
                        dcc.Graph(id='vg-ranking')
                    ], style={'width':'400px','padding':'10px','border':'1px solid #eee','borderRadius':'6px','background':'#fff'})
                ], style={'display':'flex','gap':'20px'})
            ], style={'padding':'20px'})
        ]),

        dcc.Tab(label="Análise Supervisionada (Previsão)", children=[
            html.Div([
                html.Div([
                    html.H4("Objetivo"),
                    html.P("Prever o número de casos no próximo mês para um estado. Útil para planejar vigilância e alocação de recursos."),
                    html.Label("Escolha o estado:"),
                    dcc.Dropdown(id='sup-estado', options=[{'label':u,'value':u} for u in sorted(df['sg_uf_not'].unique())] if df is not None else [], value=sorted(df['sg_uf_not'].unique())[0] if df is not None else None, style={'width':'260px'}),
                    html.Label("Meses para prever:"),
                    dcc.Slider(id='sup-meses', min=1, max=6, step=1, value=3, marks={i:str(i) for i in range(1,7)}),
                    html.Br(),
                    html.Button("Baixar previsões (CSV)", id='btn-download-pre')
                ], style={'width':'320px','padding':'12px','border':'1px solid #eee','borderRadius':'6px','background':'#fff'}),

                html.Div([
                    dcc.Graph(id='sup-graph'),
                    html.Div(id='sup-texto', style={'padding':'8px','border':'1px solid #f0f0f0','borderRadius':'6px','background':'#fafafa'})
                ], style={'flex':'1','paddingLeft':'20px'})
            ], style={'display':'flex','gap':'20px','padding':'20px'})
        ]),

        dcc.Tab(label="Análise Não Supervisionada (Grupos / Clusters)", children=[
            html.Div([
                html.Div([
                    html.H4("O que é"),
                    html.P("Aqui agrupamos estados com comportamento epidemiológico semelhante. Grupos semelhantes ajudam a direcionar ações conjuntas."),
                    html.Label("Escolha o número de grupos (k):"),
                    dcc.Slider(id='clus-k', min=2, max=6, step=1, value=3, marks={i:str(i) for i in range(2,7)}),
                    html.Br(),
                    html.Button("Baixar clusters (CSV)", id='btn-download-clusters')
                ], style={'width':'320px','padding':'12px','border':'1px solid #eee','borderRadius':'6px','background':'#fff'}),

                html.Div([
                    dcc.Graph(id='clus-scatter'),
                    dcc.Graph(id='clus-table')
                ], style={'flex':'1','paddingLeft':'20px'})
            ], style={'display':'flex','gap':'20px','padding':'20px'})
        ]),

        dcc.Tab(label="Interpretando Resultados (Para Saúde)", children=[
            html.Div([
                html.H3("Como usar este painel (guia rápido)"),
                html.Ul([
                    html.Li("Visão Geral: ver rapidamente quais estados têm maior intensidade e se houve aumento no último mês."),
                    html.Li("Previsão: ao selecionar um estado, observe o próximo mês previsto e compare com o histórico."),
                    html.Li("Clusters: estados do mesmo grupo tendem a evoluir de forma parecida — útil para ações regionais conjuntas."),
                ]),
                html.H4("Interpretação das métricas"),
                html.P("MAE: erro médio absoluto entre casos reais e previstos."),
                html.P("R²: proporção da variação dos casos explicada pelo modelo."),
                html.H4("Limitações"),
                html.Ul([
                    html.Li("Os modelos dependem da qualidade dos dados."),
                    html.Li("As previsões são indicativas — devem ser validadas com informações locais.")
                ])
            ], style={'padding':'20px','lineHeight':'1.6'})
        ])
    ]),
    dcc.Download(id="download-csv")
], style={'fontFamily':'Arial, sans-serif','maxWidth':'1100px','margin':'0 auto'})

# KPI cards
@app.callback(Output('kpis','children'), Input('vg-ano','value'))
def build_kpis(_):
    if df is None:
        return html.Div("Dados não disponíveis.")
    total = int(df['casos'].sum())
    ultimo_mes = df['ano_mes'].max()
    ultimo_total = int(df[df['ano_mes']==ultimo_mes]['casos'].sum())
    maior_estado = df.groupby('sg_uf_not')['casos'].sum().idxmax()
    return [
        html.Div([html.H5("Total de casos (histórico)"), html.H2(f"{total}")], style={'padding':'10px','border':'1px solid #eee','borderRadius':'6px','width':'260px','background':'#fff'}),
        html.Div([html.H5(f"Casos no último mês ({ultimo_mes.strftime('%Y-%m')})"), html.H2(f"{ultimo_total}")], style={'padding':'10px','border':'1px solid #eee','borderRadius':'6px','width':'320px','background':'#fff'}),
        html.Div([html.H5("Estado com mais casos (hist.)"), html.H2(f"{maior_estado}")], style={'padding':'10px','border':'1px solid #eee','borderRadius':'6px','width':'260px','background':'#fff'})
    ]

# Visão Geral - mapa e ranking
@app.callback(
    Output('vg-mapa','figure'),
    Output('vg-ranking','figure'),
    Input('vg-ano','value')
)
def update_vg_map(ano):
    if df is None or ano is None:
        return go.Figure(), go.Figure()
    df_map = df[df['ano']==ano].groupby('sg_uf_not', as_index=False)['casos'].sum()
    fig_map = px.choropleth(
        df_map,
        geojson="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson",
        locations='sg_uf_not', featureidkey='properties.sigla',
        color='casos', title=f'Casos agregados por estado - {ano}'
    )
    fig_map.update_geos(fitbounds="locations", visible=False)

    # Ranking top 10 estados
    df_rank = df_map.sort_values('casos', ascending=False).head(10)
    fig_rank = px.bar(
        df_rank,
        x='casos', y='sg_uf_not',
        orientation='h',
        title='Top 10 Estados com Mais Casos',
        text='casos'
    )
    fig_rank.update_traces(textposition='outside')
    fig_rank.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
    return fig_map, fig_rank

# Previsão
@app.callback(Output('sup-graph','figure'), Output('sup-texto','children'),
              Input('sup-estado','value'), Input('sup-meses','value'))
def sup_predict(estado, meses):
    if df is None:
        return go.Figure(), "Dados não disponíveis."
    if modelo is None:
        return go.Figure(), "Modelo de previsão indisponível. Execute src/ml_pipeline.py para treinar."
    df_prev = criar_features_para_prever(df, estado_map, modelo, meses_prev=meses, estados=[estado])
    if df_prev.empty:
        return go.Figure(), "Dados insuficientes para previsão neste estado."
    df_hist_local = df[df['sg_uf_not']==estado][['ano_mes','casos']].copy()
    df_hist_local['tipo'] = 'Observado'
    df_prev['tipo'] = 'Previsto'
    df_plot = pd.concat([df_hist_local, df_prev[['ano_mes','casos','tipo']]], ignore_index=True).sort_values('ano_mes')
    fig = px.line(df_plot, x='ano_mes', y='casos', color='tipo', markers=True, title=f"Histórico e previsão - {estado}")
    ultimo_obs = int(df_hist_local['casos'].iloc[-1])
    proximo_pred = int(df_prev['casos'].iloc[0])
    pct = ((proximo_pred-ultimo_obs)/ultimo_obs*100) if ultimo_obs>0 else np.nan
    texto = f"Último mês observado: {ultimo_obs} casos. Próximo mês previsto: {proximo_pred} casos ({pct:.1f}% vs último mês)."
    return fig, texto

# Clustering
@app.callback(Output('clus-scatter','figure'), Output('clus-table','figure'), Input('clus-k','value'))
def clus_update(k):
    if df is None:
        return go.Figure(), go.Figure()
    df_feats = gerar_features_serie_por_estado(df, janela=6)
    Xnum = df_feats[['media_6m','var_6m','tendencia_6m']].values
    imputer = SimpleImputer(strategy='mean')
    X_imp = imputer.fit_transform(Xnum)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_imp)
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(Xp)
    df_feats['cluster'] = clusters
    df_plot = pd.DataFrame(Xp, columns=['pca1','pca2'])
    df_plot['cluster'] = df_feats['cluster'].astype(str)
    df_plot['estado'] = df_feats['sg_uf_not']
    fig_scatter = px.scatter(df_plot, x='pca1', y='pca2', color='cluster', hover_data=['estado'], title=f'Grupos de estados (k={k})')
    df_tab = df_feats.sort_values('cluster')
    table = go.Figure(data=[go.Table(header=dict(values=list(df_tab.columns)),
                                     cells=dict(values=[df_tab[c] for c in df_tab.columns]))])
    return fig_scatter, table

# Downloads
@app.callback(Output("download-csv","data"),
              Input("btn-download-pre","n_clicks"),
              Input("btn-download-clusters","n_clicks"),
              State("sup-estado","value"),
              State("sup-meses","value"),
              State("clus-k","value"),
              prevent_initial_call=True)
def download(btn_pre, btn_clu, sup_estado, sup_meses, k):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    btn_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if btn_id == 'btn-download-pre':
        if modelo is None:
            return dcc.send_data_frame(pd.DataFrame().to_csv, filename="previsoes.csv")
        df_pre = criar_features_para_prever(df, estado_map, modelo, meses_prev=sup_meses, estados=[sup_estado])
        return dcc.send_data_frame(df_pre.to_csv, filename=f"previsoes_{sup_estado}.csv", index=False)
    if btn_id == 'btn-download-clusters':
        df_feats = gerar_features_serie_por_estado(df, janela=6)
        imputer = SimpleImputer(strategy='mean')
        X_imp = imputer.fit_transform(df_feats[['media_6m','var_6m','tendencia_6m']].values)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_imp)
        pca = PCA(n_components=2, random_state=42)
        Xp = pca.fit_transform(Xs)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(Xp)
        df_feats['cluster'] = clusters
        return dcc.send_data_frame(df_feats.to_csv, filename=f"clusters_k{k}.csv", index=False)

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)
