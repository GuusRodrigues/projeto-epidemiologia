# src/ml_pipeline.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import joblib

AGG_PATH = Path("data/processed/aggregated")
MODEL_PATH = Path("data/models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

def carregar_dados():
    arquivos = sorted(AGG_PATH.glob("casos_por_estado_mes_*.csv"), reverse=True)
    if not arquivos:
        raise FileNotFoundError("Nenhum arquivo agregado encontrado em data/processed/aggregated")
    df = pd.read_csv(arquivos[0])
    df['ano_mes'] = pd.to_datetime(df['ano_mes'], format='%Y-%m')
    df = df.sort_values(['sg_uf_not','ano_mes'])
    return df

def criar_features_lags(df, n_lags=3):
    df = df.copy()
    for lag in range(1, n_lags+1):
        df[f'lag_{lag}'] = df.groupby('sg_uf_not')['casos'].shift(lag)
    df['mes'] = df['ano_mes'].dt.month
    df['ano'] = df['ano_mes'].dt.year
    df = df.dropna()
    return df

def preparar_dados(df, n_lags=3):
    X = df[['sg_uf_not','ano','mes'] + [f'lag_{i}' for i in range(1,n_lags+1)]].copy()
    y = df['casos']
    estados = sorted(X['sg_uf_not'].unique())
    estado_map = {e:i for i,e in enumerate(estados)}
    X['sg_uf_not'] = X['sg_uf_not'].map(estado_map)
    return X, y, estado_map

def treinar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")
    print(f"R2: {r2_score(y_test, y_pred):.3f}")
    return model

def gerar_features_serie_por_estado(df, janela=6):
    """
    Para clustering: extrai features por estado com agregações de séries temporais.
    Retorna DataFrame com uma linha por estado e colunas de features (média, var, trend).
    """
    estados = []
    feats = []
    for estado, g in df.groupby('sg_uf_not'):
        s = g.sort_values('ano_mes')['casos'].rolling(window=janela, min_periods=1).mean().fillna(0)
        media = s.mean()
        var = s.var()
        # tendência simples (slope)
        idx = np.arange(len(s))
        if len(s) > 1:
            slope = np.polyfit(idx, s, 1)[0]
        else:
            slope = 0.0
        feats.append([media, var, slope])
        estados.append(estado)
    df_feats = pd.DataFrame(feats, columns=['mean_6m','var_6m','slope_6m'], index=estados).reset_index().rename(columns={'index':'sg_uf_not'})
    return df_feats


def treinar_clustering(df_feats_por_estado, n_clusters=3):
    X = df_feats_por_estado.select_dtypes(include=['float64', 'int64'])
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_imputed)
    
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(Xp)
    
    df_clusters = df_feats_por_estado.copy()
    df_clusters['Cluster'] = clusters
    df_clusters['PCA1'] = Xp[:, 0]
    df_clusters['PCA2'] = Xp[:, 1]
    
    return df_clusters, scaler, pca, kmeans

def salvar_modelo(model, estado_map):
    joblib.dump(model, MODEL_PATH / "mpox_casos_rf.joblib")
    joblib.dump(estado_map, MODEL_PATH / "estado_map.joblib")
    print("[INFO] Modelo e mapa de estados salvos em data/models/")

def main():
    df = carregar_dados()
    df_feat = criar_features_lags(df)
    X, y, estado_map = preparar_dados(df_feat)
    model = treinar_modelo(X, y)
    salvar_modelo(model, estado_map)

    # Clustering
    df_feats_por_estado = gerar_features_serie_por_estado(df, janela=6)
    df_clusters, scaler, pca, kmeans = treinar_clustering(df_feats_por_estado, n_clusters=3)
    df_clusters.to_csv(MODEL_PATH/'clusters_por_estado.csv', index=False)
    print("[INFO] Pipeline concluído.")

if __name__ == "__main__":
    main()
