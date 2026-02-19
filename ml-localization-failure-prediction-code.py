#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ANEXO A — Código para Replicação do Experimento
================================================================================

Título: Comparação de Métodos de Aprendizado de Máquina para Predição de
        Falhas de Localização em Robôs Móveis Autônomos

Autor:  Clístenes Grizafis Bento
Curso:  Especialização em Inteligência Artificial Aplicada - UFPR
Ano:    2024

Descrição:
    Este script implementa o pipeline completo do experimento descrito no
    artigo, incluindo:
    1. Carregamento dos dados do dataset (formato Apache Parquet)
    2. Extração de 22 features estatísticas (LiDAR + partículas AMCL)
    3. Pré-processamento (normalização, balanceamento, divisão treino/teste)
    4. Treinamento de 4 modelos de ML (Random Forest, SVM, MLP, XGBoost)
    5. Avaliação com múltiplas métricas e validação cruzada
    6. Geração de figuras e tabelas para o artigo

Dataset:
    "Synthetic Datasets for Predictive Localization Monitoring"
    Knitt, M.; Maroofi, S.; Thakkar, M.; Rose, H.; Braun, P. (2025)
    DOI: 10.2195/lj_proc_knitt_en_202503_01

Requisitos:
    pip install pandas numpy scikit-learn xgboost pyarrow matplotlib seaborn

Uso:
    python ANEXO_A_codigo_replicacao.py --data_path /caminho/para/parquets/

    Ou no Google Colab:
    1. Faça upload dos parquets para o Google Drive
    2. Monte o Drive: from google.colab import drive; drive.mount('/content/drive')
    3. Execute: !python ANEXO_A_codigo_replicacao.py --data_path /content/drive/MyDrive/tcciaa/preprocessed_data/parquets/processed/
================================================================================
"""

# =============================================================================
# IMPORTAÇÕES
# =============================================================================

import argparse
import glob
import gc
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.utils import resample

import xgboost as xgb

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTES E CONFIGURAÇÕES
# =============================================================================

# Semente aleatória para reprodutibilidade.
# Todos os componentes estocásticos (divisão treino/teste, modelos, etc.)
# usam esta semente para garantir resultados idênticos entre execuções.
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configurações de visualização
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
sns.set_style("whitegrid")
sns.set_palette("husl")

# Mapeamento dos experimentos para seus ambientes e configurações de obstáculos.
# Baseado no info.csv do dataset original (Knitt et al., 2025).
# Cada experimento (rosbag) foi realizado em um mapa específico com uma
# configuração de obstáculos (dinâmicos, estáticos ou ambos).
EXPERIMENT_INFO = {
    "rec_20250821_104113": {"map": "warehouse",        "dynamic": True,  "static": False},
    "rec_20250821_132354": {"map": "warehouse",        "dynamic": False, "static": True},
    "rec_20250821_141846": {"map": "warehouse",        "dynamic": True,  "static": True},
    "rec_20250821_151429": {"map": "symmetric_exp_1",   "dynamic": True,  "static": False},
    "rec_20250822_123916": {"map": "symmetric_exp_1",   "dynamic": False, "static": True},
    "rec_20250822_132819": {"map": "symmetric_exp_1",   "dynamic": True,  "static": True},
    "rec_20250822_135308": {"map": "symmetric_exp_2",   "dynamic": True,  "static": False},
    "rec_20250825_100748": {"map": "symmetric_exp_2",   "dynamic": False, "static": True},
    "rec_20250825_133605": {"map": "symmetric_exp_2",   "dynamic": True,  "static": True},
    "rec_20250825_135829": {"map": "symmetric_exp_3",   "dynamic": True,  "static": False},
    "rec_20250825_145941": {"map": "symmetric_exp_3",   "dynamic": False, "static": True},
    "rec_20250825_153216": {"map": "symmetric_exp_3",   "dynamic": True,  "static": True},
    "rec_20250825_161940": {"map": "unsymmetric_exp_1", "dynamic": True,  "static": False},
    "rec_20250826_095810": {"map": "unsymmetric_exp_1", "dynamic": False, "static": True},
    "rec_20250826_102611": {"map": "unsymmetric_exp_1", "dynamic": True,  "static": True},
    "rec_20250826_104543": {"map": "unsymmetric_exp_2", "dynamic": True,  "static": False},
    "rec_20250827_161251": {"map": "unsymmetric_exp_2", "dynamic": False, "static": True},
    "rec_20250828_102407": {"map": "unsymmetric_exp_2", "dynamic": True,  "static": True},
    "rec_20250828_104543": {"map": "unsymmetric_exp_3", "dynamic": True,  "static": False},
    "rec_20250828_161251": {"map": "unsymmetric_exp_3", "dynamic": False, "static": True},
    "rec_20250828_163407": {"map": "unsymmetric_exp_3", "dynamic": True,  "static": True},
}


# =============================================================================
# FUNÇÕES DE EXTRAÇÃO DE FEATURES
# =============================================================================


def _compute_entropy(values: np.ndarray, bins: int = 50) -> float:
    """
    Calcula a entropia de Shannon de uma distribuição contínua discretizada.

    A entropia mede a "diversidade" ou "incerteza" da distribuição.
    Valores altos indicam distribuição uniforme (muita diversidade);
    valores baixos indicam distribuição concentrada (pouca diversidade).

    Parâmetros:
        values: Array de valores numéricos.
        bins: Número de bins para discretização do histograma.

    Retorna:
        Entropia de Shannon em bits (base 2).
    """
    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist[hist > 0]  # Remover bins vazios para evitar log(0)
    hist = hist / hist.sum()  # Normalizar para probabilidades
    return -np.sum(hist * np.log2(hist))


def _compute_skewness(values: np.ndarray) -> float:
    """
    Calcula a assimetria (skewness) de uma distribuição.

    Skewness positivo: cauda longa à direita (mais leituras curtas).
    Skewness negativo: cauda longa à esquerda (mais leituras longas).
    Skewness zero: distribuição simétrica.

    Parâmetros:
        values: Array de valores numéricos.

    Retorna:
        Coeficiente de assimetria (Fisher).
    """
    n = len(values)
    if n < 3:
        return 0.0
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0.0
    return (n / ((n - 1) * (n - 2))) * np.sum(((values - mean) / std) ** 3)


def _compute_weight_entropy(weights: np.ndarray) -> float:
    """
    Calcula a entropia de Shannon dos pesos das partículas.

    Alta entropia: pesos uniformes -- filtro incerto sobre a localizacao.
    Baixa entropia: poucos pesos dominantes -- filtro confiante.

    Parâmetros:
        weights: Array com os pesos das partículas AMCL.

    Retorna:
        Entropia de Shannon em bits (base 2).
    """
    w = weights.copy()
    w = w[w > 0]
    if len(w) == 0:
        return 0.0
    w = w / w.sum()
    return -np.sum(w * np.log2(w))


def extract_lidar_features(scan_ranges: np.ndarray, range_max: float = 200.0) -> dict:
    """
    Extrai 12 features estatísticas das leituras do LiDAR.

    O LiDAR emite feixes laser em 360° e mede a distância até o obstáculo
    mais próximo em cada direção. Leituras próximas ao range_max indicam
    que não houve retorno (espaço aberto ou fora de alcance).

    Parâmetros:
        scan_ranges: Array com 3600 leituras de distância (metros).
        range_max: Valor máximo do range do sensor.

    Retorna:
        Dicionário com 12 features:
        - lidar_mean: Distância média aos obstáculos
        - lidar_std: Variabilidade das distâncias
        - lidar_min: Obstáculo mais próximo
        - lidar_max: Obstáculo mais distante (válido)
        - lidar_median: Mediana das distâncias
        - lidar_p10, lidar_p25, lidar_p75, lidar_p90: Percentis
        - lidar_max_ratio: Proporção de leituras "sem retorno"
        - lidar_entropy: Diversidade das distâncias
        - lidar_skewness: Assimetria da distribuição
    """
    # Filtrar leituras válidas (dentro do range operacional do sensor)
    valid = scan_ranges[scan_ranges < range_max * 0.99]
    if len(valid) == 0:
        valid = scan_ranges

    return {
        "lidar_mean": np.mean(valid),
        "lidar_std": np.std(valid),
        "lidar_min": np.min(valid),
        "lidar_max": np.max(valid),
        "lidar_median": np.median(valid),
        "lidar_p10": np.percentile(valid, 10),
        "lidar_p25": np.percentile(valid, 25),
        "lidar_p75": np.percentile(valid, 75),
        "lidar_p90": np.percentile(valid, 90),
        "lidar_max_ratio": np.sum(scan_ranges >= range_max * 0.99) / len(scan_ranges),
        "lidar_entropy": _compute_entropy(valid, bins=50),
        "lidar_skewness": _compute_skewness(valid),
    }


def extract_particle_features(particles: list) -> dict:
    """
    Extrai 8 features da nuvem de partículas do filtro AMCL.

    O AMCL mantém ~501 partículas, cada uma representando uma hipótese
    de localização do robô. Quando as partículas estão espalhadas, o
    filtro está incerto; quando concentradas, a localização é confiável.

    Parâmetros:
        particles: Lista de dicts com 'pose' (position + orientation) e 'weight'.

    Retorna:
        Dicionário com 8 features:
        - particle_std_x, particle_std_y: Dispersão espacial
        - particle_std_yaw: Dispersão angular (desvio padrão circular)
        - particle_spread: Diâmetro da nuvem de partículas
        - particle_weight_entropy: Entropia dos pesos
        - particle_weight_max: Peso máximo
        - particle_weight_mean: Peso médio
        - particle_weight_cv: Coeficiente de variação dos pesos
    """
    # Extrair posições e pesos de todas as partículas
    xs = np.array([p["pose"]["position"]["x"] for p in particles])
    ys = np.array([p["pose"]["position"]["y"] for p in particles])
    weights = np.array([p["weight"] for p in particles])

    # Converter quaternion (z, w) para ângulo yaw
    zs = np.array([p["pose"]["orientation"]["z"] for p in particles])
    ws = np.array([p["pose"]["orientation"]["w"] for p in particles])
    yaws = 2.0 * np.arctan2(zs, ws)

    return {
        "particle_std_x": np.std(xs),
        "particle_std_y": np.std(ys),
        # Desvio padrão circular: lida com a natureza cíclica dos ângulos
        "particle_std_yaw": np.sqrt(
            -2
            * np.log(
                np.clip(
                    np.sqrt(
                        np.mean(np.cos(yaws)) ** 2 + np.mean(np.sin(yaws)) ** 2
                    ),
                    1e-10,
                    1.0,
                )
            )
        ),
        "particle_spread": np.sqrt(
            (np.max(xs) - np.min(xs)) ** 2 + (np.max(ys) - np.min(ys)) ** 2
        ),
        "particle_weight_entropy": _compute_weight_entropy(weights),
        "particle_weight_max": np.max(weights),
        "particle_weight_mean": np.mean(weights),
        "particle_weight_cv": np.std(weights) / (np.mean(weights) + 1e-10),
    }


# =============================================================================
# ETAPA 1: CARREGAMENTO E EXTRAÇÃO DE FEATURES
# =============================================================================


def load_and_extract_features(data_path: str) -> pd.DataFrame:
    """
    Carrega todos os arquivos .parquet e extrai features de cada amostra.

    Estratégia de memória: processa um arquivo por vez, extrai features,
    e descarta os dados brutos antes de carregar o próximo arquivo.
    Isso permite processar o dataset completo (~417k amostras) sem
    estourar a RAM do Google Colab (~12.7 GB).

    Parâmetros:
        data_path: Caminho para a pasta contendo os arquivos .parquet.

    Retorna:
        DataFrame com 22 features + metadados + label para cada amostra.
    """
    parquet_files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
    print(f"[INFO] Encontrados {len(parquet_files)} arquivos .parquet")

    if len(parquet_files) == 0:
        raise FileNotFoundError(
            f"Nenhum arquivo .parquet encontrado em: {data_path}\n"
            "Verifique se o caminho está correto e se os dados foram extraídos."
        )

    all_features = []
    total_samples = 0

    for file_idx, filepath in enumerate(parquet_files):
        filename = os.path.basename(filepath)
        exp_name = "_".join(filename.split("_")[:3])

        print(f"  [{file_idx+1}/{len(parquet_files)}] {filename}", end="")

        df = pd.read_parquet(filepath)
        file_features = []

        for row_idx in range(len(df)):
            measurements = df.iloc[row_idx]["measurements"]

            for meas in measurements:
                try:
                    val = meas["value"]

                    # Extrair features do LiDAR (12 features)
                    lidar_feats = extract_lidar_features(
                        val["/scan/ranges"],
                        range_max=val.get("/scan/range_max", 200.0),
                    )

                    # Extrair features das partículas AMCL (8 features)
                    particle_feats = extract_particle_features(
                        val["/particle_cloud/particles"]
                    )

                    # Features de pose estimada pelo AMCL (2 features)
                    pose_feats = {
                        "amcl_x": val.get("/amcl_pose/pose.pose.position.x", 0.0),
                        "amcl_y": val.get("/amcl_pose/pose.pose.position.y", 0.0),
                    }

                    # Combinar todas as features
                    sample = {**lidar_feats, **particle_feats, **pose_feats}

                    # Metadados (não usados como features nos modelos)
                    sample["position_error"] = val.get("position_error", 0.0)
                    sample["heading_error"] = val.get("heading_error", 0.0)
                    sample["is_delocalized"] = val.get("is_delocalized", False)
                    sample["experiment"] = exp_name

                    exp_info = EXPERIMENT_INFO.get(exp_name, {})
                    sample["map_type"] = exp_info.get("map", "unknown")
                    sample["has_dynamic"] = exp_info.get("dynamic", False)
                    sample["has_static"] = exp_info.get("static", False)

                    file_features.append(sample)
                except Exception:
                    continue  # Pular amostras com dados corrompidos

        n_samples = len(file_features)
        total_samples += n_samples
        print(f" -> {n_samples} amostras (total: {total_samples})")

        if file_features:
            all_features.append(pd.DataFrame(file_features))

        # Liberar memória
        del df, file_features
        gc.collect()

    print(f"\n[OK] Extracao concluida: {total_samples} amostras")

    dataset = pd.concat(all_features, ignore_index=True)
    del all_features
    gc.collect()

    # Converter label para inteiro (0 = nominal, 1 = falha)
    dataset["label"] = dataset["is_delocalized"].astype(int)

    return dataset


# =============================================================================
# ETAPA 2: PRÉ-PROCESSAMENTO
# =============================================================================


def preprocess(dataset: pd.DataFrame, feature_cols: list):
    """
    Realiza o pré-processamento completo dos dados:
    1. Divide em treino (80%) e teste (20%) com estratificação
    2. Normaliza as features com StandardScaler
    3. Balanceia as classes por undersampling da classe majoritária

    Parâmetros:
        dataset: DataFrame com features e label.
        feature_cols: Lista de nomes das colunas de features.

    Retorna:
        Tupla com:
        - X_train_balanced: Features de treino balanceadas e normalizadas
        - y_train_balanced: Labels de treino balanceados
        - X_test_scaled: Features de teste normalizadas
        - y_test: Labels de teste (distribuição original)
        - meta_test: Metadados do conjunto de teste
        - scaler: Objeto StandardScaler ajustado
    """
    X = dataset[feature_cols].copy()
    y = dataset["label"].copy()
    metadata = dataset[
        ["experiment", "map_type", "has_dynamic", "has_static"]
    ].copy()

    # --- Divisão treino/teste estratificada ---
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, metadata,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  Treino: {len(X_train)} | Teste: {len(X_test)}")

    # --- Normalização (StandardScaler) ---
    # Ajustamos APENAS nos dados de treino para evitar data leakage
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    # --- Balanceamento por undersampling ---
    # Reduz a classe majoritária (nominal) para igualar a minoritária (falha)
    train_nominal = X_train_scaled[y_train == 0]
    train_failure = X_train_scaled[y_train == 1]
    y_nominal = y_train[y_train == 0]
    y_failure = y_train[y_train == 1]

    train_nominal_under, y_nominal_under = resample(
        train_nominal, y_nominal,
        replace=False,
        n_samples=len(train_failure),
        random_state=RANDOM_STATE,
    )

    X_train_balanced = pd.concat([train_nominal_under, train_failure])
    y_train_balanced = pd.concat([y_nominal_under, y_failure])

    # Embaralhar
    shuffle_idx = np.random.permutation(len(X_train_balanced))
    X_train_balanced = X_train_balanced.iloc[shuffle_idx]
    y_train_balanced = y_train_balanced.iloc[shuffle_idx]

    print(f"  Treino balanceado: {len(X_train_balanced)} (50/50)")

    return (
        X_train_balanced, y_train_balanced,
        X_test_scaled, y_test,
        X_train_scaled, y_train,
        meta_test, scaler,
    )


# =============================================================================
# ETAPA 3: DEFINIÇÃO E TREINAMENTO DOS MODELOS
# =============================================================================


def define_models() -> dict:
    """
    Define os 4 modelos de ML com hiperparâmetros pré-configurados.

    Retorna:
        Dicionário {nome: modelo_sklearn}.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate="adaptive",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            use_label_encoder=False,
        ),
    }


def train_and_evaluate(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Treina cada modelo e calcula métricas no conjunto de teste.

    Para o SVM, usa amostra de 50.000 se o dataset for muito grande,
    pois a complexidade do SVM é O(n²) e datasets grandes inviabilizam
    o treinamento em tempo razoável.

    Parâmetros:
        models: Dicionário {nome: modelo_sklearn}.
        X_train: Features de treino (balanceadas e normalizadas).
        y_train: Labels de treino.
        X_test: Features de teste (normalizadas).
        y_test: Labels de teste.

    Retorna:
        Dicionário com resultados de cada modelo.
    """
    results = {}

    for name, model in models.items():
        print(f"\n  [TREINO] {name}...", end=" ")

        # Amostragem para SVM (limitação computacional)
        if name == "SVM" and len(X_train) > 50000:
            sample_idx = np.random.choice(len(X_train), 50000, replace=False)
            X_fit = X_train.iloc[sample_idx]
            y_fit = y_train.iloc[sample_idx]
            print(f"(amostra de 50k)", end=" ")
        else:
            X_fit = X_train
            y_fit = y_train

        # Treinar
        t0 = time.time()
        model.fit(X_fit, y_fit)
        train_time = time.time() - t0

        # Predizer
        t0 = time.time()
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        pred_time = time.time() - t0

        # Métricas
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "train_time": train_time,
            "pred_time": pred_time,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_prob),
        }

        r = results[name]
        print(
            f"Acc={r['accuracy']:.4f} F1={r['f1']:.4f} "
            f"AUC={r['auc_roc']:.4f} ({train_time:.1f}s)"
        )

    return results


# =============================================================================
# ETAPA 4: VALIDAÇÃO CRUZADA
# =============================================================================


def cross_validate(X_train_scaled, y_train, max_samples=30000):
    """
    Realiza validação cruzada estratificada com 5 folds.

    Parâmetros:
        X_train_scaled: Features de treino normalizadas (antes do balanceamento).
        y_train: Labels de treino originais.
        max_samples: Máximo de amostras para viabilizar o SVM.

    Retorna:
        Dicionário com resultados da CV para cada modelo.
    """
    if len(X_train_scaled) > max_samples:
        cv_idx = np.random.choice(len(X_train_scaled), max_samples, replace=False)
        X_cv = X_train_scaled.iloc[cv_idx]
        y_cv = y_train.iloc[cv_idx]
        print(f"  (usando amostra de {max_samples} para CV)")
    else:
        X_cv = X_train_scaled
        y_cv = y_train

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_models = [
        ("Random Forest", RandomForestClassifier(
            n_estimators=200, max_depth=20, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1)),
        ("SVM", SVC(
            kernel="rbf", C=10.0, class_weight="balanced",
            random_state=RANDOM_STATE)),
        ("MLP", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=300,
            early_stopping=True, random_state=RANDOM_STATE)),
        ("XGBoost", xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1,
            use_label_encoder=False, eval_metric="logloss")),
    ]

    cv_results = {}
    for name, model in cv_models:
        print(f"  CV {name}...", end=" ")
        scores = cross_val_score(model, X_cv, y_cv, cv=cv, scoring="f1", n_jobs=-1)
        cv_results[name] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}
        print(f"F1 = {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_results


# =============================================================================
# ETAPA 5: GERAÇÃO DE FIGURAS
# =============================================================================


def generate_figures(dataset, results, y_test, feature_cols, meta_test, output_dir):
    """
    Gera todas as figuras do artigo e salva em PNG.

    Figuras geradas:
    1. Distribuição de classes (geral, por mapa, por obstáculo)
    2. Distribuição das features principais por classe
    3. Matriz de correlação
    4. Comparação de métricas entre modelos
    5. Matrizes de confusão
    6. Curvas ROC e Precision-Recall
    7. Feature importance (Random Forest e XGBoost)
    8. Performance por tipo de ambiente
    """
    os.makedirs(output_dir, exist_ok=True)
    colors_models = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    # --- Figura 1: Distribuição de classes ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    counts = dataset["label"].value_counts()
    colors_class = ["#2ecc71", "#e74c3c"]
    bars = axes[0].bar(
        ["Nominal (0)", "Falha (1)"], counts.values,
        color=colors_class, edgecolor="black",
    )
    axes[0].set_title("Distribuição Geral das Classes")
    axes[0].set_ylabel("Número de Amostras")
    for bar, count in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 100,
            f"{count:,}\n({count/len(dataset)*100:.1f}%)",
            ha="center", va="bottom", fontweight="bold",
        )

    failure_by_map = dataset.groupby("map_type")["label"].mean() * 100
    failure_by_map = failure_by_map.sort_values(ascending=False)
    axes[1].barh(failure_by_map.index, failure_by_map.values, color="#3498db", edgecolor="black")
    axes[1].set_title("Taxa de Falha por Tipo de Mapa")
    axes[1].set_xlabel("Taxa de Falha (%)")

    def get_obstacle_config(row):
        if row["has_dynamic"] and row["has_static"]:
            return "Dinâmico + Estático"
        elif row["has_dynamic"]:
            return "Apenas Dinâmico"
        elif row["has_static"]:
            return "Apenas Estático"
        return "Sem obstáculos"

    dataset["obstacle_config"] = dataset.apply(get_obstacle_config, axis=1)
    failure_by_obs = dataset.groupby("obstacle_config")["label"].mean() * 100
    failure_by_obs = failure_by_obs.sort_values(ascending=False)
    axes[2].barh(failure_by_obs.index, failure_by_obs.values, color="#e67e22", edgecolor="black")
    axes[2].set_title("Taxa de Falha por Config. de Obstáculos")
    axes[2].set_xlabel("Taxa de Falha (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_distribuicao_classes.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figura 2: Distribuição das features ---
    key_features = [
        "particle_std_x", "particle_std_y", "particle_spread",
        "particle_weight_entropy", "lidar_mean", "lidar_std",
        "lidar_max_ratio", "lidar_entropy",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for idx, feat in enumerate(key_features):
        ax = axes[idx]
        for label, color, name in [(0, "#2ecc71", "Nominal"), (1, "#e74c3c", "Falha")]:
            data = dataset[dataset["label"] == label][feat]
            ax.hist(data, bins=50, alpha=0.6, color=color, label=name, density=True)
        ax.set_title(feat, fontsize=11)
        ax.legend(fontsize=9)
    plt.suptitle("Distribuição das Features Principais por Classe", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_distribuicao_features.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figura 3: Matriz de correlação ---
    corr_cols = feature_cols + ["label"]
    corr_matrix = dataset[corr_cols].corr()
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, annot_kws={"size": 8},
    )
    plt.title("Matriz de Correlação das Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_correlacao.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figura 4: Comparação de métricas ---
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    metric_labels = ["Acurácia", "Precisão", "Recall", "F1-Score", "AUC-ROC"]
    model_names = list(results.keys())

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.18
    for i, (name, color) in enumerate(zip(model_names, colors_models)):
        values = [results[name][m] for m in metrics_to_plot]
        bars = ax.bar(x + i * width, values, width, label=name, color=color, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_labels)
    ax.set_title("Comparação de Métricas entre Modelos", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_comparacao_metricas.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figura 5: Matrizes de confusão ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for idx, (name, r) in enumerate(results.items()):
        cm = confusion_matrix(y_test, r["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
            xticklabels=["Nominal", "Falha"], yticklabels=["Nominal", "Falha"], cbar=False,
        )
        axes[idx].set_title(name, fontsize=12, fontweight="bold")
        axes[idx].set_ylabel("Real" if idx == 0 else "")
        axes[idx].set_xlabel("Predito")
    plt.suptitle("Matrizes de Confusão", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_matrizes_confusao.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figura 6: Curvas ROC e Precision-Recall ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for (name, r), color in zip(results.items(), colors_models):
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        axes[0].plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={r['auc_roc']:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Aleatório (AUC=0.5)")
    axes[0].set_xlabel("Taxa de Falsos Positivos")
    axes[0].set_ylabel("Taxa de Verdadeiros Positivos")
    axes[0].set_title("Curvas ROC", fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    for (name, r), color in zip(results.items(), colors_models):
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, r["y_prob"])
        ap = average_precision_score(y_test, r["y_prob"])
        axes[1].plot(rec_curve, prec_curve, color=color, linewidth=2, label=f"{name} (AP={ap:.4f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precisão")
    axes[1].set_title("Curvas Precision-Recall", fontweight="bold")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_curvas_roc_pr.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figura 7: Feature importance ---
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for idx, name in enumerate(["Random Forest", "XGBoost"]):
        model = results[name]["model"]
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = min(15, len(feature_cols))
        top_indices = indices[:top_n]
        top_features = [feature_cols[i] for i in top_indices]
        top_importances = importances[top_indices]
        colors_feat = [
            "#e74c3c" if "particle" in f else "#3498db" if "lidar" in f else "#2ecc71"
            for f in top_features
        ]
        axes[idx].barh(range(top_n), top_importances[::-1], color=colors_feat[::-1], edgecolor="black", linewidth=0.5)
        axes[idx].set_yticks(range(top_n))
        axes[idx].set_yticklabels(top_features[::-1])
        axes[idx].set_xlabel("Importância")
        axes[idx].set_title(f"Feature Importance - {name}", fontweight="bold")
        legend_elements = [
            Patch(facecolor="#e74c3c", edgecolor="black", label="Partículas AMCL"),
            Patch(facecolor="#3498db", edgecolor="black", label="LiDAR"),
            Patch(facecolor="#2ecc71", edgecolor="black", label="Pose AMCL"),
        ]
        axes[idx].legend(handles=legend_elements, loc="lower right")
    plt.suptitle("Importância das Features", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig7_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  [INFO] Figuras salvas em: {output_dir}")


# =============================================================================
# ETAPA 6: EXPORTAÇÃO DE RESULTADOS
# =============================================================================


def export_results(dataset, results, cv_results, feature_cols, y_test, output_dir):
    """
    Exporta todos os resultados em formato JSON para referência.
    """
    export = {
        "dataset": {
            "total_amostras": int(len(dataset)),
            "n_features": len(feature_cols),
            "classe_nominal": int((dataset["label"] == 0).sum()),
            "classe_falha": int((dataset["label"] == 1).sum()),
            "taxa_falha_pct": round(float(dataset["label"].mean() * 100), 2),
        },
        "modelos": {},
        "cross_validation": {},
    }

    for name, r in results.items():
        export["modelos"][name] = {
            "accuracy": round(float(r["accuracy"]), 4),
            "precision": round(float(r["precision"]), 4),
            "recall": round(float(r["recall"]), 4),
            "f1": round(float(r["f1"]), 4),
            "auc_roc": round(float(r["auc_roc"]), 4),
            "train_time_s": round(float(r["train_time"]), 2),
            "confusion_matrix": confusion_matrix(y_test, r["y_pred"]).tolist(),
        }

    for name, r in cv_results.items():
        export["cross_validation"][name] = {
            "f1_mean": round(float(r["mean"]), 4),
            "f1_std": round(float(r["std"]), 4),
        }

    output_path = os.path.join(output_dir, "resultados_tcc.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"  [INFO] Resultados exportados: {output_path}")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================


def main():
    """
    Pipeline principal do experimento.

    Uso:
        python ANEXO_A_codigo_replicacao.py --data_path /caminho/para/parquets/
        python ANEXO_A_codigo_replicacao.py --csv /caminho/para/dataset_features.csv
    """
    parser = argparse.ArgumentParser(
        description="Replicação: Comparação de ML para Predição de Falhas de Localização"
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Caminho para a pasta com arquivos .parquet do dataset",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Caminho para CSV com features já extraídas (pula a etapa de extração)",
    )
    parser.add_argument(
        "--output", type=str, default="./resultados",
        help="Pasta de saída para figuras e resultados (padrão: ./resultados)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  COMPARAÇÃO DE ML PARA PREDIÇÃO DE FALHAS DE LOCALIZAÇÃO")
    print("  EM ROBÔS MÓVEIS AUTÔNOMOS")
    print("=" * 70)

    # --- Etapa 1: Carregar dados ---
    print("\nETAPA 1: Carregamento dos dados")
    if args.csv and os.path.exists(args.csv):
        print(f"  Carregando CSV: {args.csv}")
        dataset = pd.read_csv(args.csv)
        print(f"  [OK] {len(dataset)} amostras carregadas")
    elif args.data_path:
        dataset = load_and_extract_features(args.data_path)
        # Salvar checkpoint
        csv_path = os.path.join(args.output, "dataset_features.csv")
        os.makedirs(args.output, exist_ok=True)
        dataset.to_csv(csv_path, index=False)
        print(f"  [INFO] Checkpoint salvo: {csv_path}")
    else:
        print("ERRO: Forneça --data_path ou --csv")
        print("Exemplo: python ANEXO_A_codigo_replicacao.py --data_path ./parquets/")
        return

    # Identificar colunas de features (excluir metadados e label)
    feature_cols = [
        c for c in dataset.columns
        if c not in [
            "is_delocalized", "label", "experiment", "map_type",
            "has_dynamic", "has_static", "position_error", "heading_error",
            "obstacle_config",
        ]
    ]
    print(f"  Features: {len(feature_cols)}")
    print(f"  Classes: Nominal={int((dataset['label']==0).sum())}, Falha={int((dataset['label']==1).sum())}")

    # --- Etapa 2: Pré-processamento ---
    print("\nETAPA 2: Pre-processamento")
    (
        X_train_balanced, y_train_balanced,
        X_test_scaled, y_test,
        X_train_scaled, y_train,
        meta_test, scaler,
    ) = preprocess(dataset, feature_cols)

    # --- Etapa 3: Treinamento ---
    print("\nETAPA 3: Treinamento dos modelos")
    models = define_models()
    results = train_and_evaluate(
        models, X_train_balanced, y_train_balanced, X_test_scaled, y_test
    )

    # --- Etapa 4: Validação cruzada ---
    print("\nETAPA 4: Validacao cruzada (5-fold)")
    cv_results = cross_validate(X_train_scaled, y_train)

    # --- Etapa 5: Figuras ---
    print("\nETAPA 5: Geracao de figuras")
    generate_figures(dataset, results, y_test, feature_cols, meta_test, args.output)

    # --- Etapa 6: Exportação ---
    print("\nETAPA 6: Exportacao de resultados")
    export_results(dataset, results, cv_results, feature_cols, y_test, args.output)

    # --- Resumo final ---
    print("\n" + "=" * 70)
    print("  RESUMO FINAL")
    print("=" * 70)
    print(f"  Dataset: {len(dataset):,} amostras, {len(feature_cols)} features")
    print(f"\n  {'Modelo':<20s} {'F1':>8s} {'AUC-ROC':>8s} {'Treino':>8s}")
    print("  " + "-" * 48)
    for name, r in results.items():
        print(f"  {name:<20s} {r['f1']:>8.4f} {r['auc_roc']:>8.4f} {r['train_time']:>7.1f}s")
    print(f"\n  CV (5-fold F1):")
    for name, r in cv_results.items():
        print(f"  {name:<20s} {r['mean']:.4f} ± {r['std']:.4f}")
    print("=" * 70)
    print("  [OK] Experimento concluido com sucesso!")
    print("=" * 70)


if __name__ == "__main__":
    main()
