# Comparação de Métodos de Aprendizado de Máquina para Predição de Falhas de Localização em Robôs Móveis Autônomos

## Descrição

Este repositório contém o código-fonte completo para replicação dos experimentos descritos no artigo:

> **Comparação de Métodos de Aprendizado de Máquina para Predição de Falhas de Localização em Robôs Móveis Autônomos**
>
> Clístenes Grizafis Bento
>
> Especialização em Inteligência Artificial Aplicada — UFPR, 2026

O trabalho compara quatro algoritmos de aprendizado de máquina (Random Forest, SVM, MLP e XGBoost) para a tarefa de predição binária de falhas de localização em robôs móveis autônomos, utilizando dados de LiDAR e do filtro de partículas AMCL (Adaptive Monte Carlo Localization).

## Dataset

Os experimentos utilizam o dataset **Robot Localization Failure Prediction Dataset** (KNITT et al., 2025), disponível publicamente em:

- **Repositório**: [TORE — Technische Universität Hamburg](https://tore.tuhh.de/entities/product/71944c77-d6f3-4ecf-9dce-805e51b3a57f)
- **DOI**: [10.15480/882.15836](https://doi.org/10.15480/882.15836)

O dataset contém 417.185 amostras rotuladas de 21 experimentos em 7 ambientes simulados no NVIDIA Isaac Sim, com dados de um robô omnidirecional MoMo equipado com LiDAR Velodyne VLP-16.

## Pipeline do Experimento

1. Carregamento dos dados (formato Apache Parquet)
2. Extração de 22 características estatísticas (LiDAR + partículas AMCL)
3. Pré-processamento (divisão treino/teste, normalização, balanceamento)
4. Treinamento de 4 modelos de ML (Random Forest, SVM, MLP, XGBoost)
5. Avaliação com múltiplas métricas e validação cruzada estratificada (5-fold)
6. Análise segmentada por tipo de ambiente e configuração de obstáculos

## Requisitos

```bash
pip install pandas numpy scikit-learn xgboost pyarrow matplotlib seaborn
```

## Uso

### Execução local

```bash
python ANEXO_A_codigo_replicacao.py --data_path /caminho/para/parquets/
```

### Google Colab

1. Faça upload dos arquivos Parquet para o Google Drive
2. Monte o Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Execute:
   ```bash
   !python ANEXO_A_codigo_replicacao.py --data_path /content/drive/MyDrive/tcciaa/preprocessed_data/parquets/processed/
   ```

## Resultados Principais

| Modelo | F1-Score | AUC-ROC | Tempo Treino (s) |
| --- | --- | --- | --- |
| **Random Forest** | **0,9604** | **0,9983** | 19,43 |
| SVM | 0,6599 | 0,8877 | 537,98 |
| MLP | 0,9031 | 0,9905 | 232,64 |
| XGBoost | 0,9516 | 0,9977 | **2,35** |

## Referências

- KNITT, M.; MAROOFI, S.; THAKKAR, M.; ROSE, H.; BRAUN, P. Synthetic Datasets for Data-Driven Localization Monitoring. **Logistics Journal: Proceedings**, 2025. DOI: [10.2195/lj_proc_knitt_en_202503_01](https://doi.org/10.2195/lj_proc_knitt_en_202503_01)
- KNITT, M.; THAKKAR, M. B.; MAROOFI, S.; ROSE, H. W.; BRAUN, P. M. **Robot Localization Failure Prediction Dataset**. Hamburg: Technische Universität Hamburg, 2025. DOI: [10.15480/882.15836](https://doi.org/10.15480/882.15836)
- EDER, M.; REIP, M.; STEINBAUER, G. Creating a robot localization monitor using particle filter and machine learning approaches. **Applied Intelligence**, v. 52, p. 6955-6969, 2022. DOI: [10.1007/s10489-020-02157-6](https://doi.org/10.1007/s10489-020-02157-6)

## Licença

Este projeto é disponibilizado para fins acadêmicos e de pesquisa.

## Autor

**Clístenes Grizafis Bento**
Especialização em Inteligência Artificial Aplicada — Universidade Federal do Paraná (UFPR)
