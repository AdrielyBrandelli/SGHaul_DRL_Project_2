import pandas as pd
import numpy as np
from scipy.stats import linregress
import os
import json

def analisar_dados_lactec(caminho_csv: str):
    if not os.path.exists(caminho_csv):
        print(f"Erro: O arquivo {caminho_csv} não existe. Execute o log_parser.py primeiro.")
        return

    df = pd.read_csv(caminho_csv)
    if df.empty:
        print("Erro: O CSV está vazio.")
        return

    df['tipo_trafego'] = df['tipo_trafego'].replace(['UC1', 'UC17'], 'mMTC')

    print("--- EXTRAÇÃO DE PARÂMETROS FÍSICOS DO LACTEC POR TOPOLOGIA ---\n")
    
    calibracao_geral = {}
    topologias = df['topologia'].unique()

    for topo in topologias:
        print(f"================ TOPOLOGIA: {topo.upper()} ================")
        df_topo = df[df['topologia'] == topo]
        
        dados_topo = {}
        betas = []
        jitters = []

        # --- VALIDAÇÃO DE TAMANHOS DE PACOTE ---
        tm_uc16 = df_topo[df_topo['tipo_trafego'] == 'UC16']['tamanho_bytes'].mean()
        tm_uc3 = df_topo[df_topo['tipo_trafego'] == 'UC3']['tamanho_bytes'].mean()
        tm_mmtc = df_topo[df_topo['tipo_trafego'] == 'mMTC']['tamanho_bytes'].mean()
        
        if pd.isna(tm_uc16):
            print("  -> [ALERTA DADOS] Nenhum tamanho válido para UC16. Usando fallback (64B).")
            dados_topo['tamanho_uc16'] = 64
        else:
            dados_topo['tamanho_uc16'] = int(tm_uc16)

        if pd.isna(tm_uc3):
            print("  -> [ALERTA DADOS] Nenhum tamanho válido para UC3. Usando fallback (1024B).")
            dados_topo['tamanho_uc3'] = 1024
        else:
            dados_topo['tamanho_uc3'] = int(tm_uc3)

        if pd.isna(tm_mmtc):
            print("  -> [ALERTA DADOS] Nenhum tamanho válido para mMTC. Usando fallback (256B).")
            dados_topo['tamanho_mmtc'] = 256
        else:
            dados_topo['tamanho_mmtc'] = int(tm_mmtc)

        # --- MODELAGEM DO ATRASO DE PROPAGAÇÃO ---
        for trafego in ['UC16', 'UC3', 'mMTC']:
            df_filtrado = df_topo[(df_topo['tipo_trafego'] == trafego) & (df_topo['entregue'] == 1)]
            if len(df_filtrado) < 2:
                print(f"  -> [ALERTA MATH] [{trafego}] Sem dados suficientes para regressão. Acionando fallback (20ms).")
                dados_topo[f'alpha_{trafego.lower()}'] = 20.0
                continue

            hops = df_filtrado['hop_count'].values
            latencias = df_filtrado['latencia_ms'].values

            try:
                slope, intercept, _, _, _ = linregress(hops, latencias)
                if np.isnan(slope) or np.isnan(intercept):
                    raise ValueError("Regressão retornou NaN")
                    
                latencias_previstas = (slope * hops) + intercept
                residuos = latencias - latencias_previstas
                jitter_std = float(np.std(residuos))

                betas.append(intercept)
                jitters.append(jitter_std)

                print(f"[{trafego}] Alpha (Tempo/Salto): {slope:.2f} ms | Beta (Processamento): {intercept:.2f} ms | Jitter: {jitter_std:.2f} ms")
                dados_topo[f'alpha_{trafego.lower()}'] = float(slope)
                
            except Exception as e:
                print(f"  -> [ERRO MATH] [{trafego}] Falha na regressão ({e}). Usando fallback (20ms).")
                dados_topo[f'alpha_{trafego.lower()}'] = 20.0

        dados_topo['beta'] = float(np.mean(betas)) if betas else 10.0
        dados_topo['std_dev_jitter'] = float(np.mean(jitters)) if jitters else 5.0

        # --- MODELAGEM DA CONFIABILIDADE (PDR) ---
        pdr_por_hop = df_topo.groupby('hop_count')['entregue'].mean()
        
        if len(pdr_por_hop) > 1:
            try:
                slope_pdr, intercept_pdr, _, _, _ = linregress(pdr_por_hop.index, pdr_por_hop.values)
                if np.isnan(slope_pdr):
                    raise ValueError("PDR Slope é NaN")
                print(f"[PDR] Equação: prob_sucesso = ({slope_pdr:.4f} * hop_count) + {intercept_pdr:.4f}")
                dados_topo['pdr_slope'] = float(slope_pdr)
                dados_topo['pdr_intercept'] = float(intercept_pdr)
            except Exception as e:
                print(f"  -> [ALERTA PDR] Falha na atenuação ({e}). Usando fallback.")
                dados_topo['pdr_slope'] = -0.0005
                dados_topo['pdr_intercept'] = 1.0014
        else:
            print("  -> [ALERTA PDR] Apenas 1 hop mapeado. Impossível calcular atenuação. Usando fallback.")
            dados_topo['pdr_slope'] = -0.0005
            dados_topo['pdr_intercept'] = 1.0014

        # --- MODELAGEM DE TRÁFEGO ---
        t_min = df_topo['t0'].min()
        t_max = df_topo['t0'].max()
        duracao_segundos = (t_max - t_min) / 1000.0 if pd.notna(t_max) and pd.notna(t_min) else 1.0
        if duracao_segundos <= 0: duracao_segundos = 1.0

        df_uc16 = df_topo[df_topo['tipo_trafego'] == 'UC16']
        df_uc3 = df_topo[df_topo['tipo_trafego'] == 'UC3'].copy()
        df_mmtc = df_topo[df_topo['tipo_trafego'] == 'mMTC']

        dados_topo['lambda_uc16'] = float(len(df_uc16) / duracao_segundos) if len(df_uc16) > 0 else 2.0
        dados_topo['lambda_mmtc'] = float(len(df_mmtc) / duracao_segundos) if len(df_mmtc) > 0 else 50.0

        if not df_uc3.empty:
            df_uc3['t0_sec'] = df_uc3['t0'] // 1000
            bursts = df_uc3.groupby('t0_sec').size()
            dados_topo['burst_uc3'] = float(bursts.mean()) 
            dados_topo['periodo_uc3'] = float(duracao_segundos / len(bursts)) 
        else:
            print("  -> [ALERTA TRÁFEGO] Sem pacotes UC3. Usando fallback de burst.")
            dados_topo['burst_uc3'] = 50.0
            dados_topo['periodo_uc3'] = 5.0

        print(f"[TAMANHOS REAIS] UC16: {dados_topo['tamanho_uc16']}B | UC3: {dados_topo['tamanho_uc3']}B | mMTC: {dados_topo['tamanho_mmtc']}B\n")
        
        calibracao_geral[topo] = dados_topo

    # --- EXPORTAÇÃO JSON ---
    caminho_json = "data/calibracao.json"
    os.makedirs(os.path.dirname(caminho_json), exist_ok=True)
    with open(caminho_json, "w") as f:
        json.dump(calibracao_geral, f, indent=4)
        
    print(f"✅ Calibração Física Concluída! JSON salvo em '{caminho_json}'.")

if __name__ == "__main__":
    csv_caminho = "data/dados_lactec_raw.csv"
    analisar_dados_lactec(csv_caminho)