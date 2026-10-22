import re
import csv
import os
import glob
import pandas as pd

MAPA_UC = {
    'event notification': 'UC16',
    'event acknowledge': 'UC16',
    'operation data request': 'UC3',
    'operation data response': 'UC3',
    'association request': 'UC3',
    'association response': 'UC3',
    'disassociation request': 'UC3',
    'disassociation response': 'UC3',
    'register request': 'UC1',
    'availability request': 'UC17',
    'availability response': 'UC17'
}

def _extrair_dados_tx(line):
    msg_clean = None
    for key in MAPA_UC.keys():
        if key in line.lower():
            msg_clean = key
            break
            
    if not msg_clean:
        return None
        
    uc_type = MAPA_UC[msg_clean]
    
    match_t0 = re.search(r't0=(\d+)', line)
    if not match_t0:
        print(f"[ALERTA PARSER] Mensagem '{msg_clean}' sem t0 na linha: {line[:60]}...")
        return None
    t0 = int(match_t0.group(1))
    
    match_size = re.search(r'UDP SIZE\s*=\s*(\d+)', line)
    if match_size:
        tamanho_bytes = int(match_size.group(1))
    else:
        # Só imprime alerta para UC3 porque 64 bytes pro UC3 é irreal
        if uc_type == 'UC3':
            print(f"[ALERTA PARSER] UC3 sem tamanho definido no log. Usando fallback 64B. Linha: {line[:60]}...")
        tamanho_bytes = 64
    
    return uc_type, tamanho_bytes, t0

def parse_excel_linear(input_file: str) -> list:
    print(f"Lendo Excel (Linear): {input_file}...")
    packets = {}
    if not os.path.exists(input_file):
        print(f"  -> Arquivo {input_file} não encontrado. Pulando...")
        return []

    routes = {}
    rota_lactec = "0001-0001;0002-0001;0003-0002;0004-0003;0005-0004;0006-0005;0007-0006;0008-0007;0009-0008;0010-0009;0011-0010;"
    for pair in rota_lactec.split(';'):
        if '-' in pair:
            child, parent = pair.split('-')
            routes[int(child)] = int(parent)

    try:
        df = pd.read_excel(input_file, header=None)
    except Exception as e:
        print(f"Erro ao ler Excel: {e}")
        return []

    re_node_id = re.compile(r'\d+\.\d+\s+(\d+)\s+\[INFO')
    re_rx = re.compile(r'MESSAGE ->.*?->.*?t0=(\d+).*?lat\([A-Z]+->[A-Z]+\)\s*=\s*(\d+)\s*ms')

    for index, row in df.iterrows():
        line = ' '.join(str(val) for val in row.values if pd.notna(val))

        if 'UDP ROUTER TX' in line or 'UDP BORDER TX' in line:
            dados_tx = _extrair_dados_tx(line)
            match_node = re_node_id.search(line)
            
            if dados_tx and match_node:
                uc_type, tamanho, t0 = dados_tx
                node_id = int(match_node.group(1))

                hops, curr, visited = 0, node_id, set()
                while curr in routes and routes.get(curr) != curr and curr not in visited:
                    visited.add(curr)
                    hops += 1
                    curr = routes.get(curr, curr)
                
                packets[str(t0)] = {
                    'topologia': 'Linear', 'tipo_trafego': uc_type, 
                    'tamanho_bytes': tamanho, 'hop_count': max(1, hops), 
                    'latencia_ms': 0.0, 'entregue': 0, 't0': t0
                }

        elif 'UDP BORDER RX' in line or 'UDP ROUTER RX' in line:
            match_rx = re_rx.search(line)
            if match_rx:
                t0_rx = match_rx.group(1).strip()
                if t0_rx in packets:
                    packets[t0_rx]['entregue'] = 1
                    packets[t0_rx]['latencia_ms'] = float(match_rx.group(2).strip())

    return list(packets.values())

def parse_txt_topology(diretorio_base: str, nome_topologia: str) -> list:
    print(f"Varrendo pasta ({nome_topologia}): {diretorio_base}...")
    todos_pacotes = []
    
    if not os.path.exists(diretorio_base):
        print(f"  -> Diretório {diretorio_base} não encontrado. Pulando...")
        return []

    txt_files = glob.glob(os.path.join(diretorio_base, '**', '*.txt'), recursive=True)
    
    re_route = re.compile(r'\$ROUTES\$(.*)')
    re_node_id = re.compile(r'^\d+\.\d+\s*;\s*(\d+)\s*;')
    re_rx = re.compile(r'MESSAGE ->.*?->.*?t0=(\d+).*?lat\([A-Z]+->[A-Z]+\)\s*=\s*;?(\d+);?\s*ms')

    for filepath in txt_files:
        routes = {}
        packets = {}
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if '$ROUTES$' in line:
                    match = re_route.search(line)
                    if match:
                        route_str = match.group(1).strip()
                        for pair in route_str.split(';'):
                            if '-' in pair:
                                child, parent = pair.split('-')
                                routes[int(child)] = int(parent)

                elif 'UDP ROUTER TX' in line or 'UDP BORDER TX' in line:
                    dados_tx = _extrair_dados_tx(line)
                    match_node = re_node_id.search(line)
                    
                    if dados_tx and match_node:
                        uc_type, tamanho, t0 = dados_tx
                        node_id = int(match_node.group(1))

                        hops, curr, visited = 0, node_id, set()
                        while curr in routes and routes.get(curr) != curr and curr not in visited:
                            visited.add(curr)
                            hops += 1
                            curr = routes.get(curr, curr)
                        
                        packets[str(t0)] = {
                            'topologia': nome_topologia, 'tipo_trafego': uc_type, 
                            'tamanho_bytes': tamanho, 'hop_count': max(1, hops), 
                            'latencia_ms': 0.0, 'entregue': 0, 't0': t0
                        }

                elif 'UDP BORDER RX' in line or 'UDP ROUTER RX' in line:
                    match_rx = re_rx.search(line)
                    if match_rx:
                        t0_rx = match_rx.group(1).strip()
                        if t0_rx in packets:
                            packets[t0_rx]['entregue'] = 1
                            packets[t0_rx]['latencia_ms'] = float(match_rx.group(2).strip())
                            
        todos_pacotes.extend(packets.values())
        
    return todos_pacotes

def consolidar_logs():
    dados_totais = []
    
    excel_linear = r"C:\Users\adriely\Desktop\SGHaul_DRL_Project_2\data\linhas_UDP.xlsx"
    dados_totais.extend(parse_excel_linear(excel_linear))
    
    dir_mesh = r"C:\Users\adriely\Desktop\SGHaul_DRL_Project_2\LOG__ALL_DLMS\51_mesh"
    dados_totais.extend(parse_txt_topology(dir_mesh, "Mesh"))
    
    dir_quadrado = r"C:\Users\adriely\Desktop\SGHaul_DRL_Project_2\LOG__ALL_DLMS\101_quadrado"
    dados_totais.extend(parse_txt_topology(dir_quadrado, "Quadrada"))

    output_file = "data/dados_lactec_raw.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['topologia', 'tipo_trafego', 'tamanho_bytes', 'hop_count', 'latencia_ms', 'entregue', 't0'])
        writer.writeheader()
        writer.writerows(dados_totais)

    print("\n--- EXTRAÇÃO CONSOLIDADA CONCLUÍDA ---")
    print(f"Total de pacotes mapeados: {len(dados_totais)}")
    
    # Validação final: Quantos pacotes ficaram com tamanho 64?
    fallbacks = sum(1 for p in dados_totais if p['tamanho_bytes'] == 64 and p['tipo_trafego'] == 'UC3')
    if fallbacks > 0:
        print(f"[ALERTA GERAL] {fallbacks} pacotes UC3 não tinham 'UDP SIZE' no log e receberam 64 bytes por padrão.")

if __name__ == "__main__":
    consolidar_logs()