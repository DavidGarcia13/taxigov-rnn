"""
Unifica todos os datasets do TaxiGOV em um unico CSV padronizado.

Fontes:
  - 8 arquivos anuais 2017-2024 (esquema completo, 41 colunas)
  - taxigov-corridas-completo2.csv (26 colunas, nomes divergentes, ~165k corridas exclusivas)
Descartado: taxigov-corridas-completo.csv (redundante com os anuais).

Tratamentos:
  - Padroniza para o esquema canonico de 41 colunas (datapackage.yaml).
  - data_abertura vazia -> preenchida com data_inicio.
  - Deduplicacao por (base_origem, qru_corrida), mantendo a 1a ocorrencia.
  - Encoding UTF-8 (os arquivos ja estao em UTF-8 correto).
"""
import pandas as pd
import glob, os, csv

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz do projeto
BASE = os.path.join(RAIZ, "data", "raw")
SAIDA = os.path.join(BASE, "taxigov-corridas-unificado.csv")

# ---- esquema canonico (ordem oficial do datapackage.yaml) ----
COLS = [
    "base_origem","qru_corrida","orgao_nome","orgao_codigo_siorg","orgao_categoria",
    "unidade_administrativa_nome","unidade_administrativa_codigo_siorg","status","motivo",
    "justificativa","km_total","km_rota_calculada","valor_corrida","data_abertura",
    "data_despacho","data_local_embarque","data_inicio","data_final","data_cancelamento",
    "origem_endereco","origem_bairro","origem_cidade","origem_uf","destino_solicitado_endereco",
    "destino_efetivo_endereco","origem_latitude","origem_longitude","destino_solicitado_latitude",
    "destino_solicitado_longitude","destino_efetivo_latitude","destino_efetivo_longitude",
    "avaliacao_usuario_nota","avaliacao_usuario_texto","veiculo_fabricante","veiculo_modelo",
    "veiculo_ano_fabricacao","veiculo_ano_modelo","veiculo_cor","veiculo_placa",
    "ateste_setorial_data","conteste_info",
]

# ---- mapeamento das colunas do completo2 -> esquema canonico ----
MAP_COMPLETO2 = {
    "nome_orgao": "orgao_nome",
    "status_corrida": "status",
    "motivo_corrida": "motivo",
}

anuais = sorted(glob.glob(os.path.join(BASE, "taxigov-corridas-*", "taxigov-corridas-*.csv")))
completo2 = os.path.join(BASE, "taxigov-corridas-completo2.csv")
FONTES = [(p, "anual") for p in anuais] + [(completo2, "completo2")]

vistos = set()          # chaves (base_origem|qru_corrida) ja gravadas
stats = {
    "lidas": 0, "gravadas": 0, "dup_removidas": 0,
    "abertura_preenchida": 0, "abertura_ainda_nula": 0,
    "por_fonte": {},
}

def norm_key_qru(x):
    # 463653.0 -> 463653 ; mantem strings como estao
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return str(x)

primeira_escrita = True

for path, tipo in FONTES:
    nome = os.path.basename(path)
    stats["por_fonte"][nome] = {"lidas": 0, "gravadas": 0, "dup": 0, "abertura_fix": 0}
    for ch in pd.read_csv(path, encoding="utf-8", dtype=str, chunksize=200_000,
                          low_memory=False, keep_default_na=True):
        # renomeia colunas do completo2
        if tipo == "completo2":
            ch = ch.rename(columns=MAP_COMPLETO2)
        # garante todas as colunas do esquema canonico (faltantes = NA)
        for c in COLS:
            if c not in ch.columns:
                ch[c] = pd.NA
        ch = ch[COLS]

        n = len(ch)
        stats["lidas"] += n
        stats["por_fonte"][nome]["lidas"] += n

        # data_abertura vazia -> data_inicio
        falta = ch["data_abertura"].isna() | (ch["data_abertura"].astype(str).str.strip() == "")
        preencheu = falta & ch["data_inicio"].notna() & (ch["data_inicio"].astype(str).str.strip() != "")
        ch.loc[preencheu, "data_abertura"] = ch.loc[preencheu, "data_inicio"]
        stats["abertura_preenchida"] += int(preencheu.sum())
        stats["por_fonte"][nome]["abertura_fix"] += int(preencheu.sum())

        # deduplicacao por (base_origem, qru_corrida) normalizada
        key = ch["base_origem"].astype(str) + "|" + ch["qru_corrida"].map(norm_key_qru)
        manter = []
        for k in key:
            if k in vistos:
                manter.append(False)
            else:
                vistos.add(k)
                manter.append(True)
        manter = pd.Series(manter, index=ch.index)
        dup = int((~manter).sum())
        stats["dup_removidas"] += dup
        stats["por_fonte"][nome]["dup"] += dup
        ch = ch[manter]

        # grava
        ch.to_csv(SAIDA, mode="w" if primeira_escrita else "a",
                  header=primeira_escrita, index=False, encoding="utf-8",
                  quoting=csv.QUOTE_MINIMAL)
        primeira_escrita = False
        stats["gravadas"] += len(ch)
        stats["por_fonte"][nome]["gravadas"] += len(ch)

# quantas data_abertura ainda ficaram nulas no final
abertura_nula = 0
for ch in pd.read_csv(SAIDA, usecols=["data_abertura"], dtype=str, chunksize=200_000, low_memory=False):
    s = ch["data_abertura"]
    abertura_nula += int((s.isna() | (s.astype(str).str.strip()=="")).sum())
stats["abertura_ainda_nula"] = abertura_nula

print("="*70)
print("UNIFICACAO CONCLUIDA ->", SAIDA)
print("="*70)
print(f"Linhas lidas (todas as fontes):        {stats['lidas']:>10,}")
print(f"Duplicatas removidas:                  {stats['dup_removidas']:>10,}")
print(f"Linhas gravadas (unico):               {stats['gravadas']:>10,}")
print(f"data_abertura preenchida c/ data_inicio: {stats['abertura_preenchida']:>8,}")
print(f"data_abertura ainda nula (sem fallback): {stats['abertura_ainda_nula']:>8,}")
print("\nPor fonte:")
for nome, s in stats["por_fonte"].items():
    print(f"  {nome:35} lidas={s['lidas']:>8,}  gravadas={s['gravadas']:>8,}  dup={s['dup']:>6,}  abertura_fix={s['abertura_fix']:>6,}")
