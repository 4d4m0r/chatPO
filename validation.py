import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

CSV_PATH = "./csv/perguntas_respostas_llm.csv"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

df = pd.read_csv(CSV_PATH, encoding="utf-8")
assert {"question","resposta_llm","answer"}.issubset(df.columns), "Faltam colunas obrigatórias."

model = SentenceTransformer(EMBED_MODEL)
ref_emb = model.encode(df["answer"].fillna("").tolist(), convert_to_numpy=True, normalize_embeddings=True)
pred_emb = model.encode(df["resposta_llm"].fillna("").tolist(), convert_to_numpy=True, normalize_embeddings=True)
sim = (ref_emb * pred_emb).sum(axis=1)

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_l = []
for ref, pred in zip(df["answer"].fillna(""), df["resposta_llm"].fillna("")):
    scores = scorer.score(ref, pred)
    rouge_l.append(scores["rougeL"].fmeasure)
rouge_l = np.array(rouge_l)

def contains_all_keywords(pred, ref, min_keywords=2):
    kws = [w for w in set(ref.lower().split()) if len(w) > 3]
    hit = sum(int(kw in pred.lower()) for kw in kws)
    return int(hit >= min_keywords)

df["sim_cosseno"] = sim
df["rougeL_f"] = rouge_l
df["len_pred"] = df["resposta_llm"].fillna("").apply(lambda s: len(s.split()))
df["kw_coverage2+"] = [
    contains_all_keywords(pred, ref, min_keywords=2)
    for pred, ref in zip(df["resposta_llm"].fillna(""), df["answer"].fillna(""))
]

THRESH = 0.60

df["acerto_bin"] = (df["sim_cosseno"] >= THRESH)

resumo = {
    "n": len(df),
    "sim_cosseno_medio": float(df["sim_cosseno"].mean()),
    "rougeL_medio": float(df["rougeL_f"].mean()),
    "kw_coverage2+_%": float(df["kw_coverage2+"].mean())
}
print("Resumo:", resumo)

if "categoria" in df.columns:
    print("\nPor categoria:")
    print(
        df.groupby("categoria")[["sim_cosseno","rougeL_f","acerto_bin","kw_coverage2+"]]
          .mean()
          .sort_values("sim_cosseno", ascending=False)
    )

OUT_PATH = CSV_PATH.replace(".csv", "_avaliado.csv")
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"Arquivo salvo em: {OUT_PATH}")
