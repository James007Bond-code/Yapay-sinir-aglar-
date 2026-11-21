"""
Grup projesi – dört yöntem ailesi (A–D) için ortak nested CV boru hattı
=======================================================================
Her yöntem aynı veri, aynı seed, aynı dış/iç katlar ve aynı 6 manuel + 6 otomatik
aday listesiyle **baştan sona** çalıştırılır. Yöntemler arasındaki tek fark
METHODS sözlüğündeki yöntem-düzeyi kuraldır:

  A  Standart (yönerge akışı)               – değişiklik yok, eşik 0.50
  B  Sınıf ağırlıklı kayıp                  – BCE pos_weight = kat-içi neg/pos
  C  İç CV'de seçilen karar eşiği            – A ile aynı eğitim; eşik iç CV out-of-fold
                                               tahminlerinde F1-optimal olarak seçilir
  D  Alternatif ön işleme                   – unknown → en sık sınıf, P1–P99 clipping,
                                               pdays = 999 ham kodlama (gösterge yok)
"""

import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .ysa_utils import (
    MANUAL_CONFIGS, compute_metrics, build_preprocessor, f1_optimal_threshold, history_frame,
    index_hash, make_cv_splits, make_random_configs, result_record, run_cached, train_mlp,
)

METHODS = {
    "A": {"name": "Standart (yönerge akışı)", "override": {}, "threshold_mode": "fixed",
          "training_from": "A",
          "how": "Adaylar yönergede tanımlandığı gibi eğitilir: ağırlıksız BCE, `unknown` ayrı kategori, "
                 "M1–M6'da kırpma yok, `pdays = 999` → `prev_contacted` göstergesi + 0. Karar eşiği 0.50.",
          "hypothesis": "Yönergenin temel akışı, diğer yöntemlerin karşılaştırılacağı referans performansı verir."},
    "B": {"name": "Sınıf ağırlıklı kayıp", "override": {"pos_weight": "auto"}, "threshold_mode": "fixed",
          "training_from": "B",
          "how": "Her eğitimde eğitim kaybı `BCEWithLogitsLoss(pos_weight = neg/pos)` olur. Oran, o eğitimin fit alt "
                 "kümesinden hesaplanır (kat-içi, sızıntısız). Early stopping ve scheduler ağırlıksız BCE'yi izler. Eşik 0.50.",
          "hypothesis": "Azınlık sınıfa daha fazla ağırlık vermek 0.50 eşiğinde Sensitivity'yi belirgin artırır, "
                        "sıralama metriği AP'yi az değiştirir, olasılık kalibrasyonunu (Brier) bozar."},
    "C": {"name": "İç CV'de seçilen karar eşiği", "override": {}, "threshold_mode": "inner_oof_f1",
          "training_from": "A",
          "how": "Model eğitimi ve aday seçimi Yöntem A ile aynıdır (aynı ağırlıklar). Fark karar kuralındadır: "
                 "seçilen adayın iç CV out-of-fold tahminlerinde F1'i en yüksek yapan eşik seçilir ve dış değerlendirme "
                 "ile testte bu eşik kullanılır. Eşik hiçbir zaman değerlendirme/test verisiyle seçilmez.",
          "hypothesis": "Dengesiz veride 0.50 eşiği gereğinden yüksektir. İç CV ile seçilen eşik AP'yi değiştirmeden "
                        "F1, Sensitivity ve Balanced Accuracy'yi artırır."},
    "D": {"name": "Alternatif ön işleme", "override": {"unknown_strategy": "most_frequent", "clip_outliers": True,
                                                       "pdays_mode": "raw"},
          "threshold_mode": "fixed", "training_from": "D",
          "how": "Tüm adaylarda `unknown` kategorileri kat-içi en sık sınıfla doldurulur, sayısal değişkenler eğitim katında "
                 "öğrenilen P1–P99 sınırlarıyla kırpılır, `pdays` ham kodlamasıyla (999 dahil) kullanılır ve "
                 "`prev_contacted` göstergesi eklenmez. Diğer her şey A ile aynıdır. Eşik 0.50.",
          "hypothesis": "`unknown`'ı bilgi taşıyan ayrı bir kategori olarak tutmak ve `pdays` özel kodunu ayrı göstermek "
                        "bilgi kaybını önler. Bu yüzden D'nin AP'si A'dan eşit veya düşük olur."},
}
METHOD_ORDER = ["A", "B", "C", "D"]


def method_candidates(mid, seeds):
    """Ortak 6 manuel + 6 otomatik aday listesine yöntem-düzeyi kuralı uygular."""
    base = MANUAL_CONFIGS + make_random_configs(seeds["random_search_seed"])
    tm = METHODS[mid]["training_from"]
    return [dict(c, **METHODS[tm]["override"]) for c in base]


def _run(key, X_tr, y_tr, g_tr, X_ev, cfg, num_cols, cat_cols, model_seed, es_seed):
    extra = {"model_seed": model_seed, "es_seed": es_seed,
             "train_idx": index_hash(X_tr.index), "eval_idx": index_hash(X_ev.index)}
    def fn():
        res = train_mlp(X_tr, y_tr, {"eval": X_ev}, cfg, num_cols, cat_cols,
                        model_seed=model_seed, es_seed=es_seed, groups=g_tr)
        return result_record(res, "eval")
    return run_cached(key, cfg, extra, fn)


def _select(df, stype):
    s = (df[df.search_type == stype].groupby("config_id", sort=False)["inner_AP"]
         .agg(mean="mean", sd="std").reset_index())
    s["_o"] = range(len(s))
    s = s.sort_values(["mean", "sd", "_o"], ascending=[False, True, True])
    return s.iloc[0]


# ---------------------------------------------------------------------------
def run_method_nested(mid, X_dev, y_dev, g_dev, seeds, num_cols, cat_cols, verbose=True):
    """3×3 nested CV: 108 iç eğitim + 6 dış refit (aynı katlar tüm yöntemlerde)."""
    meth = METHODS[mid]
    tm = meth["training_from"]
    cands = method_candidates(mid, seeds)
    y_dev = pd.Series(np.asarray(y_dev), index=X_dev.index)
    g_dev = np.asarray(g_dev)
    outer_splits = make_cv_splits(y_dev, g_dev, 3, seeds["outer_cv_seed"])
    inner_rows, outer_rows, sel_rows, logs = [], [], [], []
    t0 = time.time()
    for k, (otr, oev) in enumerate(outer_splits, start=1):
        X_otr, y_otr, g_otr = X_dev.iloc[otr], y_dev.iloc[otr], g_dev[otr]
        X_oev, y_oev = X_dev.iloc[oev], y_dev.iloc[oev]
        inner_splits = make_cv_splits(y_otr, g_otr, 3, seeds["inner_cv_seed"] + k)
        oof = {c["config_id"]: np.full(len(otr), np.nan) for c in cands}
        for cfg in cands:
            for j, (itr, iva) in enumerate(inner_splits, start=1):
                rec = _run(f"{tm}_outer{k}_inner{j}_{cfg['config_id']}", X_otr.iloc[itr], y_otr.iloc[itr], g_otr[itr],
                           X_otr.iloc[iva], cfg, num_cols, cat_cols,
                           seeds["model_init_seeds"][0] + 100 * k + j, seeds["early_stop_seed"] + 100 * k + j)
                oof[cfg["config_id"]][iva] = rec["probs"]
                inner_rows.append({"method": mid, "outer_fold": k, "inner_fold": j, "config_id": cfg["config_id"],
                                   "search_type": cfg["search_type"], "optimizer": cfg["optimizer"],
                                   "inner_AP": float(average_precision_score(y_otr.iloc[iva], rec["probs"])),
                                   **{x: rec[x] for x in ["best_epoch", "stopped_epoch", "n_lr_drops", "n_params",
                                                          "input_dim", "elapsed_sec", "status", "warnings",
                                                          "pos_weight_used"]},
                                   "from_cache": rec["from_cache"]})
                logs.append(history_frame(rec, method=mid, phase="nested_inner", outer_fold=k, inner_fold=j,
                                          config_id=cfg["config_id"]))
        dfk = pd.DataFrame([r for r in inner_rows if r["outer_fold"] == k])
        for label, stype in [("manual", "manual"), ("random", "random")]:
            best = _select(dfk, stype)
            cid = best["config_id"]
            cfg = next(c for c in cands if c["config_id"] == cid)
            thr = (f1_optimal_threshold(y_otr, oof[cid]) if meth["threshold_mode"] == "inner_oof_f1" else 0.50)
            rec = _run(f"{tm}_outer{k}_refit_{label}_{cid}", X_otr, y_otr, g_otr, X_oev, cfg, num_cols, cat_cols,
                       seeds["model_init_seeds"][0] + 1000 * k, seeds["early_stop_seed"] + 1000 * k)
            m = compute_metrics(y_oev, rec["probs"], thr)
            outer_rows.append({"method": mid, "search": label, "outer_fold": k, "selected_config": cid,
                               "inner_mean_AP": float(best["mean"]), "inner_sd_AP": float(best["sd"]),
                               "best_epoch": rec["best_epoch"], "elapsed_sec": rec["elapsed_sec"], **m})
            sel_rows.append({"method": mid, "outer_fold": k, "search": label, "selected_config": cid,
                             "inner_mean_AP": float(best["mean"]), "threshold": thr})
            logs.append(history_frame(rec, method=mid, phase="outer_refit", outer_fold=k, inner_fold=0,
                                      config_id=cid))
        if verbose:
            sel = ", ".join(f"{r['search']}→{r['selected_config']}" for r in sel_rows if r["outer_fold"] == k)
            print(f"[{mid}] dış kat {k} tamam ({time.time() - t0:.0f} sn) | seçim: {sel}")
    inner = pd.DataFrame(inner_rows)
    assert len(inner) == 108, len(inner)
    return {"inner": inner, "outer": pd.DataFrame(outer_rows), "selection": pd.DataFrame(sel_rows),
            "logs": pd.concat(logs, ignore_index=True)}


def run_references_nested(X_dev, y_dev, g_dev, seeds, num_cols, cat_cols):
    """Dummy ve Logistic Regression: aynı dış katlarda outer-train → outer-eval."""
    y_dev = pd.Series(np.asarray(y_dev), index=X_dev.index)
    rows = []
    for k, (otr, oev) in enumerate(make_cv_splits(y_dev, g_dev, 3, seeds["outer_cv_seed"]), start=1):
        X_otr, y_otr, X_oev, y_oev = X_dev.iloc[otr], y_dev.iloc[otr], X_dev.iloc[oev], y_dev.iloc[oev]
        p = DummyClassifier(strategy="most_frequent").fit(X_otr, y_otr).predict_proba(X_oev)[:, 1]
        rows.append({"method": "Dummy", "search": "-", "outer_fold": k, "selected_config": "most_frequent",
                     **compute_metrics(y_oev, p)})
        p = fit_logreg(X_otr, y_otr, X_oev, seeds, num_cols, cat_cols)
        rows.append({"method": "LogReg", "search": "-", "outer_fold": k, "selected_config": "L2, C=1",
                     **compute_metrics(y_oev, p)})
    return pd.DataFrame(rows)


def fit_logreg(X_tr, y_tr, X_ev, seeds, num_cols, cat_cols):
    pre = build_preprocessor(num_cols, cat_cols, "StandardScaler", False, "keep_category")
    lr = LogisticRegression(max_iter=2000, random_state=seeds["logreg_seed"])
    lr.fit(pre.fit_transform(X_tr), y_tr)
    return lr.predict_proba(pre.transform(X_ev))[:, 1]


# ---------------------------------------------------------------------------
def run_method_fulldev(mid, X_dev, y_dev, g_dev, seeds, num_cols, cat_cols):
    """Tam development 3-kat iç CV (36 eğitim) → yöntemin nihai konfigürasyonu ve eşiği."""
    meth = METHODS[mid]
    tm = meth["training_from"]
    cands = method_candidates(mid, seeds)
    y_dev = pd.Series(np.asarray(y_dev), index=X_dev.index)
    g_dev = np.asarray(g_dev)
    splits = make_cv_splits(y_dev, g_dev, 3, seeds["inner_cv_seed"])
    rows, logs = [], []
    oof = {c["config_id"]: np.full(len(y_dev), np.nan) for c in cands}
    for cfg in cands:
        for j, (itr, iva) in enumerate(splits, start=1):
            rec = _run(f"{tm}_fulldev_inner{j}_{cfg['config_id']}", X_dev.iloc[itr], y_dev.iloc[itr], g_dev[itr],
                       X_dev.iloc[iva], cfg, num_cols, cat_cols,
                       seeds["model_init_seeds"][0] + 5000 + j, seeds["early_stop_seed"] + 5000 + j)
            oof[cfg["config_id"]][iva] = rec["probs"]
            rows.append({"method": mid, "config_id": cfg["config_id"], "search_type": cfg["search_type"],
                         "inner_fold": j, "inner_AP": float(average_precision_score(y_dev.iloc[iva], rec["probs"])),
                         **{x: rec[x] for x in ["best_epoch", "stopped_epoch", "n_lr_drops", "n_params", "input_dim",
                                                "elapsed_sec", "status", "warnings", "pos_weight_used"]}})
            logs.append(history_frame(rec, method=mid, phase="full_dev_inner", outer_fold=0, inner_fold=j,
                                      config_id=cfg["config_id"]))
    folds = pd.DataFrame(rows)
    assert len(folds) == 36
    agg = (folds.groupby("config_id", sort=False)
           .agg(search_type=("search_type", "first"), mean_AP=("inner_AP", "mean"), sd_AP=("inner_AP", "std"),
                mean_best_epoch=("best_epoch", "mean"), mean_lr_drops=("n_lr_drops", "mean"),
                n_params=("n_params", "median"), total_time_sec=("elapsed_sec", "sum"),
                mean_pos_weight=("pos_weight_used", "mean"))
           .reset_index())
    params = pd.DataFrame(cands)
    params["hidden_layers"] = params["hidden_layers"].astype(str)
    keep = ["config_id", "optimizer", "hidden_layers", "activation", "lr", "weight_decay", "batch_size", "scaler",
            "dropout", "use_scheduler", "clip_outliers", "unknown_strategy"]
    summary = params[keep].merge(agg, on="config_id")
    summary.insert(0, "method", mid)

    def best(stype):
        s = summary[summary.search_type == stype].sort_values(["mean_AP", "sd_AP"], ascending=[False, True])
        return s.iloc[0]
    bm, br = best("manual"), best("random")
    win = br if (br["mean_AP"] > bm["mean_AP"] or (br["mean_AP"] == bm["mean_AP"] and br["sd_AP"] < bm["sd_AP"])) else bm
    cfg = dict(next(c for c in cands if c["config_id"] == win["config_id"]))
    thr = f1_optimal_threshold(y_dev, oof[win["config_id"]]) if meth["threshold_mode"] == "inner_oof_f1" else 0.50
    final = {"method": mid, "method_name": meth["name"], "config": cfg, "config_id": win["config_id"],
             "selected_search": "random_search" if win["search_type"] == "random" else "manual_search",
             "full_dev_mean_AP": float(win["mean_AP"]), "full_dev_sd_AP": float(win["sd_AP"]),
             "best_manual": {"config_id": bm["config_id"], "mean_AP": float(bm["mean_AP"]), "sd_AP": float(bm["sd_AP"])},
             "best_random": {"config_id": br["config_id"], "mean_AP": float(br["mean_AP"]), "sd_AP": float(br["sd_AP"])},
             "threshold": float(thr), "threshold_mode": meth["threshold_mode"], "training_from": tm}
    oof_df = pd.DataFrame({f"oof_{k}": v for k, v in oof.items()})
    return {"folds": folds, "summary": summary.sort_values("mean_AP", ascending=False), "final": final,
            "oof": oof_df, "logs": pd.concat(logs, ignore_index=True)}


# ---------------------------------------------------------------------------
def md_table(df, digits=4):
    """tabulate gerektirmeyen basit Markdown tablo."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in d.astype(str).values]
    return "\n".join([head, sep] + body)
