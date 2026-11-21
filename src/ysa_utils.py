"""
YSA Ara Sınav Projesi – ortak yardımcı modül
=============================================
01_eda, 02_nested_cv ve 03_final_test defterlerinin TAMAMI bu dosyadaki
fonksiyonları kullanır. Böylece nested CV'de seçilen model ile nihai testte
eğitilen model birebir aynı kodla (aynı ön işleme, aynı eğitim döngüsü)
üretilir.

İçerik
------
- Seed türetme ve sabitleme
- Kural tabanlı (fit gerektirmeyen) özellik hazırlama: pdays = 999 kodu
- Kat-içi (fold-local) ön işleme: imputer → clipper → scaler | unknown → one-hot
- PyTorch MLP, optimizer, ReduceLROnPlateau + early stopping eğitim döngüsü
- Metrikler, Mean ± SD özetleri ve bootstrap güven aralıkları
- Manuel (M1–M6) ve otomatik (R1–R6) hiperparametre adayları
"""

import copy
import hashlib
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    auc, brier_score_loss, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

# ---------------------------------------------------------------------------
# 0. Sabitler
# ---------------------------------------------------------------------------
TARGET = "y"
ID_COL = "original_index"
GROUP_COL = "dup_group"          # kopya grubu kimliği (duration hariç aynı girdiler)
DROP_COLS = ["duration"]           # Bölüm 2.2: görüşme öncesi bilinmiyor
PDAYS_NO_CONTACT = 999             # Bölüm 2.4: "daha önce temas yok" kodu
CLIP_EXCLUDE = ["pdays", "prev_contacted"]   # Özel kodlu / ikili sütunlar kırpılmaz

TRAINING_SETTINGS = {
    "max_epochs": 150,
    "early_stopping": {"monitor": "internal_val_BCE", "mode": "min",
                       "patience": 15, "min_delta": 1e-6, "restore_best_weights": True},
    "early_stop_subset": {"fraction_of_training_part": 0.15, "stratified": True},
    "lr_scheduler": {"name": "ReduceLROnPlateau", "mode": "min", "factor": 0.5,
                     "patience": 5, "min_lr": 1e-5, "monitor": "internal_val_BCE"},
    "loss": "BCEWithLogitsLoss (tek logit çıkış, olasılık = sigmoid(logit))",
    "selection_metric": "average_precision_score (inner CV ortalama AP)",
    "threshold": 0.50,
    "sgd_momentum": 0.9,
    "clip_quantiles": [0.01, 0.99],
    "device": "cpu",
}


# ---------------------------------------------------------------------------
# 1. Seed yönetimi
# ---------------------------------------------------------------------------
def derive_seeds(student_last5: int) -> dict:
    """Bölüm 3: Temel seed = öğrenci numarasının son 5 hanesi + 2026.
    Tüm alt seed'ler temel seed'den sabit ofsetlerle deterministik türetilir."""
    base = int(student_last5) + 2026
    return {
        "student_last5": int(student_last5),
        "base_seed": base,
        "split_seed": base + 1,
        "outer_cv_seed": base + 2,
        "inner_cv_seed": base + 3,          # dış kat k için: base + 3 + k ; tam dev için: base + 3
        "random_search_seed": base + 4,
        "early_stop_seed": base + 5,
        "bootstrap_seed": base + 6,
        "logreg_seed": base + 7,
        "model_init_seeds": [base + 10, base + 20, base + 30],   # ilki birincil (primary)
        "derivation": "base = last5 + 2026; split=+1, outer=+2, inner=+3(+k), random_search=+4, "
                      "early_stop=+5, bootstrap=+6, logreg=+7, model_init=+10/+20/+30",
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)                    # CPU'da toplama sırası sabit → tekrarlanabilirlik
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False, default=_json_default)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# 2. Kural tabanlı özellik hazırlama (fit YOK → sızıntı riski yok)
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hedefi ve kimlik sütununu çıkarır, pdays = 999 kodunu iki sütuna ayırır.

    pdays = 999 bir aykırı değer değil, "daha önce temas yok" kodudur (Bölüm 2.4).
    Ham haliyle bırakıldığında (i) sayısal olarak 0–27 gün aralığındaki gerçek
    değerlerden ~40 kat büyük görünür, (ii) gözlemlerin ~%96'sı 999 olduğundan
    IQR = 0 olur ve RobustScaler bu sütunu ölçeklemeden [-999, 0] aralığında
    bırakır. Bu nedenle satır bazlı, veri istatistiği kullanmayan bir kural
    uygulanır:
        prev_contacted = 1[pdays != 999]
        pdays          = pdays  (temas varsa) | 0 (temas yoksa)
    Hiçbir gözlem silinmez; bilgi kaybı yoktur.
    """
    X = df.drop(columns=[c for c in [TARGET, ID_COL, GROUP_COL] + DROP_COLS if c in df.columns]).copy()
    no_contact = X["pdays"] == PDAYS_NO_CONTACT
    X["prev_contacted"] = (~no_contact).astype(int)
    X["pdays"] = np.where(no_contact, 0, X["pdays"]).astype(float)
    return X


def get_feature_lists(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols


# ---------------------------------------------------------------------------
# 2b. Kopya grubu farkındalıklı bölünmeler
# ---------------------------------------------------------------------------
def duplicate_groups(df: pd.DataFrame, feature_cols) -> pd.Series:
    """duration hariç tüm girdileri aynı olan satırlara ortak grup kimliği verir."""
    return df.groupby(list(feature_cols), sort=False, dropna=False).ngroup().astype(int).rename(GROUP_COL)


def group_aware_holdout(y, groups, test_size, seed):
    """Stratified hold-out; ardından iki tarafa bölünmüş kopya grupları bütünüyle
    eğitim tarafına taşınır. Kopya yoksa sonuç sıradan stratified split ile aynıdır.
    Dönüş: (train_pos, holdout_pos) – konum indeksleri."""
    y = np.asarray(y)
    pos = np.arange(len(y))
    tr, te = train_test_split(pos, test_size=test_size, stratify=y, random_state=seed)
    if groups is not None:
        g = np.asarray(groups)
        straddle = np.intersect1d(g[tr], g[te])
        if len(straddle):
            move = np.isin(g[te], straddle)
            tr = np.concatenate([tr, te[move]])
            te = te[~move]
    return np.sort(tr), np.sort(te)


def make_cv_splits(y, groups, n_splits, seed):
    """Kopya yoksa StratifiedKFold; varsa aynı grup tek kata düşsün diye StratifiedGroupKFold."""
    y = np.asarray(y)
    dummy = np.zeros(len(y))
    if groups is None or len(np.unique(groups)) == len(y):
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(dummy, y))
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(dummy, y, np.asarray(groups)))


# ---------------------------------------------------------------------------
# 2c. Önbellek (Colab bağlantısı koparsa kaldığı yerden devam)
# ---------------------------------------------------------------------------
RESULT_KEYS = ["best_epoch", "stopped_epoch", "best_es_loss", "status", "warnings", "n_lr_drops",
               "final_lr", "n_params", "input_dim", "n_fit", "n_es", "n_es_pos", "elapsed_sec"]


def index_hash(index) -> str:
    return hashlib.sha1(np.asarray(index).astype(np.int64).tobytes()).hexdigest()[:12]


def run_hash(cfg, extra) -> str:
    keys = [k for k in cfg if k not in ("purpose",)]
    blob = json.dumps({"cfg": {k: cfg[k] for k in sorted(keys)}, "extra": extra, "settings": TRAINING_SETTINGS},
                      sort_keys=True, default=_json_default)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def run_cached(key, cfg, extra, fn, cache_dir="results/cache"):
    """fn() JSON'a yazılabilir bir sözlük döndürmeli. Aynı anahtar + aynı hash varsa yeniden eğitilmez."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.json")
    h = run_hash(cfg, extra)
    if os.path.exists(path):
        try:
            rec = load_json(path)
            if rec.get("hash") == h:
                rec["from_cache"] = True
                return rec
        except Exception:
            pass
    rec = fn()
    rec["hash"] = h
    save_json(rec, path)
    rec["from_cache"] = False
    return rec


def result_record(res, eval_name) -> dict:
    """train_mlp çıktısından JSON'a yazılabilir kayıt (olasılıklar + geçmiş + bilgi)."""
    rec = {k: res[k] for k in RESULT_KEYS}
    rec["probs"] = [float(v) for v in res["probs"][eval_name]]
    rec["history"] = res["history"]
    return rec


# ---------------------------------------------------------------------------
# 3. Kat-içi ön işleme bileşenleri
# ---------------------------------------------------------------------------
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Eğitim alt kümesinde öğrenilen yüzdelik sınırlarla kırpma (winsorizing).

    NOT: Pipeline içinde SimpleImputer numpy dizisi döndürdüğü için sütun ADI
    değil sütun İNDEKSİ ile çalışır (eski sürümdeki "pdays muafiyeti çalışmıyor"
    hatası bu şekilde giderildi).
    """

    def __init__(self, lower_q=0.01, upper_q=0.99, enabled=False, exclude_idx=()):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.enabled = enabled
        self.exclude_idx = exclude_idx

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[1]
        self.n_features_in_ = n
        self.lower_ = np.full(n, -np.inf)
        self.upper_ = np.full(n, np.inf)
        if self.enabled:
            excl = set(self.exclude_idx)
            cols = [j for j in range(n) if j not in excl]
            if cols:
                self.lower_[cols] = np.quantile(X[:, cols], self.lower_q, axis=0)
                self.upper_[cols] = np.quantile(X[:, cols], self.upper_q, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


class UnknownHandler(BaseEstimator, TransformerMixin):
    """'unknown' kategorisi için iki seçenek (Bölüm 2.3):
    - keep_category : 'unknown' ayrı bir kategori olarak kalır
    - most_frequent : 'unknown' → NaN, ardından kat-içi en sık sınıf ile doldurulur
    """

    def __init__(self, strategy="keep_category"):
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).astype(object).copy()
        if self.strategy == "most_frequent":
            X_df = X_df.replace("unknown", np.nan)
        return X_df


def make_scaler(name: str):
    if name == "StandardScaler":
        return StandardScaler()
    if name == "MinMaxScaler":
        return MinMaxScaler()
    if name == "RobustScaler":
        return RobustScaler()
    raise ValueError(f"Bilinmeyen scaler: {name}")


def build_preprocessor(num_cols, cat_cols, scaler_name="StandardScaler",
                       clip_outliers=False, unknown_strategy="keep_category"):
    exclude_idx = tuple(num_cols.index(c) for c in CLIP_EXCLUDE if c in num_cols)
    q_lo, q_hi = TRAINING_SETTINGS["clip_quantiles"]
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper(q_lo, q_hi, enabled=clip_outliers, exclude_idx=exclude_idx)),
        ("scaler", make_scaler(scaler_name)),
    ])
    cat_steps = [("unk_handler", UnknownHandler(strategy=unknown_strategy))]
    if unknown_strategy == "most_frequent":
        cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", Pipeline(cat_steps), cat_cols),
    ])


def get_learned_stats(preprocessor, num_cols, cat_cols) -> dict:
    """Fit edilmiş ön işlemcinin öğrendiği istatistikleri JSON'a yazılabilir hale getirir
    (Bölüm 12.3: öğrenilmiş dönüşüm istatistikleri saklanacak)."""
    num_pipe = preprocessor.named_transformers_["num"]
    cat_pipe = preprocessor.named_transformers_["cat"]
    scaler = num_pipe.named_steps["scaler"]
    clipper = num_pipe.named_steps["clipper"]
    stats = {
        "numeric_columns": num_cols,
        "imputer_median": dict(zip(num_cols, num_pipe.named_steps["imputer"].statistics_.tolist())),
        "clip_enabled": bool(clipper.enabled),
        "clip_lower": dict(zip(num_cols, [None if np.isinf(v) else float(v) for v in clipper.lower_])),
        "clip_upper": dict(zip(num_cols, [None if np.isinf(v) else float(v) for v in clipper.upper_])),
        "scaler": type(scaler).__name__,
    }
    for attr in ["mean_", "scale_", "center_", "data_min_", "data_max_"]:
        if hasattr(scaler, attr) and getattr(scaler, attr) is not None:
            stats[f"scaler_{attr.rstrip('_')}"] = dict(zip(num_cols, getattr(scaler, attr).tolist()))
    onehot = cat_pipe.named_steps["onehot"]
    stats["onehot_categories"] = {c: [str(v) for v in cats] for c, cats in zip(cat_cols, onehot.categories_)}
    if "imputer" in cat_pipe.named_steps:
        stats["cat_imputer_most_frequent"] = dict(
            zip(cat_cols, [str(v) for v in cat_pipe.named_steps["imputer"].statistics_]))
    stats["n_output_features"] = int(len(num_cols) + sum(len(c) for c in onehot.categories_))
    return stats


# ---------------------------------------------------------------------------
# 4. MLP modeli ve eğitim döngüsü
# ---------------------------------------------------------------------------
class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation="ReLU", dropout=0.0):
        super().__init__()
        act_cls = nn.ReLU if activation == "ReLU" else nn.Tanh
        layers, in_d = [], input_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_d, h), act_cls()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_d = h
        layers.append(nn.Linear(in_d, 1))           # tek logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_mlp_params(input_dim, hidden_layers) -> int:
    """Analitik parametre sayısı: Σ (girdi × çıktı + bias)."""
    dims = [input_dim] + list(hidden_layers) + [1]
    return int(sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1)))


def make_optimizer(model, cfg):
    opt = cfg["optimizer"]
    if opt == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if opt == "Adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if opt == "SGD":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                               momentum=TRAINING_SETTINGS["sgd_momentum"],
                               weight_decay=cfg["weight_decay"])
    raise ValueError(f"Bilinmeyen optimizer: {opt}")


def train_mlp(X_train_raw, y_train, eval_sets, cfg, num_cols, cat_cols,
              model_seed, es_seed, max_epochs=None, groups=None):
    """Tek bir MLP eğitimi (iç CV, dış kat refit ve nihai eğitim için ORTAK fonksiyon).

    X_train_raw / y_train : ağırlık eğitimi için kullanılan bölüm (ör. inner-train)
    eval_sets             : {"ad": X_raw} → yalnızca tahmin üretilecek kümeler
                            (inner validation, outer evaluation veya test).
                            Bu kümeler hiçbir fit/early-stop/scheduler kararında kullanılmaz.

    Üçlü ayrım (Bölüm 5):
      (i)  X_train_raw → %85 "fit" alt kümesi : ön işleme fit + ağırlık güncelleme
      (ii) X_train_raw → %15 "early-stop" alt kümesi : early stopping + scheduler (BCE)
      (iii) eval_sets : yalnızca transform + tahmin
    groups : kopya grubu kimlikleri; verilirse aynı kopya hem fit hem early-stop
             alt kümesine düşmez.
    """

    t0 = time.time()
    max_epochs = max_epochs or TRAINING_SETTINGS["max_epochs"]
    es_cfg = TRAINING_SETTINGS["early_stopping"]
    y_train = pd.Series(np.asarray(y_train), index=X_train_raw.index)

    fit_pos, es_pos = group_aware_holdout(
        y_train.values, groups,
        TRAINING_SETTINGS["early_stop_subset"]["fraction_of_training_part"], es_seed)
    X_fit_raw, X_es_raw = X_train_raw.iloc[fit_pos], X_train_raw.iloc[es_pos]
    y_fit, y_es = y_train.iloc[fit_pos], y_train.iloc[es_pos]

    pre = build_preprocessor(num_cols, cat_cols, cfg["scaler"], cfg["clip_outliers"], cfg["unknown_strategy"])
    X_fit = pre.fit_transform(X_fit_raw)          # FIT yalnızca burada
    X_es = pre.transform(X_es_raw)
    X_eval = {k: pre.transform(v) for k, v in eval_sets.items()}

    seed_everything(model_seed)
    device = torch.device(TRAINING_SETTINGS["device"])
    model = MLPNet(X_fit.shape[1], cfg["hidden_layers"], cfg["activation"], cfg["dropout"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = make_optimizer(model, cfg)
    sch_cfg = TRAINING_SETTINGS["lr_scheduler"]
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sch_cfg["mode"], factor=sch_cfg["factor"],
        patience=sch_cfg["patience"], min_lr=sch_cfg["min_lr"])
        if cfg.get("use_scheduler", True) else None)

    to_t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)
    X_fit_t, y_fit_t = to_t(X_fit), to_t(y_fit.values)
    X_es_t, y_es_t = to_t(X_es), to_t(y_es.values)

    g = torch.Generator(device="cpu")
    g.manual_seed(model_seed)
    n, bs = X_fit_t.shape[0], int(cfg["batch_size"])

    best_loss, best_epoch, bad_epochs = float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    history, status, warnings_ = [], "max_epochs_reached", []
    stopped_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        nan_hit = False
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_fit_t[idx]), y_fit_t[idx])
            if not torch.isfinite(loss):
                nan_hit = True
                break
            loss.backward()
            optimizer.step()
            tot += loss.item()
            nb += 1
        if nan_hit:
            status = "nan_loss"
            warnings_.append(f"epoch {epoch}: NaN/Inf kayıp")
            stopped_epoch = epoch
            break

        model.eval()
        with torch.no_grad():
            es_loss = float(criterion(model(X_es_t), y_es_t).item())
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": tot / max(nb, 1),
                        "es_val_loss": es_loss, "lr": lr_now})
        stopped_epoch = epoch

        if scheduler is not None:
            scheduler.step(es_loss)

        if es_loss < best_loss - es_cfg["min_delta"]:
            best_loss, best_epoch, bad_epochs = es_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= es_cfg["patience"]:
                status = "early_stopped"
                break

    if best_epoch > 0:
        model.load_state_dict(best_state)
    if best_epoch <= 2 and status != "nan_loss":
        warnings_.append(f"en iyi epoch = {best_epoch} (çok erken; hızlı aşırı öğrenme veya yüksek LR olabilir)")
    if status == "max_epochs_reached":
        warnings_.append("150 epoch üst sınırına ulaşıldı (yakınsama tamamlanmamış olabilir)")
    n_lr_drops = int(sum(1 for i in range(1, len(history)) if history[i]["lr"] < history[i - 1]["lr"]))

    model.eval()
    probs = {}
    with torch.no_grad():
        for k, Xa in X_eval.items():
            probs[k] = torch.sigmoid(model(to_t(Xa))).cpu().numpy()

    return {
        "probs": probs, "history": history, "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch, "best_es_loss": best_loss, "status": status,
        "warnings": "; ".join(warnings_), "n_lr_drops": n_lr_drops,
        "final_lr": history[-1]["lr"] if history else cfg["lr"],
        "n_params": model.count_parameters(), "input_dim": int(X_fit.shape[1]),
        "n_fit": int(len(y_fit)), "n_es": int(len(y_es)),
        "n_es_pos": int(y_es.sum()), "elapsed_sec": time.time() - t0,
        "model": model, "preprocessor": pre,
    }


def history_frame(result, **tags) -> pd.DataFrame:
    """Eğitim geçmişini etiketlerle birlikte DataFrame'e çevirir (CSV loglama için)."""
    h = pd.DataFrame(result["history"])
    for k, v in tags.items():
        h[k] = v
    return h


# ---------------------------------------------------------------------------
# 5. Metrikler
# ---------------------------------------------------------------------------
METRIC_COLS = ["AP", "ROC_AUC", "F1", "Sensitivity", "Specificity",
               "Balanced_Accuracy", "Accuracy", "Precision", "Brier", "PR_AUC_trapz"]


def compute_metrics(y_true, y_prob, threshold=0.50) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AP": float(average_precision_score(y_true, y_prob)),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        # Trapez PR-AUC: AP ile AYNI şey değildir (Bölüm 11), ayrı raporlanır
        "PR_AUC_trapz": float(auc(*precision_recall_curve(y_true, y_prob)[1::-1])),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "threshold": float(threshold),
    }


def mean_sd_table(df, group_col, metrics=METRIC_COLS) -> pd.DataFrame:
    """Kat sonuçlarından Mean ± örnek SD (ddof=1) tablosu üretir."""
    rows = []
    for g, sub in df.groupby(group_col, sort=False):
        row = {group_col: g, "n": len(sub)}
        for m in metrics:
            vals = sub[m].astype(float).values
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            row[f"{m}_mean"], row[f"{m}_sd"] = mu, sd
            row[m] = f"{mu:.4f} ± {sd:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(y_true, y_prob, n_boot=1000, seed=0, threshold=0.50, alpha=0.05):
    """Yüzdelik bootstrap %95 GA (AP, ROC-AUC, F1). Tek sınıf içeren yeniden
    örneklemeler REDDEDİLİR ve yeniden çekilir (Bölüm 13.1)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = {"AP": [], "ROC_AUC": [], "F1": []}
    rejected = 0
    while len(vals["AP"]) < n_boot:
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            rejected += 1
            continue
        yp = y_prob[idx]
        vals["AP"].append(average_precision_score(yt, yp))
        vals["ROC_AUC"].append(roc_auc_score(yt, yp))
        vals["F1"].append(f1_score(yt, (yp >= threshold).astype(int), zero_division=0))
    point = compute_metrics(y_true, y_prob, threshold)
    out = {}
    for m, v in vals.items():
        v = np.asarray(v)
        out[m] = {"point": point[m],
                  "ci_low": float(np.percentile(v, 100 * alpha / 2)),
                  "ci_high": float(np.percentile(v, 100 * (1 - alpha / 2))),
                  "boot_mean": float(v.mean()), "boot_sd": float(v.std(ddof=1))}
    return out, rejected


def f1_optimal_threshold(y_true, y_prob) -> float:
    """İç CV (out-of-fold) tahminlerinde F1'i en yüksek yapan eşik. YALNIZCA development
    tahminleriyle çağrılır; test verisiyle asla."""
    prec, rec, thr = precision_recall_curve(np.asarray(y_true).astype(int), np.asarray(y_prob, float))
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.nanargmax(f1))])


# ---------------------------------------------------------------------------
# 6. Hiperparametre adayları
# ---------------------------------------------------------------------------
_M_COMMON = {"activation": "ReLU", "batch_size": 64, "dropout": 0.0, "weight_decay": 1e-4,
             "clip_outliers": False, "unknown_strategy": "keep_category"}


def _m(cid, purpose, **kw):
    base = {"config_id": cid, "search_type": "manual", "optimizer": "AdamW", "hidden_layers": [32],
            "lr": 1e-3, "scaler": "StandardScaler", "use_scheduler": True, **_M_COMMON}
    base.update(kw)
    base["purpose"] = purpose
    return base


MANUAL_CONFIGS = [
    _m("M1", "Temel manuel konfigürasyon"),
    _m("M2", "Optimizer etkisi (AdamW → Adam)", optimizer="Adam"),
    _m("M3", "Optimizer etkisi (AdamW → SGD+Momentum 0.9)", optimizer="SGD"),
    _m("M4", "Ölçekleme etkisi (Standard → MinMax)", scaler="MinMaxScaler"),
    _m("M5", "Mimari etkisi ((32,) → (64,32))", hidden_layers=[64, 32]),
    _m("M6", "LR scheduler etkisi (scheduler kapalı)", use_scheduler=False),
]

SEARCH_SPACE = {
    "optimizer": ["SGD", "Adam", "AdamW"],
    "candidates_per_optimizer": 2,
    "hidden_layers": [[16], [32], [64], [64, 32], [128, 64]],
    "activation": ["ReLU", "tanh"],
    "lr_log_uniform": {"Adam": [1e-4, 1e-2], "AdamW": [1e-4, 1e-2], "SGD": [1e-3, 1e-1]},
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "batch_size": [32, 64, 128],
    "scaler": ["StandardScaler", "MinMaxScaler", "RobustScaler"],
    "dropout": [0.0, 0.2],
    "clip_outliers": [False, True],
    "unknown_strategy": ["keep_category", "most_frequent"],
    "use_scheduler": [True],
}


def make_random_configs(seed: int) -> list:
    """Tekrarlanabilir Random Search: her optimizer için tam 2 aday (toplam 6).
    LR log-uniform örneklenir; SGD için daha geniş aralık (Bölüm 9–10)."""
    rng = np.random.default_rng(seed)
    S = SEARCH_SPACE
    configs, k = [], 1
    for opt in S["optimizer"]:
        for _ in range(S["candidates_per_optimizer"]):
            lo, hi = S["lr_log_uniform"][opt]
            lr = float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
            configs.append({
                "config_id": f"R{k}", "search_type": "random", "optimizer": opt,
                "hidden_layers": list(S["hidden_layers"][int(rng.integers(len(S["hidden_layers"])))]),
                "activation": str(rng.choice(S["activation"])),
                "lr": round(lr, 6),
                "weight_decay": float(rng.choice(S["weight_decay"])),
                "batch_size": int(rng.choice(S["batch_size"])),
                "scaler": str(rng.choice(S["scaler"])),
                "dropout": float(rng.choice(S["dropout"])),
                "use_scheduler": True,
                "clip_outliers": bool(rng.choice(S["clip_outliers"])),
                "unknown_strategy": str(rng.choice(S["unknown_strategy"])),
                "purpose": "Random Search adayı",
            })
            k += 1
    return configs


def config_label(cfg) -> str:
    return (f"{cfg['optimizer']}, {tuple(cfg['hidden_layers'])}, {cfg['activation']}, "
            f"lr={cfg['lr']:g}, wd={cfg['weight_decay']:g}, bs={cfg['batch_size']}, "
            f"{cfg['scaler'].replace('Scaler', '')}, do={cfg['dropout']}, "
            f"clip={'E' if cfg['clip_outliers'] else 'H'}, unk={cfg['unknown_strategy']}, "
            f"sch={'açık' if cfg.get('use_scheduler', True) else 'kapalı'}")
