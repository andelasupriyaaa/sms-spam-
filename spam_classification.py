import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)

warnings.filterwarnings("ignore")

# Always work in the folder that contains this script (so spam.csv is found in VS Code)
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except NameError:          # running inside a notebook
    pass

SHOW_PLOTS = True   # False = only save figures to ./figures (no pop-up windows)

SEED = 42
np.random.seed(SEED)
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

HAM_COLOR, SPAM_COLOR = "#2E86AB", "#D1495B"


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# ----------------------------------------------------------------------
# 1. LOAD DATA  (Slide 3)
# ----------------------------------------------------------------------
def load_data(path=None):
    candidates = [path] if path else ["spam.csv", "SMSSpamCollection",
                                       "SMSSpamCollection.txt", "data/spam.csv"]
    for p in candidates:
        if p and os.path.exists(p):
            if p.lower().endswith(".csv"):
                df = pd.read_csv(p, encoding="latin-1")
                df = df.iloc[:, :2]
            else:
                df = pd.read_csv(p, sep="\t", header=None, encoding="latin-1")
            df.columns = ["label", "text"]
            print(f"Loaded dataset from: {p}")
            return df
    # Not found locally: try to download it from the UCI repository
    try:
        import io, urllib.request, zipfile
        print("Dataset not found locally. Downloading from UCI ...")
        url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        data = urllib.request.urlopen(url, timeout=30).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            with z.open("SMSSpamCollection") as f:
                raw = f.read().decode("latin-1")
        with open("SMSSpamCollection", "w", encoding="latin-1") as f:
            f.write(raw)
        df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, encoding="latin-1")
        df.columns = ["label", "text"]
        return df
    except Exception as e:
        sys.exit(f"Dataset not found and download failed ({e}).\n"
                 "Download the UCI SMS Spam Collection manually and place 'spam.csv' "
                 "(or 'SMSSpamCollection') in the same folder as this script.")


# Works in VS Code / terminal (optional path argument) and in Jupyter
# (where sys.argv holds kernel arguments such as "-f", which we ignore).
_arg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
df = load_data(_arg)
df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label"].isin(["ham", "spam"])].dropna(subset=["text"]).reset_index(drop=True)

print("\n=== DATASET OVERVIEW ===")
print(f"Messages        : {len(df):,}")
print(f"Missing values  : {int(df.isnull().sum().sum())}")
counts = df["label"].value_counts()
print(f"Ham             : {counts['ham']:,} ({counts['ham'] / len(df):.1%})")
print(f"Spam            : {counts['spam']:,} ({counts['spam'] / len(df):.1%})")
print(f"Duplicate texts : {int(df.duplicated(subset='text', keep=False).sum())} "
      f"messages appear more than once (kept; see limitations)")
print(f"'Always Ham' baseline accuracy: {counts['ham'] / len(df):.1%}")

# Encode labels: Ham = 0, Spam = 1
df["y"] = (df["label"] == "spam").astype(int)

# ----------------------------------------------------------------------
# 2. EXPLORATORY ANALYSIS  (Slides 3 and 4)
# ----------------------------------------------------------------------
# Class balance chart
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["Ham", "Spam"], [counts["ham"], counts["spam"]],
              color=[HAM_COLOR, SPAM_COLOR])
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 50,
            f"{int(b.get_height()):,}\n({b.get_height() / len(df):.1%})",
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Number of messages")
ax.set_title("Class balance")
ax.set_ylim(0, counts.max() * 1.18)
savefig("01_class_balance.png")

# Message length histogram
df["length"] = df["text"].str.len()
print(f"\nAverage length: spam = {df.loc[df.y == 1, 'length'].mean():.0f} chars, "
      f"ham = {df.loc[df.y == 0, 'length'].mean():.0f} chars")

fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 400, 50)
ax.hist(df.loc[df.y == 0, "length"], bins=bins, alpha=0.7, color=HAM_COLOR,
        label="Ham", density=True)
ax.hist(df.loc[df.y == 1, "length"], bins=bins, alpha=0.7, color=SPAM_COLOR,
        label="Spam", density=True)
ax.set_xlabel("Message length (characters)")
ax.set_ylabel("Density")
ax.set_title("Message length: spam is longer than ham")
ax.legend()
savefig("02_length_histogram.png")

# ----------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT  (Slide 5)
#    Stratified 80/20 so the ~13.4% spam share is kept in both sets.
#    The vectoriser is fitted on the TRAINING set only (no leakage).
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["y"], test_size=0.20, stratify=df["y"], random_state=SEED
)
print("\n=== SPLIT ===")
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Spam share  train: {y_train.mean():.1%} | test: {y_test.mean():.1%}")


def make_vectorizer(ngram_range=(1, 2), stop_words=None):
    """TF-IDF: lowercase, 1-2 word grams, sublinear TF, min_df = 2, stop words kept."""
    return TfidfVectorizer(lowercase=True, ngram_range=ngram_range, min_df=2,
                           sublinear_tf=True, stop_words=stop_words)


vectorizer = make_vectorizer()
Xtr = vectorizer.fit_transform(X_train)     # fit on train only
Xte = vectorizer.transform(X_test)          # transform only
print(f"TF-IDF features (fitted on train): {Xtr.shape[1]:,}")

# Top words that signal spam / ham (from a logistic regression, as on slide 4)
lr_probe = LogisticRegression(C=10, max_iter=1000, random_state=SEED).fit(Xtr, y_train)
terms = np.array(vectorizer.get_feature_names_out())
order = np.argsort(lr_probe.coef_[0])
top_ham, top_spam = order[:12], order[::-1][:12]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].barh(terms[top_spam][::-1], lr_probe.coef_[0][top_spam][::-1], color=SPAM_COLOR)
axes[0].set_title("Words that signal SPAM")
axes[0].set_xlabel("Logistic regression weight")
axes[1].barh(terms[top_ham][::-1], -lr_probe.coef_[0][top_ham][::-1], color=HAM_COLOR)
axes[1].set_title("Words that signal HAM")
axes[1].set_xlabel("Logistic regression weight (absolute)")
savefig("03_top_words.png")
print("Top spam terms:", ", ".join(terms[top_spam][:8]))
print("Top ham terms :", ", ".join(terms[top_ham][:8]))

# ----------------------------------------------------------------------
# 4. PARAMETER EXPERIMENTS  (Slide 7)
#    5-fold stratified CV on the TRAINING set only; metric = mean Spam F1.
#    Using a Pipeline so the vectoriser is re-fitted inside every fold.
# ----------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def cv_f1(clf, ngram_range=(1, 2), stop_words=None):
    pipe = Pipeline([("tfidf", make_vectorizer(ngram_range, stop_words)), ("clf", clf)])
    return cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()


print("\n=== PARAMETER EXPERIMENTS (CV spam F1, training set) ===")

nb_alphas = [0.01, 0.1, 0.5, 1.0]
nb_scores = [cv_f1(MultinomialNB(alpha=a)) for a in nb_alphas]
print("Naive Bayes alpha :", {a: round(s, 3) for a, s in zip(nb_alphas, nb_scores)})

svm_cs = [0.1, 1, 10]
svm_scores = [cv_f1(LinearSVC(C=c, random_state=SEED)) for c in svm_cs]
print("SVM C             :", {c: round(s, 3) for c, s in zip(svm_cs, svm_scores)})

ngram_opts = {"1 only": (1, 1), "1 to 2": (1, 2), "1 to 3": (1, 3)}
ng_scores = [cv_f1(LinearSVC(C=1, random_state=SEED), ngram_range=r) for r in ngram_opts.values()]
print("N-gram range      :", {k: round(s, 3) for k, s in zip(ngram_opts, ng_scores)})

sw_opts = {"Removed": "english", "Kept": None}
sw_scores = [cv_f1(LinearSVC(C=1, random_state=SEED), stop_words=v) for v in sw_opts.values()]
print("Stop words        :", {k: round(s, 3) for k, s in zip(sw_opts, sw_scores)})

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
panels = [
    (axes[0, 0], [str(a) for a in nb_alphas], nb_scores, "Naive Bayes alpha"),
    (axes[0, 1], [str(c) for c in svm_cs], svm_scores, "SVM C"),
    (axes[1, 0], list(ngram_opts), ng_scores, "N-gram range"),
    (axes[1, 1], list(sw_opts), sw_scores, "Stop words"),
]
for ax, labels, scores, title in panels:
    colors = ["#BBBBBB"] * len(scores)
    colors[int(np.argmax(scores))] = SPAM_COLOR          # highlight best row
    ax.bar(labels, scores, color=colors)
    for i, s in enumerate(scores):
        ax.text(i, s + 0.005, f"{s:.3f}", ha="center", fontsize=9)
    ax.set_ylim(min(scores) - 0.05, 1.0)
    ax.set_title(title)
    ax.set_ylabel("CV F1 (Spam)")
fig.suptitle("Parameter experiments (5-fold CV on the training set)", fontsize=13)
savefig("04_parameter_experiments.png")

# ----------------------------------------------------------------------
# 5. COMPARE FIVE CLASSIFIERS  (Slides 6 and 8)
#    Same split, same features, fixed seed (42).
# ----------------------------------------------------------------------
models = {
    "Naive Bayes":         MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(C=10, max_iter=1000, random_state=SEED),
    "Linear SVM":          LinearSVC(C=1, random_state=SEED),
    "Random Forest":       RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "MLP (64)":            MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam",
                                         early_stopping=True, random_state=SEED, max_iter=300),
}

rows, preds = [], {}
for name, model in models.items():
    model.fit(Xtr, y_train)
    p = model.predict(Xte)
    preds[name] = p
    rows.append({
        "Model": name,
        "Accuracy":  accuracy_score(y_test, p),
        "Precision": precision_score(y_test, p),
        "Recall":    recall_score(y_test, p),
        "F1":        f1_score(y_test, p),
    })

results = pd.DataFrame(rows).set_index("Model")
print("\n=== MODEL COMPARISON ON THE TEST SET (Spam = positive class) ===")
print(results.round(4).to_string())
results.round(4).to_csv("model_comparison.csv")

# Grouped bar chart: precision / recall / F1
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(results))
w = 0.26
for i, (metric, col) in enumerate([("Precision", "#2E86AB"), ("Recall", "#F6AE2D"), ("F1", "#D1495B")]):
    b = ax.bar(x + (i - 1) * w, results[metric], w, label=metric, color=col)
    for rect in b:
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.004,
                f"{rect.get_height():.3f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(results.index)
ax.set_ylim(0.6, 1.05)          # truncated axis, as noted on slide 8
ax.set_ylabel("Score")
ax.set_title("Spam class: precision, recall and F1 (test set, axis starts at 0.6)")
ax.legend(loc="lower left")
savefig("05_model_comparison.png")

# Overall accuracy chart
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(results.index, results["Accuracy"], color="#6C757D")
ax.axhline(counts["ham"] / len(df), color=SPAM_COLOR, ls="--",
           label=f"'Always Ham' baseline ({counts['ham'] / len(df):.1%})")
for i, a in enumerate(results["Accuracy"]):
    ax.text(i, a + 0.001, f"{a:.3f}", ha="center", fontsize=9)
ax.set_ylim(0.84, 1.0)
ax.set_ylabel("Accuracy")
ax.set_title("Test accuracy vs majority-class baseline")
ax.legend(loc="lower right")
plt.xticks(rotation=15)
savefig("06_accuracy_vs_baseline.png")

# Confusion matrices for all five models
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, (name, p) in zip(axes, preds.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, p),
                           display_labels=["Ham", "Spam"]).plot(ax=ax, cmap="Blues",
                                                               colorbar=False, values_format="d")
    ax.set_title(name)
savefig("07_all_confusion_matrices.png")

# ----------------------------------------------------------------------
# 6. FINAL MODEL: TF-IDF (1-2 grams) + LINEAR SVM, C = 1  (Slide 9)
# ----------------------------------------------------------------------
final_pipeline = Pipeline([
    ("tfidf", make_vectorizer((1, 2), None)),
    ("svm", LinearSVC(C=1, random_state=SEED)),
])
final_pipeline.fit(X_train, y_train)          # tested ONCE on the unseen test set
y_pred = final_pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n=== FINAL MODEL: Linear SVM (tested once on unseen messages) ===")
print(f"Accuracy          : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (Spam)  : {precision_score(y_test, y_pred):.4f}")
print(f"Recall (Spam)     : {recall_score(y_test, y_pred):.4f}")
print(f"F1 (Spam)         : {f1_score(y_test, y_pred):.4f}")
print(f"Ham wrongly blocked: {fp} of {tn + fp}")
print(f"Spam missed        : {fn} of {tp + fn}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"], digits=4))

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"]).plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion matrix: TF-IDF + Linear SVM")
savefig("08_final_confusion_matrix.png")

# ----------------------------------------------------------------------
# 7. ANALYSIS AND LIMITATIONS  (Slide 10)
# ----------------------------------------------------------------------
print("\n=== ANALYSIS ===")
print(f"Accuracy {accuracy_score(y_test, y_pred):.1%} vs {counts['ham'] / len(df):.1%} "
      f"for always predicting Ham")
print(f"Spam caught: {tp} / {tp + fn}  (recall {tp / (tp + fn):.1%})")
print(f"Legitimate messages wrongly blocked: {fp} / {tn + fp}")

test_df = pd.DataFrame({"text": X_test.values, "actual": y_test.values, "pred": y_pred})
missed_spam = test_df[(test_df.actual == 1) & (test_df.pred == 0)]
false_alarms = test_df[(test_df.actual == 0) & (test_df.pred == 1)]

print(f"\n--- {len(missed_spam)} spam messages that slipped through (first 5) ---")
for t in missed_spam["text"].head(5):
    print("  *", t[:140])
print(f"\n--- {len(false_alarms)} legitimate messages wrongly flagged ---")
for t in false_alarms["text"]:
    print("  *", t[:140])

# Overlap of duplicate messages between train and test (limitation)
overlap = X_test.isin(set(X_train)).sum()
print(f"\nTest messages that also appear verbatim in training data: {overlap} "
      f"({overlap / len(X_test):.1%})  -> may inflate scores slightly")

# Extra: decision-threshold tuning on the SVM score (Future work)
scores = final_pipeline.decision_function(X_test)
thresholds = np.linspace(-1.0, 1.0, 81)
prec = [precision_score(y_test, scores > t, zero_division=0) for t in thresholds]
rec = [recall_score(y_test, scores > t) for t in thresholds]
f1s = [f1_score(y_test, scores > t) for t in thresholds]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(thresholds, prec, label="Precision", color="#2E86AB")
ax.plot(thresholds, rec, label="Recall", color="#F6AE2D")
ax.plot(thresholds, f1s, label="F1", color=SPAM_COLOR)
ax.axvline(0, color="grey", ls="--", label="Default threshold (0)")
ax.set_xlabel("Decision threshold on SVM score")
ax.set_ylabel("Score")
ax.set_title("Threshold tuning: trade-off between false alarms and missed spam")
ax.legend()
savefig("09_threshold_tuning.png")

# ----------------------------------------------------------------------
# 8. PREDICT ON NEW MESSAGES
# ----------------------------------------------------------------------
def classify(messages):
    """Return a DataFrame with the predicted label and SVM confidence score."""
    if isinstance(messages, str):
        messages = [messages]
    s = final_pipeline.decision_function(messages)
    out = pd.DataFrame({
        "message": messages,
        "prediction": np.where(s > 0, "SPAM", "HAM"),
        "score": np.round(s, 3),
    })
    return out


samples = [
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward. Call 09061701461 to claim.",
    "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121 to receive entry question",
    "URGENT! Your mobile number has won £2000 bonus caller prize. Reply CLAIM now, 150p/msg",
    "Ok, I'll be home by 7. Can you pick up some milk on the way?",
    "Hey, are we still meeting for lunch tomorrow?",
    "Can you send me the notes from today's class? Thanks!",
]
print("\n=== PREDICTIONS ON NEW MESSAGES ===")
print(classify(samples).to_string(index=False))

# Plot of the sample predictions
sample_df = classify(samples)
fig, ax = plt.subplots(figsize=(9, 4))
colors = [SPAM_COLOR if p == "SPAM" else HAM_COLOR for p in sample_df["prediction"]]
ax.barh([m[:45] + "..." for m in sample_df["message"]][::-1], sample_df["score"][::-1], color=colors[::-1])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("SVM decision score  (> 0 = Spam, < 0 = Ham)")
ax.set_title("Predictions on new messages")
savefig("10_sample_predictions.png")

print(f"\nDone. Figures saved in ./{FIG_DIR}/ and metrics in model_comparison.csv")