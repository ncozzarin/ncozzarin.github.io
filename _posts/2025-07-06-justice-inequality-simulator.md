---
title: "Justice Inequality Simulator — A Generative & Probabilistic Approach to Judicial Bias"
date: 2025-07-06
author: "Nicolas Cozzarin (@ncozzarin)"
---

[Share on Twitter](#) | [Share on LinkedIn](#)

## HFU Deep Generative Models — Project Report

### 1. Motivation

Judicial outcomes shape lives, yet historical data show that race, gender, or socioeconomic status sometimes correlate with harsher sentences.  
**Goal**: reveal and quantify those correlations through counterfactual generation — not to automate justice, but to explain bias.

---

### 2. Datasets

| Source                      | Rows     | Key Features                                |
|----------------------------|----------|---------------------------------------------|
| COMPAS recidivism          | ≈ 7,000  | age, priors, race, charge                   |
| U.S. Supreme Court (SCDB)  | ≈ 28,000 | issue, petitioner, lower-court direction    |
| Justice CSV (course data)  | ≈ 5,000  | free-text facts, first_party_winner         |

**Figure 1 – Class imbalance before resampling**  

---

### 3. Pre-processing

Before training the model, raw text data often requires cleaning to remove unwanted HTML tags, punctuation, and extra whitespace. This step ensures the model receives consistent, noise-free input. Additionally, the target variable is converted into a numeric binary label to facilitate supervised learning.


{% highlight python %}
import re, nltk, pandas as pd
TAG_RE = re.compile(r"<[^>]+>")
PUNCT = str.maketrans("", "", r"""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~""")

def clean(html: str) -> str:
"""Strip HTML, punctuation, collapse whitespace."""
return re.sub(r"\s+", " ",
TAG_RE.sub("", html or "").translate(PUNCT)).strip()

df = pd.read_csv("justice.csv").dropna(subset=["facts", "first_party_winner"])
df["text"] = df["facts"].map(clean)
df["label"] = (df["first_party_winner"]
.astype(str).str.lower()
.map({"true":1,"false":0,"1":1,"0":0}))
{% endhighlight %}

💡 Figure 2 – Example cleaned record
(A formatted <blockquote> can be added here when the dataset becomes public.)
---
4. Balancing the data
{% highlight python %}
from sklearn.utils import resample

pos = df[df.label == 1]
neg = df[df.label == 0]

minority = pos if len(pos) < len(neg) else neg
majority = neg if len(pos) < len(neg) else pos

minority_up = resample(minority,
replace=True,
n_samples=len(majority),
random_state=42)

df_bal = pd.concat([majority, minority_up]).reset_index(drop=True)
{% endhighlight %}


Figure 3 – Balanced class histogram

5. Feature Extraction with Legal-BERT
In this step, we convert the raw legal text descriptions (text) into dense numerical representations using Legal-BERT, a version of BERT pre-trained on legal corpora. These embeddings serve as input features to our classifier.

We use the hidden state of the [CLS] token as a compact representation of each text. The [CLS] vector is commonly used in classification tasks because it captures the overall semantics of the input.

Batch processing is applied with GPU acceleration if available. The final output is a matrix of embeddings, one per legal case, which we save to disk as a NumPy .npy file for faster reuse in training.

{% highlight python %}
from transformers import AutoTokenizer, AutoModel
import torch, numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "nlpaueb/legal-bert-base-uncased"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
bert = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

@torch.no_grad()
def embed(txts, batch=16, max_len=256):
vecs = []
for i in range(0, len(txts), batch):
enc = tok(txts[i:i+batch],
padding=True, truncation=True,
max_length=max_len,
return_tensors="pt").to(DEVICE)
h = bert(**enc).last_hidden_state[:, 0] # CLS
vecs.append(h.cpu())
return torch.cat(vecs).numpy()

X = embed(df_bal.text.tolist())
np.save("legal_cls.npy", X) # cache
{% endhighlight %}


Figure 4 – t-SNE visualization of [CLS] embeddings colored by class label



