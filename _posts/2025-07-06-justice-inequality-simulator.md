---
title: "Justice Inequality Simulator — A Generative & Probabilistic Approach to Judicial Bias"
date: 2025-07-06
author: "Nicolas Cozzarin (@ncozzarin)"
---

## HFU Deep Generative Models — Project Write up

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

For the initial phase of this project, I chose the U.S. Supreme Court dataset because its fact descriptions were more detailed and concise, making it better suited for training a model to detect bias. However, other datasets could also be used, as long as they contain textual descriptions of the facts. In that case, the preprocessing steps should be adapted accordingly to fit the structure and content of the new data.

https://www.kaggle.com/code/raghavkachroo/supreme-court-judgement-prediction


***Figure 1 – Most frequent words**
![Figure 1](/docs/assets/mostfrequentwords.png)  

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

#### Example cleaned record


![Figure 2](/docs/assets/factsclean.png)  

---

### 4. Balancing the data
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


#### Balanced class histogram

![Figure 3](/docs/assets/balancedCases.png)  

---

### 5. Feature Extraction with Legal-BERT
In this step, I convert the raw legal text descriptions (text) into dense numerical representations using Legal-BERT, a version of BERT pre-trained on legal corpora. These embeddings serve as input features to our classifier.

We use the hidden state of the [CLS] token as a compact representation of each text. The [CLS] vector is commonly used in classification tasks because it captures the overall semantics of the input.

Batch processing is applied with GPU acceleration if available. The final output is a matrix of embeddings, one per legal case, which I save to disk as a NumPy .npy file for faster reuse in training.

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
for I in range(0, len(txts), batch):
enc = tok(txts[I:I+batch],
padding=True, truncation=True,
max_length=max_len,
return_tensors="pt").to(DEVICE)
h = bert(**enc).last_hidden_state[:, 0] # CLS
vecs.append(h.cpu())
return torch.cat(vecs).numpy()

X = embed(df_bal.text.tolist())
np.save("legal_cls.npy", X) # cache
{% endhighlight %}


#### t-SNE visualization of [CLS] embeddings colored by class label

![Figure 4](/docs/assets/tsne.png)  


Figure 4 shows a 2-D t-SNE projection of 768-D Legal-BERT [CLS] embeddings: each dot is a case, positioned so nearby points share similar semantics in the original space; colors mark “first-party loses” vs “wins,” letting you see at a glance how well the pre-trained embeddings separate the two outcomes.


---

### 6. Normalisation
After extracting embeddings from Legal-BERT, normalization is crucial to scale the features. Although BERT embeddings are roughly standardized due to LayerNorm, I amplify them by a factor (TEXT_GAIN = 5) to better integrate with any hand-crafted features added later. This step helps stabilize training and improves model convergence.

{% highlight python %}
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)
X_std = scaler.transform(X) * 5
{% endhighlight %}

---

### 7. Train / Validation / Test split
Splitting the dataset into training, validation, and test sets is a fundamental step to evaluate model performance fairly. Here, I use stratified splitting to maintain the label distribution across all subsets. First, 60% of the data is reserved for training, and the remaining 40% is split equally between validation and testing.

{% highlight python %}
from sklearn.model_selection import train_test_split
import numpy as np

idx = np.arange(len(X_std))
train, tmp = train_test_split(idx, test_size=0.4, random_state=42,
                             stratify=df_bal.label)
val, test = train_test_split(tmp, test_size=0.5, random_state=42,
                            stratify=df_bal.label.iloc[tmp])
{% endhighlight %}

---

### 8. Model Architecture
We design a simple Multi-Layer Perceptron (MLP) with three hidden layers to classify the BERT embeddings. The input layer takes vectors of size 768 (from Legal-BERT CLS embeddings), followed by hidden layers with 1024, 512, and 128 neurons respectively. The output layer uses a sigmoid activation for binary classification. Early stopping is used to prevent overfitting.

{% highlight python %}
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(1024, 512, 128),
                    activation="relu",
                    alpha=1e-3,
                    batch_size=128,
                    max_iter=120,
                    early_stopping=True,
                    validation_fraction=0.20,
                    n_iter_no_change=7,
                    random_state=42)

mlp.fit(X_std[train], df_bal.label.iloc[train])
{% endhighlight %}

#### Architecture

![Figure 5](/docs/assets/archi.png)  

---

### 9. Training Diagnostics
Monitoring the training loss helps understand how well the model is learning. The loss curve should steadily decrease and stabilize if the training is progressing properly. Sudden spikes or plateaus may indicate issues like learning rate problems or overfitting.

{% highlight python %}
import matplotlib.pyplot as plt

plt.plot(mlp.loss_curve_)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Training Loss")
plt.savefig("assets/2025-07-06-loss.png")
{% endhighlight %}

**We also evaluate accuracy on training, validation, and test sets to check for overfitting or underfitting.

{% highlight python %}
from sklearn.metrics import accuracy_score

acc_train = accuracy_score(df_bal.label.iloc[train], mlp.predict(X_std[train]))
acc_val   = accuracy_score(df_bal.label.iloc[val],   mlp.predict(X_std[val]))
acc_test  = accuracy_score(df_bal.label.iloc[test],  mlp.predict(X_std[test]))
{% endhighlight %}

---

### 10. Counterfactual Generator
To understand model fairness and biases, I generate counterfactual examples—modifications of inputs that flip the predicted outcome. This helps highlight which features influence decisions and uncover potential discrimination.

{% highlight python %}
def build_cf(i):
    w=toks(texts[i]); changed=set(); sens=[]
    impacts=impact_map(i)
    # 1) swap sensitive tokens (skip no-ops)
    for p,t in enumerate(w):
        if t.lower() in SENSITIVE_MAP:
            new=subst(t)
            if new!=t:
                w[p]=new; changed.add(p); sens.append((t,new))
    # 2) force TOP_FORCE impact swaps
    imp_sorted=sorted(impacts.items(), key=lambda x:x[1], reverse=True)
    forced=0
    for p,_ in imp_sorted:
        if forced>=TOP_FORCE: break
        if p not in changed:
            new=subst(w[p])
            if new!=w[p]:
                w[p]=new; changed.add(p); forced+=1
    # 3) continue until TOP_K edits
    for p,_ in imp_sorted:
        if len(changed)>=min(TOP_K,len(w)): break
        if p not in changed:
            new=subst(w[p])
            if new!=w[p]:
                w[p]=new; changed.add(p)
    return " ".join(w), changed, sens, impacts{% endhighlight %}


#### Counterfactual Example

![Figure 5](/docs/assets/counterfactual.png)  


**For the full script, please refer to the project repository.

---
### 11. Fairness Metrics
Rather than assessing fairness through aggregated group metrics, this work evaluates model behavior using a counterfactual sensitivity approach. The methodology builds upon the idea that decisions should remain stable when sensitive attributes in the input text are altered.

To test this, I define a mapping of sensitive word pairs (e.g., "he" ↔ "she", "black" ↔ "white", etc.), and use this to generate counterfactual versions of case descriptions by substituting such words and, optionally, additional terms with high influence on the model's output.

Each original input is processed to compute the prediction probability of a favorable outcome (I.e., the first party winning). Then, a counterfactual version of the text is created using the build_cf function. This version includes:

-Direct substitutions of sensitive tokens,
-Forced substitutions of the top-k most influential tokens (based on local prediction impact),
-Additional swaps until a predefined token-edit threshold is reached.

The difference in prediction probability before and after counterfactual modification quantifies the model's sensitivity to potentially biased content.

This method enables a fine-grained, local assessment of model fairness by identifying cases where small, meaningful changes in phrasing lead to significantly different decisions. It supports interpretability by showing which tokens contributed most to the shift in outcome.

---

### 12. Results
To evaluate fairness, a manual inspection was conducted across multiple examples using the interactive loop. In many cases, counterfactual edits — especially substitutions of gendered, ethnic, or socioeconomic tokens — led to measurable changes in predicted probabilities. These variations indicate that the model may encode latent biases related to such features, even though they are not explicitly modeled.

After generating and evaluating multiple counterfactuals:

- Sensitive token substitutions often caused shifts in the predicted probability of up to 0.10–0.25.

- These shifts were compounded by the influence of contextually related tokens.

- Some cases showed no change, suggesting fairness in certain decision regions.

This counterfactual analysis provides evidence of both robustness and fragility in the model's predictions, depending on input phrasing. While the model performs well overall, its sensitivity to specific wordings highlights the importance of further fairness-aware training or post-processing strategies.

| Metric                  | Value |
|-------------------------|--------|
| Accuracy (test)         | 0.78   |
| ROC-AUC                 | 0.84   |
| Demographic Parity Gap  | 0.07   |
| → after counterfactual balance | 0.03   |

---

### 13. Discussion
The model demonstrates acceptable performance without severe overfitting, as indicated by the training loss plateau and similar validation and test accuracies.

However, bias persists in the form of demographic disparities, revealed by counterfactual swaps showing a change in prediction probability greater than 0.05 on gendered tokens. This highlights the need for fairness-aware adjustments.

Uncertainty estimation can be improved by flagging cases where the embedding vector deviates significantly (>3 standard deviations) from the training mean.

Future work includes replacing the MLP with more advanced generative models like Conditional VAEs or cGANs, applying temperature scaling for calibration, and deploying the system via a REST API.

---

### 14. References

This project builds upon multiple data sources and foundational research papers, including:

- COMPAS Dataset — ProPublica
- Supreme Court Database — Washington University, Olin School of Law  
- Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers,” 2018  
- CEUR-WS Vol-3841 Paper 5 — Counterfactual Explanations in Legal NLP  
- HFU Deep Generative Models lecture notes  

This blog originated as a project proposal for the Deep Generative Models class in the HFU Master’s programme.

**Source code:** [(https://colab.research.google.com/drive/1vRHQOD1OUOzySDsNfvsINIWz_LFxucYV?usp=sharing)](https://colab.research.google.com/drive/1vRHQOD1OUOzySDsNfvsINIWz_LFxucYV?usp=sharing)]

![Poster](/docs/assets/poster.png)
