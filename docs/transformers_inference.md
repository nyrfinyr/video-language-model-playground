# Inferenza con Hugging Face Transformers

## Installazione

```bash
pip install transformers torch accelerate
```

---

## Caricamento del modello

Il punto di ingresso principale è `AutoModelForCausalLM` + `AutoTokenizer`. Usare `from_pretrained` con il nome del modello su HuggingFace Hub oppure un path locale.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # o qualsiasi altro modello

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,   # riduce memoria GPU
    device_map="auto",           # distribuisce automaticamente su GPU/CPU
)
```

### Opzioni principali di `from_pretrained`

| Parametro | Descrizione |
|---|---|
| `torch_dtype` | Precisione dei pesi: `float32`, `float16`, `bfloat16` |
| `device_map` | `"auto"`, `"cuda"`, `"cpu"`, o mappa manuale per layer |
| `load_in_8bit` | Quantizzazione 8-bit (richiede `bitsandbytes`) |
| `load_in_4bit` | Quantizzazione 4-bit (richiede `bitsandbytes`) |
| `trust_remote_code` | Necessario per alcuni modelli con codice custom |

---

## Inferenza base

```python
prompt = "Spiega la differenza tra supervised e unsupervised learning."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

## Parametri di generazione

I parametri si passano direttamente a `model.generate()` oppure tramite un oggetto `GenerationConfig`.

### Lunghezza output

```python
model.generate(
    **inputs,
    max_new_tokens=512,    # token massimi da generare (preferibile a max_length)
    min_new_tokens=10,     # forza almeno N token
)
```

### Sampling (testo creativo / vario)

```python
model.generate(
    **inputs,
    do_sample=True,
    temperature=0.7,       # < 1 più deterministico, > 1 più creativo
    top_p=0.9,             # nucleus sampling: considera i token con prob cumulativa <= top_p
    top_k=50,              # considera solo i top-k token più probabili
)
```

### Greedy / Beam search (testo preciso / riassunti)

```python
# Greedy (default, do_sample=False)
model.generate(**inputs, do_sample=False)

# Beam search
model.generate(
    **inputs,
    num_beams=4,
    early_stopping=True,
)
```

### Penalità per evitare ripetizioni

```python
model.generate(
    **inputs,
    repetition_penalty=1.2,    # > 1 penalizza token già generati
    no_repeat_ngram_size=3,    # vieta di ripetere n-gram di dimensione N
)
```

### Tabella riassuntiva parametri

| Parametro | Range tipico | Effetto |
|---|---|---|
| `temperature` | 0.1 – 1.5 | Creatività vs determinismo |
| `top_p` | 0.7 – 1.0 | Diversità del vocabolario usato |
| `top_k` | 10 – 100 | Limita il pool di token candidati |
| `num_beams` | 2 – 8 | Qualità (beam search), più lento |
| `repetition_penalty` | 1.0 – 1.5 | Riduce ripetizioni |
| `max_new_tokens` | libero | Limite lunghezza risposta |

---

## Utilizzo di `GenerationConfig`

Permette di salvare e riutilizzare la configurazione.

```python
from transformers import GenerationConfig

gen_config = GenerationConfig(
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    max_new_tokens=300,
    repetition_penalty=1.1,
)

outputs = model.generate(**inputs, generation_config=gen_config)
```

---

## Pipeline (API ad alto livello)

Per un utilizzo rapido senza gestire tokenizer e modello manualmente:

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

result = pipe(
    prompt,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
)
print(result[0]["generated_text"])
```

---

## Modelli chat (con template)

I modelli instruct/chat richiedono un formato specifico per il prompt.

```python
messages = [
    {"role": "system", "content": "Sei un assistente utile."},
    {"role": "user", "content": "Cos'è il gradient descent?"},
]

# Applica il template del modello
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Considerazioni su memoria GPU

| Precisione | VRAM approssimativa (7B params) |
|---|---|
| `float32` | ~28 GB |
| `float16` / `bfloat16` | ~14 GB |
| 8-bit (`load_in_8bit`) | ~8 GB |
| 4-bit (`load_in_4bit`) | ~4 GB |

Per la quantizzazione 4/8-bit:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
```
