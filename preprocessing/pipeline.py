import pandas as pd
import time
from steps.html_removal import remove_html
from steps.lowercasing import lowercase
from steps.tokenization import tokenize
#from steps.tokenization_spacy import tokenize
from steps.punctuation import remove_punctuation
from steps.stopwords import remove_stopwords





df = pd.read_csv('data/IMDB Dataset.csv')

# ── HTML REMOVAL ──────────────────────────────
print("=== HTML REMOVAL ===")
print("BEFORE:", df['review'][1][:200])
df['cleaned'] = df['review'].apply(remove_html)
print("AFTER: ", df['cleaned'][1][:200])

# ── LOWERCASING ───────────────────────────────
print("\n=== LOWERCASING ===")
print("BEFORE:", df['cleaned'][1][:200])
df['cleaned'] = df['cleaned'].apply(lowercase)
print("AFTER: ", df['cleaned'][1][:200])


# ── TOKENIZATION SPACY ────────────────────────
#print("\n=== TOKENIZATION SPACY ===")
#print("BEFORE:", df['cleaned'][1][:200])
#start = time.time()
#df['cleaned_spacy'] = df['cleaned'].apply(tokenize)
#print(f"Time: {time.time() - start:.2f}s")
#print("AFTER: ", df['cleaned_spacy'][1][:10])

# ── TOKENIZATION ──────────────────────────────
print("\n=== TOKENIZATION ===")
print("BEFORE:", df['cleaned'][1][:200])
start = time.time()
df['cleaned'] = df['cleaned'].apply(tokenize)
print(f"Time: {time.time() - start:.2f}s")
print("AFTER: ", df['cleaned'][1][:10])



# ── PUNCTUATION REMOVAL ───────────────────────
print("\n=== PUNCTUATION REMOVAL ===")
print("BEFORE:", df['cleaned'][1][:10])
df['cleaned'] = df['cleaned'].apply(remove_punctuation)
print("AFTER: ", df['cleaned'][1][:10])



# ── STOP WORD REMOVAL ─────────────────────────
print("\n=== STOP WORD REMOVAL ===")
print("BEFORE:", df['cleaned'][1][:10])
df['cleaned'] = df['cleaned'].apply(remove_stopwords)
print("AFTER: ", df['cleaned'][1][:10])

