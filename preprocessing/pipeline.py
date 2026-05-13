import pandas as pd
from steps.html_removal import remove_html
from steps.lowercasing import lowercase

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