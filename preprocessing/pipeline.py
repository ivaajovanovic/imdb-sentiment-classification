import pandas as pd
from steps.html_removal import remove_html

df = pd.read_csv('data/IMDB Dataset.csv')

print("BEFORE:", df['review'][1][:200])
df['cleaned'] = df['review'].apply(remove_html)
print("AFTER:", df['cleaned'][1][:200])