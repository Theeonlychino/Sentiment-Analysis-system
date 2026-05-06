import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# 1. Dataset with 3 categories: 1=Pos, 0=Neg, 2=Neutral
data = {
    'text': [
        'I love this, it is amazing', 'Very good service', 'Happy and joyful', # Positive
        'I hate this, it is terrible', 'Worst experience ever', 'Very bad and sad', # Negative
        'This is a book', 'The weather is normal', 'I am eating rice', 'It is a house' # Neutral
    ],
    'label': [1, 1, 1, 0, 0, 0, 2, 2, 2, 2] 
}
df = pd.DataFrame(data)

# 2. Vectorize
vec = TfidfVectorizer()
X = vec.fit_transform(df['text'])
y = df['label']

# 3. Train
model = MultinomialNB()
model.fit(X, y)

# 4. Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vec, f)

print("Training complete! 3-way brain (Pos/Neg/Neu) created.")





