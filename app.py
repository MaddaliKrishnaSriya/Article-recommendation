import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv('Article.csv')

df['combined_text'] = df['title'] + ' ' + df['text'] + ' ' + df['summary'] + ' ' + df['keywords'].apply(lambda x: ' '.join(x))

tfidf_vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim, df=df):
    try:
        idx = df[df['title'] == title].index[0]  
    except IndexError:
        print(f"Error: Title '{title}' not found in the dataset.")
        return pd.Series([])  

    sim_scores = list(enumerate(cosine_sim[idx]))  
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  
    sim_scores = sim_scores[1:11] 
    title_indices = [i[0] for i in sim_scores]  
    return df['title'].iloc[title_indices]

st.title('Article Recommendation App')

st.header('Select a Article for Recommendation')

selected_article = st.selectbox('Choose a Article', df['title'].unique())

if st.button('Get Recommendations'):
    st.subheader('Recommended Articles:')
    recommendations = get_recommendations(selected_article)
    st.write(recommendations)