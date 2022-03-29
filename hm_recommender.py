import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px
import numpy as np

# pip uninstall click
# pip install click==8.0.4

st.title('This is a recommender')

st.write('The purpose of this recommender is to show you the pieces that you may like based on the purchases of other people.')


df_sampled = pd.read_csv('df_sampled_42.csv') # create data framewith sampled data for recommendation
sample_customer_pivot = df_sampled.pivot_table(index='invoice', columns=['article_id'], values='quantity').fillna(0) # make the table needed for recommendation


## make recommender function
def get_recommendation(df,item):
    '''
    generate a set of product recommondations using item-based collaborative filtering.

    Args: df: Dataframe, pandas dataframe containing a matrix of items purchased
    item (string): columns name for target item

    returs:
        recommoncations (dataframe): pandas dataframe containing product recommondation
    '''
    recommendations = df.corrwith(df[item])
    recommendations.dropna(inplace=True)
    recommendations = pd.DataFrame(recommendations, columns = ['correlation']).reset_index()
    recommendations = recommendations.sort_values(by='correlation', ascending=False)

    return recommendations

result = st.button('click here to get new clothes') # returns True

if result: 
    single_articles = np.random.choice(df_sampled['article_id'].unique(), 4) # make list
else:
    single_articles = np.random.choice(df_sampled['article_id'].unique(), 4) # make list

## the recommendation itself
product = st.selectbox('Choose which product you like',single_articles) #select one specific product
product_recommended = list(get_recommendation(sample_customer_pivot, product).article_id.iloc[1:5]) # get the top 4 recommended products

st.write('The three recommendations are:', product_recommended)
st.write('One', product_recommended[0])
st.write('Two', product_recommended[1])
st.write('Three', product_recommended[2])
st.write('Four', product_recommended[3])
