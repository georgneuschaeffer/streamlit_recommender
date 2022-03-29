import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px
import numpy as np
from PIL import Image

# pip uninstall click
# pip install click==8.0.4

st.title('Get best style recommendations based on purchased baskets of millions of users!')

st.write('The purpose of this recommender is to show you the pieces that you may like based on the purchases of other people. The company sells more than 30,000 items and therefore you only see a small sample of products here. You can refresh the sample of clothes anytime you want')


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

## Get a list of clothes that are presented to the user
single_articles = np.random.choice(df_sampled['article_id'].unique(), 4) # one specific list of products is
product = st.selectbox('Choose which product you like', single_articles) #select one specific product

#product = np.random.choice(df_sampled['article_id'].unique()[0], 1)[0] #select one specific product

st.write('The selected product is: ', product)
product_recommended = list(get_recommendation(sample_customer_pivot, product).article_id.iloc[1:5]) # get the top 4 recommended products
st.write('The recommender recommends those products')
st.write('One', product_recommended[0])
st.write('Two', product_recommended[1])
st.write('Three', product_recommended[2])
st.write('Four', product_recommended[3])


#opening the image
image1 = Image.open('pictures/opt_0153115039.jpg')
image2 = Image.open('pictures/opt_0156224001.jpg')
image3 = Image.open('pictures/opt_0153115039.jpg')
image4 = Image.open('pictures/opt_0156231001.jpg')
image5 = Image.open('pictures/opt_0156610007.jpg')

images = [image1, image2 ,image3, image4, image5] # list of pictures

images_caption = ['image1_text',  'image2_text', 'image3_text', 'image4_text', 'image5_text'] #list of picture titles

#displaying the image on streamlit app
st.image(images, width=120, caption=images_caption)

# image_iterator = paginator("Select a sunset page", sunset_imgs)
# indices_on_page, images_on_page = map(list, zip(*image_iterator))
# st.image(images_on_page, width=100, caption=indices_on_page)


# result = st.button('click here to get new clothes') # resamples the lsit of products.
# if result: 
#     st.write('new set of clothes!!')
#     single_articles = np.random.choice(df_sampled['article_id'].unique(), 4) # one specific list of products is
# else:
#     st.write('still the same :)')