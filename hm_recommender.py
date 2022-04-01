import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px
import numpy as np
from PIL import Image

# pip uninstall click
# pip install click==8.0.4

st.title('Welcome: This is a fashion product recommender')

st.write('This recommender shows you the pieces that other customers bought, when buying the selected product.')
st.write('1. The recommender focus only ladieswear')
st.write('2. It recommends products which are the top 5000 sold products (out of nearly 40000 products).')
# st.write('3. Focus only on customers, who bought more than 20 products')


article_id_women_desc = pd.read_csv('article_id_women_desc.csv') # link between prod id and color and prod name
df_sampled = pd.read_csv('df_sampled_42.csv') # create data framewith sampled data for recommendation
sample_customer_pivot = df_sampled.pivot_table(index='invoice', columns=['article_id'], values='price_quant').fillna(0) # make the table needed for recommendation


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
# single_articles = np.random.choice(df_sampled['article_id'].unique(), 4) # one specific list of products is
# product = st.selectbox('Choose which product you like', single_articles) #select one specific product

if st.button('Select a random product'):
    product = np.random.choice(df_sampled['article_id'].unique(), 1)[0]
else:
    product = np.random.choice(df_sampled['article_id'].unique(), 1)[0]

#product = np.random.choice(df_sampled['article_id'].unique()[0], 1)[0] #select one specific product



#st.write('The selected product is:', article_id_women_desc[article_id_women_desc['article_id'] == product].prod_color.unique()[0])


st.write('**The selected product is**') #, article_id_women_desc[article_id_women_desc['article_id'] == product].prod_color.unique()[0])
#slected_prod_pic = Image.open('pictures/opt_0156289011.jpg')
picture_path = 'images_ladieswear/opt_0'+str(product)+'.jpg'
slected_prod_pic = Image.open(picture_path)


st.image(slected_prod_pic, caption=article_id_women_desc[article_id_women_desc['article_id'] == product].prod_color.unique()[0])

st.write(' ')
st.write(' ')
st.write(' ')

df_recommendation = get_recommendation(sample_customer_pivot, product) # get the top 4 recommended products

product_recommended = list(df_recommendation.article_id.iloc[1:6]) # get the top 4 recommended products
product_correlation = list(df_recommendation.correlation.iloc[1:6]) # get the top 4 recommended products

product_rec_name0 = article_id_women_desc[article_id_women_desc['article_id'] == product_recommended[0]].prod_color.unique()[0]
product_rec_name1 = article_id_women_desc[article_id_women_desc['article_id'] == product_recommended[1]].prod_color.unique()[0]
product_rec_name2 = article_id_women_desc[article_id_women_desc['article_id'] == product_recommended[2]].prod_color.unique()[0]
product_rec_name3 = article_id_women_desc[article_id_women_desc['article_id'] == product_recommended[3]].prod_color.unique()[0]
product_rec_name4 = article_id_women_desc[article_id_women_desc['article_id'] == product_recommended[4]].prod_color.unique()[0]

picture_path1 = 'images_ladieswear/opt_0'+str(product_recommended[0])+'.jpg'
picture_path2 = 'images_ladieswear/opt_0'+str(product_recommended[1])+'.jpg'
picture_path3 = 'images_ladieswear/opt_0'+str(product_recommended[2])+'.jpg'
picture_path4 = 'images_ladieswear/opt_0'+str(product_recommended[3])+'.jpg'
picture_path5 = 'images_ladieswear/opt_0'+str(product_recommended[4])+'.jpg'

#opening the image
image1 = Image.open(picture_path1)
image2 = Image.open(picture_path2)
image3 = Image.open(picture_path3)
image4 = Image.open(picture_path4)
image5 = Image.open(picture_path5)

images = [image1, image2 ,image3, image4, image5] # list of pictures

images_caption = [(product_rec_name0, round(product_correlation[0],2)),  
(product_rec_name1, round(product_correlation[1],2)), 
(product_rec_name2, round(product_correlation[2],2)), 
(product_rec_name3, round(product_correlation[3],2)), 
(product_rec_name4, round(product_correlation[4],2))] #list of picture titles

#displaying the image on streamlit app
st.write('**Other customers bought also these products:**')
st.image(images, width=120, caption=images_caption)


### make the substitutes for the shown products:
# product_recommended_subs = list(df_recommendation.article_id.iloc[-6:-1]) # get the top 4 recommended products
# product_correlation_subs = list(df_recommendation.correlation.iloc[-6:-1]) # get the top 4 recommended products

# product_rec_name0_subs = article_id_women_desc[article_id_women_desc['article_id'] == product_recommended_subs[0]].prod_color.unique()[0]
# product_rec_name1_subs = article_id_women_desc[article_id_women_desc['article_id'] == product_recommended_subs[1]].prod_color.unique()[0]
# product_rec_name2_subs = article_id_women_desc[article_id_women_desc['article_id'] == product_recommended_subs[2]].prod_color.unique()[0]
# product_rec_name3_subs = article_id_women_desc[article_id_women_desc['article_id'] == product_recommended_subs[3]].prod_color.unique()[0]
# product_rec_name4_subs = article_id_women_desc[article_id_women_desc['article_id'] == product_recommended_subs[4]].prod_color.unique()[0]

# picture_path1_subs = 'images_ladieswear/opt_0'+str(product_recommended_subs[0])+'.jpg'
# picture_path2_subs = 'images_ladieswear/opt_0'+str(product_recommended_subs[1])+'.jpg'
# picture_path3_subs = 'images_ladieswear/opt_0'+str(product_recommended_subs[2])+'.jpg'
# picture_path4_subs = 'images_ladieswear/opt_0'+str(product_recommended_subs[3])+'.jpg'
# picture_path5_subs = 'images_ladieswear/opt_0'+str(product_recommended_subs[4])+'.jpg'

# #opening the image
# image1_subs = Image.open(picture_path1_subs)
# image2_subs = Image.open(picture_path2_subs)
# image3_subs = Image.open(picture_path3_subs)
# image4_subs = Image.open(picture_path4_subs)
# image5_subs = Image.open(picture_path5_subs)

# images_subs = [image1_subs, image2_subs ,image3_subs, image4_subs, image5_subs] # list of pictures

# images_caption_subs = [(product_rec_name0_subs, round(product_correlation_subs[0],2)),  
# (product_rec_name1_subs, round(product_correlation_subs[1],2)), 
# (product_rec_name2_subs, round(product_correlation_subs[2],2)), 
# (product_rec_name3_subs, round(product_correlation_subs[3],2)), 
# (product_rec_name4_subs, round(product_correlation_subs[4],2))] #list of picture titles

# #displaying the image on streamlit app
# st.write('**Other customers did not by these products (product and price based) - Substitutes**')
# st.image(images_subs, width=120, caption=images_caption_subs)