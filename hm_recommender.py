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

st.write('The purpose of this recommender is to show you the pieces that a customer may like based on the purchases of previous customers. The company sells more than 30,000 items and therefore you only see a small sample of products here. You can refresh the sample of clothes anytime you want')

st.write('1. Focus only on ladieswear (largest group) and only focus on customer level (but not invoice level): 32m -> 28m.')
st.write('2. Focus on 5000 most bought items 28m -> 10m')
st.write('3. Focus only on customers, who bought more than 20 products')


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
single_articles = np.random.choice(df_sampled['article_id'].unique(), 4) # one specific list of products is
product = st.selectbox('Choose which product you like', single_articles) #select one specific product

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

# st.write('I recommend you the following products:')
# st.write('    -->', product_rec_name0)
# st.write('    -->', product_rec_name1)
# st.write('    -->', product_rec_name2)
# st.write('    -->', product_rec_name3)
# st.write('    -->', product_rec_name4)

picture_path1 = 'images_ladieswear/opt_0'+str(product_recommended[0])+'.jpg'
picture_path2 = 'images_ladieswear/opt_0'+str(product_recommended[1])+'.jpg'
picture_path3 = 'images_ladieswear/opt_0'+str(product_recommended[2])+'.jpg'
picture_path4 = 'images_ladieswear/opt_0'+str(product_recommended[3])+'.jpg'
picture_path5 = 'images_ladieswear/opt_0'+str(product_recommended[4])+'.jpg'

# st.write(picture_path1)
# st.write(picture_path2)
# st.write(picture_path3)
# st.write(picture_path4)
# st.write(picture_path5)

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
st.write('**Other customers bought also (product based)**')
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

st.write('I hope you like the recommondations!!!')