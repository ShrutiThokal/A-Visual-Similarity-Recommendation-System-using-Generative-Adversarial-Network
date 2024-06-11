# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pathlib 
import os

import matplotlib.pyplot as plt 

plt.style.use('ggplot')

import numpy as np 

import pandas as pd 

import joblib

import cv2

from sklearn.metrics.pairwise import cosine_similarity

from pandas.core.common import flatten


import torch
import torch.nn as nn


import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable


from PIL import Image

import warnings
warnings.filterwarnings("ignore")
DATASET_PATH = "F:\\final\\Data\\"

print(os.listdir(DATASET_PATH))
import pandas as pd

with open(DATASET_PATH + 'styles.csv', 'r', errors='ignore') as f:
    df = pd.read_csv(f)

print(df.head(10))

import os

def get_all_filenames(directory):
    """
    Returns a set of all filenames in the given directory.
    """
    filenames = {entry.name for entry in os.scandir(directory) if entry.is_file()}
    return filenames

images = get_all_filenames(DATASET_PATH + "images/")
def check_image_exists(image_filename):
    """
    Checks if the desired filename exists within the filenames found in the given directory.
    Returns True if the filename exists, False otherwise.
    """
    global images
    if image_filename in images:
        return image_filename
    else:
        return np.nan

df['image'] = df["id"].apply(lambda image: check_image_exists(str(image) + ".jpg"))
df = df.reset_index(drop=True)
df.head()
import cv2


IMAGE_PATH = "F:\\final\\Data\\images\\"

def plot_figures(figures, nrows=1, ncols=1, figsize=(8, 8)):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()  

def image_location(img):
    return IMAGE_PATH + img


def import_img(image):
    image = cv2.imread(image_location(image))
    return image
figures = {'im'+str(i): cv2.imread(IMAGE_PATH + row.image) for i, row in df.sample(6).iterrows()}

plot_figures(figures, 2, 3)
plt.figure(figsize=(7,28))
df.articleType.value_counts().sort_values().plot(kind='barh')
width= 224
height= 224

resnetmodel = models.resnet18(pretrained=True)

layer = resnetmodel._modules.get('avgpool')
resnetmodel.eval()
s_data = transforms.Resize((224, 224))

standardize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

convert_tensor = transforms.ToTensor()
from PIL import Image

missing_img = []
def vector_extraction(resnetmodel, image_id):
   
    try: 
        img = Image.open(IMAGE_PATH + image_id).convert('RGB')
        
        t_img = Variable(standardize(convert_tensor(s_data(img))).unsqueeze(0))
        
        embeddings = torch.zeros(512)
        
        def copy_data(m, i, o):
            embeddings.copy_(o.data.reshape(o.data.size(1)))
            
        hlayer = layer.register_forward_hook(copy_data)
        
        resnetmodel(t_img)
        
        hlayer.remove()
        emb = embeddings
        
        return embeddings
    
    except FileNotFoundError:
        missed_img = df[df['image'] == image_id].index
        print(missed_img)
        missing_img.append(missed_img)
sample_embedding_0 = vector_extraction(resnetmodel, df.iloc[0].image)
img_array = import_img(df.iloc[0].image)
plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
df.shape

import swifter

df_embeddings = df[:5000] 

map_embeddings = df_embeddings['image'].swifter.apply(lambda img: vector_extraction(resnetmodel, img))

df_embs = map_embeddings.apply(pd.Series)
print(df_embs.shape)
df_embs.head()
df_embs.to_csv('F:\\final\\Data\\df_embs.csv')
  
df_embs = pd.read_csv('F:\\final\\Data\\df_embs.csv')
df_embs.drop(['Unnamed: 0'], axis=1, inplace=True)
df_embs.dropna(inplace=True)
import joblib

joblib.dump(df_embs, 'F:\\final\\Data\\df_embs.pkl', compress=9)

df_embs = joblib.load('F:\\final\\Data\\df_embs.pkl')
cosine_sim = cosine_similarity(df_embs) 
cosine_sim[:4, :4]
index_values = pd.Series(range(len(df)), index=df.index)

def recommend_images(ImId, df, top_n=6):
    
    sim_ImId = index_values[ImId]
    
    sim_scores = list(enumerate(cosine_sim[sim_ImId]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:top_n+1]
    
    ImId_rec = [i[0] for i in sim_scores]
    
    ImId_sim = [i[1] for i in sim_scores]
    
    return index_values.iloc[ImId_rec].index, ImId_sim

recommend_images(3810, df, top_n=5)
def Rec_viz_image(input_imageid): 

    idx_rec, idx_sim = recommend_images(input_imageid, df, top_n=6)
    
    print(idx_sim)
    
    plt.imshow(cv2.cvtColor(import_img(df.iloc[input_imageid].image), cv2.COLOR_BGR2RGB))

    figures = {'im'+str(i): import_img(row.image) for i, row in df.loc[idx_rec].iterrows()}
    
    plot_figures(figures, 2, 3)
Rec_viz_image(3810)
Rec_viz_image(2001)
def recm_user_input(image_id):
    
    img = Image.open('F:/final/Data/images/' + image_id).convert('RGB')
        
    t_img = Variable(standardize(convert_tensor(s_data(img))).unsqueeze(0))
       
    embeddings = torch.zeros(512)
    def select_d(m, i, o):
        embeddings.copy_(o.data.reshape(o.data.size(1)))
    hlayer = layer.register_forward_hook(select_d)
    resnetmodel(t_img)
    hlayer.remove()
    emb = embeddings
    
    cs = cosine_similarity(emb.unsqueeze(0), df_embs)
    cs_list = list(flatten(cs))
    cs_df = pd.DataFrame(cs_list, columns=['Score'])
    cs_df = cs_df.sort_values(by=['Score'], ascending=False)
        
    print(cs_df['Score'][:10])
    
    top10 = cs_df[:10].index
    top10 = list(flatten(top10))
    images_list = []
    for i in top10:
        image_id = df[df.index == i]['image']
        print(image_id)
        images_list.append(image_id)
    images_list = list(flatten(images_list))
    print(images_list)
    
    figures = {'im' + str(i): Image.open('F:/final/Data/images/' + i) for i in images_list}
    fig, axes = plt.subplots(2, 5, figsize=(8, 8))
    for index, name in enumerate(figures):
        axes.ravel()[index].imshow(figures[name])
        axes.ravel()[index].set_title(name)
        axes.ravel()[index].set_axis_off()
    plt.tight_layout()
recm_user_input('1163.jpg')
Rec_viz_image(1005)
import matplotlib.pyplot as plt

similarity_scores = [
    [0.9455690477712965, 0.9208131851230633, 0.9095907496582392, 0.893906776609137, 0.8663102645631924, 0.8549615234896648],
    [0.9455690477712965, 0.9208131851230633, 0.9095907496582392, 0.893906776609137, 0.8663102645631924, 0.8549615234896648],
    [0.9437440402332754, 0.9332969869959742, 0.9253229215714276, 0.9206422767283048, 0.9143679451871038, 0.8362684996122964],
    [0.9450994264887427, 0.9431201885786531, 0.9400545022379712, 0.9346222037080203, 0.9340654931824199, 0.9287955775353653],
    [0.9726985504510337, 0.9467401798211644, 0.9442472091289329, 0.9418147257779246, 0.9382510708235534, 0.9371736945295107]
]

accuracy_percentages = [(sum(scores) / len(scores)) * 100 for scores in similarity_scores]

input_images = list(range(1, len(accuracy_percentages) + 1))

plt.figure(figsize=(10, 6))
plt.plot(input_images, accuracy_percentages, marker='o', color='b', linestyle='-', linewidth=2)
plt.xlabel('Input Image')
plt.ylabel('Accuracy Percentage')
plt.title('Accuracy Percentage for Each Input Image')
plt.xticks(input_images)
plt.grid(True)
plt.show()