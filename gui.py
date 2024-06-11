import os
import torch
import pandas as pd
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image, ImageTk
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import Tk, Label, Entry, Button, filedialog, Frame
import cv2

DATASET_PATH = "F:\\final\\Data\\"
IMAGE_PATH = DATASET_PATH + "images\\"

df = pd.read_csv(os.path.join(DATASET_PATH, 'styles.csv'))

resnetmodel = models.resnet18(pretrained=True)
layer = resnetmodel._modules.get('avgpool')
resnetmodel.eval()

s_data = transforms.Resize((224, 224))
convert_tensor = transforms.ToTensor()
standardize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

df_embs = pd.read_csv(os.path.join(DATASET_PATH, 'df_embs.csv'), index_col=0)

def vector_extraction(resnetmodel, image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        t_img = Variable(standardize(convert_tensor(s_data(img))).unsqueeze(0))
        embeddings = torch.zeros(512)
        def copy_data(m, i, o):
            embeddings.copy_(o.data.reshape(o.data.size(1)))
        hlayer = layer.register_forward_hook(copy_data)
        resnetmodel(t_img)
        hlayer.remove()
        return embeddings
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return None

def recommend_images(emb, df_embs, top_n=6):
    sim_scores = list(enumerate(cosine_similarity(emb.unsqueeze(0), df_embs.values)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    ImId_rec = [i[0] for i in sim_scores]
    ImId_sim = [i[1] for i in sim_scores]
    return ImId_rec, ImId_sim

def import_img(image_path):
    return cv2.imread(image_path)

browsed_image = None

def browse_files():
    global browsed_image
    filename = filedialog.askopenfilename(initialdir=IMAGE_PATH, title="Select an Image", filetypes=(("JPEG files", "*.jpg"), ("all files", "*.*")))
    entry.delete(0, 'end')
    entry.insert(0, filename)

    display_image(filename)
    browsed_image = Image.open(filename)

def display_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((400, 400))
    img = ImageTk.PhotoImage(img)
    selected_image_panel.configure(image=img)
    selected_image_panel.image = img

def show_similar_images():
    global similar_images_frame
    for widget in similar_images_frame.winfo_children():
        widget.destroy()

    image_path = entry.get()
    

    emb = vector_extraction(resnetmodel, image_path)
    
    ImId_rec, ImId_sim = recommend_images(emb, df_embs, top_n=6)
    
    for i, (idx, score) in enumerate(zip(ImId_rec, ImId_sim)):
        similar_image_path = os.path.join(IMAGE_PATH, str(df.iloc[idx]['id']) + ".jpg")
        image = import_img(similar_image_path)
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img.thumbnail((200, 200))
        img = ImageTk.PhotoImage(img)
        
        img_label = Label(similar_images_frame, image=img)
        img_label.image = img
        img_label.grid(row=i // 3 * 2, column=i % 3, padx=5, pady=5)
        
        score_label = Label(similar_images_frame, text=f"Accuracy: {score:.2f}")
        score_label.grid(row=i // 3 * 2 + 1, column=i % 3, padx=5, pady=5)

def save_browsed_image():
    image_path = entry.get()
    if image_path:
        save_image(image_path)

root = Tk()
root.title('Image Similarity Finder')
root.geometry('1000x800')

selected_image_frame = Frame(root)
selected_image_frame.pack(pady=10)
selected_image_panel = Label(selected_image_frame)
selected_image_panel.pack()

label = Label(root, text="Select Image File:")
label.pack(pady=10)

entry = Entry(root, width=50)
entry.pack(pady=10)

browse_button = Button(root, text="Browse", command=browse_files)
browse_button.pack(pady=5)

show_button = Button(root, text="Show Similar Images", command=show_similar_images)
show_button.pack(pady=5)

similar_images_frame = Frame(root)
similar_images_frame.pack(pady=10)

root.mainloop()