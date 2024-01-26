import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[3]:


from encdec_model import EncDec, Encoder, Decoder
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


# 5. Predictions

# transform image array
tf_resize_normalize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
])


# In[62]:


class TestDataset(Dataset):
    def __init__(self, csv_file, pdir='.', directory='SyntheticData', transform=tf_resize_normalize):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.dir = directory
        self.pdir = pdir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        if self.dir == 'SyntheticData':
            image = Image.open(f"{self.pdir}/col_774_A4_2023/{self.dir}/images/{img_name}")
        else:
            image = Image.open(f"{self.pdir}/col_774_A4_2023/{self.dir}/images/train/{img_name}")

        if self.transform:
            image_tensor = self.transform(image)
            if self.dir != 'SyntheticData':
                image_tensor = torch.cat((image_tensor, image_tensor, image_tensor), dim=0)

        return image_tensor


# In[29]:


def predict(model: EncDec, dir_data, max_len=629, device='cuda', batch_size=100):

    # if dir_data == 'test':
    #     loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    # else:
    #     loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    loader = DataLoader(dir_data, batch_size=batch_size, shuffle=True)

    final_latex = []
    model.eval()

    with torch.no_grad():
        for i, (images) in enumerate(loader):

            images = images.to(device)
            context = model.encoder(images)
            
            input_dim = context.shape[0]
            hidden = None

            input_token = torch.tensor(
                [[model.word_to_index['<SOF>']]]*input_dim).to(device)

            for _ in range(max_len):
                output, hidden = model.decoder(
                    context, input_token, hidden)

                predicted_token = output.argmax(dim=2)

                if _ == 0:
                    predicted_tokens = predicted_token
                else:
                    predicted_tokens = torch.cat((predicted_tokens, predicted_token), dim=1)

                input_token = predicted_token
                # print(predicted_tokens)

            for i in range(input_dim):
                predicted_latex_list = []
                for j in range(max_len):
                    symbol = model.index_to_word[int(predicted_tokens[i,j])]
                    predicted_latex_list.append(symbol)
                
                # print(predicted_latex_list[0])
                predicted_latex = ' '.join(predicted_latex_list)

                final_latex.append(predicted_latex)

    return final_latex


# In[38]:
# def generate_csv(model, dir_list=['SyntheticData/val','HandwrittenData/val_hw']):
#     for dirt in dir_list:
#         data = TestDataset(f"./col_774_A4_2023/{dirt}.csv")
#         print(len(data))
#         # set max_len=200 bcuz it rarely goes to max
#         predict_list = predict(model, dir_data=data, max_len=200)
        
#         pred_df = pd.DataFrame(data=test_csv['image'], columns=['image'])
#         pred_df['formula'] = test_predict

#         file_name = '_'.join(dirt.split('/')) + '_pred'

#         pred_df.to_csv(file_name, index=False)
#         print(file_name, 'done')

# In[14]:

def load_and_predict(dirn='.'):

    model = torch.load(f'{dirn}/model_epoch3_colab.pth')
    model.state_dict()


    # In[5]:


    test_csv = pd.read_csv(f'{dirn}/col_774_A4_2023/SyntheticData/test.csv')
    val_csv = pd.read_csv(f'{dirn}/col_774_A4_2023/SyntheticData/val.csv')


    test_data = TestDataset(f"{dirn}/col_774_A4_2023/SyntheticData/test.csv", pdir=dirn)
    val_data = TestDataset(f"{dirn}/col_774_A4_2023/SyntheticData/val.csv", pdir=dirn)


    # In[25]:
    test_predict = predict(model, dir_data=test_data, max_len=200)

    test_pred_df = pd.DataFrame(data=test_csv['image'], columns=['image'])
    test_pred_df['formula'] = test_predict

    test_pred_df.to_csv('pred1b_test', index=False)

    print('test don')


    val_predict = predict(model, dir_data=val_data, max_len=200)

    val_pred_df = pd.DataFrame(data=val_csv['image'], columns=['image'])
    val_pred_df['formula'] = val_predict

    val_pred_df.to_csv('pred1b', index=False)

    print('val done')
    return


def main():
    parser = argparse.ArgumentParser(
        description='Load encoder-decoder and predict.')
    parser.add_argument('dataset_dir', type=str,
                        help='Path to the dataset directory')

    args = parser.parse_args()

    load_and_predict(args.dataset_dir)


if __name__ == "__main__":
    main()
