import pandas as pd
import os,sys
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel
import torch
import nltk

dir_ = sys.argv[1]

train_df = pd.read_csv(dir_ + '/col_774_A4_2023/HandwrittenData/train_hw.csv')
train_df.rename(columns={'image': 'file_name', 'formula': 'text'}, inplace=True)
# del df[2]
# print(df)
train_df.head()

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
    
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
train_dataset = IAMDataset(root_dir=dir_+'/col_774_A4_2023/HandwrittenData/images/train/',
                           df=train_df,
                           processor=processor,
                           max_target_length = 128)
train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)

# Training the model



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
model.to(device)

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 128
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 1
model.config.length_penalty = 1.5
model.config.num_beams = 2


def sbleu(GT,PRED):
    score = 0
    for i in range(len(GT)):
        Lgt = len(GT[i].split(' '))
        if Lgt > 4 :
            cscore = nltk.translate.bleu_score.sentence_bleu([GT[i].split(' ')],PRED[i].split(' '),weights=(0.25,0.25,0.25,0.25),smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)
        else:
            weight_lst = tuple([1.0/Lgt]*Lgt)
            cscore = nltk.translate.bleu_score.sentence_bleu([GT[i].split(' ')],PRED[i].split(' '),weights=weight_lst,smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)
        score += cscore
    return score/(len(GT))

from transformers import AdamW
from tqdm.notebook import tqdm

optimizer = AdamW(model.parameters(), lr=5e-5)

# Model Training
for epoch in range(10):  # loop over the dataset multiple times
   # train
   model.train()
   train_loss = 0.0
   for batch in tqdm(train_dataloader):
      # get the inputs
      for k,v in batch.items():
        if k == 'labels' or k == 'pixel_values':
          batch[k] = v.to(device)

      # forward + backward + optimize
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      train_loss += loss.item()

   print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
   torch.save(model,dir_+'model_1.pth')
   print("Model Saved")
   
model.save_pretrained(".")

# Predictions

def pred(file,processor:TrOCRProcessor,model):
    image = Image.open(file).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to('cuda')
    generated_ids = model.generate(pixel_values, max_length = 60)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # g = generated_text.split()

    return generated_text


test_output = pd.read_csv(dir_ + '/sample_sub.csv')
test_output.rename(columns={'image': 'file_name', 'formula': 'text'}, inplace=True)
test_output.head()

arr = test_output.to_numpy()

root_dir = dir_ + '/col_774_A4_2023/HandwrittenData/images/test/'

output_ = [""]*(len(test_output)+2)
for i in range(len(test_output)):
    output_[i] = pred(root_dir+test_output['file_name'][i],processor=processor,model=model)

for o in range(len(output_)):
  if output_[o] == '' or output_[o] == "" or output_[o] == ' ' or output_[o] == " ":
    output_[o] = "$ $"

x = np.column_stack((arr[:,0],np.array(output_[:-2])))

DF = pd.DataFrame(x)
DF.rename(columns={0: 'image', 1: 'formula'}, inplace=True)
DF.to_csv("./comp.csv",index=False)

