import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)

from transformers import AutoTokenizer,AutoModel
from tqdm import tqdm

# with open ("model","rb") as fin:
#     model = pickle.load(fin)
# with open ("tokenizer","rb") as fin:
#     tokenizer = pickle.load(fin)



# data
fp=open("res.txt")
content_list=[]
for line in fp:
    # content = line.split("\t")[3]
    # print(line)
    # exit()
    content_list.append(line)
fp.close()


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')


# with open ("model","wb") as fout:
#     pickle.dump(model,fout)

# with open ("tokenizer","wb") as fout:
#     pickle.dump(tokenizer,fout)

# exit()


device=torch.device("cuda")

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# def get_embed(y_def,model,tokenizer ):
#     # Tokenize sentences
#     encoded_input = tokenizer(y_def, padding=True, truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         model_output = model(**encoded_input.to(device))
#         # Perform pooling
#         sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
#         # Normalize embeddings
#         sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
#         return sentence_embeddings



evaluate_cos=nn.CosineSimilarity(dim=1,eps=1e-6)
# softmax = nn.Softmax()

model.eval()
model.cuda()


class DatasetRetriever(torch.utils.data.Dataset):
    def __init__(self, review):
        self.review = review

    def __len__(self):
        return len(self.review)
    
    def __getitem__(self, index):
        
        review = self.review[index]
        return {"review": review }


train_dataset1 = DatasetRetriever(content_list)
# train_dataset2 = DatasetRetriever(content_list)

train_dataloader1 = DataLoader(
        train_dataset1,
        batch_size=8, 
        sampler=SequentialSampler(train_dataset1),
        num_workers=0
    )

# train_dataloader2 = DataLoader(
#         train_dataset2,
#         batch_size=4, 
#         sampler=SequentialSampler(train_dataset2),
#         num_workers=0
#     )

embedding_list=[]
# matrix=[ [1 for _ in range(len(content_list))  ] for _ in range(len(content_list))  ]

# idx1=0
with torch.no_grad():
    for batch1 in tqdm(train_dataloader1):
        encoded_input1 = tokenizer(batch1["review"], padding=True, truncation=True, return_tensors='pt')
        model_output1 = model(**encoded_input1.to(device))
        sentence_embeddings1 = mean_pooling(model_output1, encoded_input1['attention_mask'])
        sentence_embeddings1 = F.normalize(sentence_embeddings1, p=2, dim=1)
        # print(sentence_embeddings1.shape)
        # exit()
        for emb in sentence_embeddings1:
            embedding_list.append(emb)

    with open ("embedding-res","wb") as fout:
        pickle.dump(embedding_list,fout)

    # for idx1,review1 in enumerate(embedding_list):
    #     for idx2, review2 in enumerate(embedding_list):

    #         results= evaluate_cos(review1,review2)
                
                
    #         matrix[idx1].append(results)
                
    #     idx+=1
    #     if idx1%200==0:
    #         with open ("matrix"+str(idx1),"wb") as fout:
    #             pickle.dump(matrix,fout)
    #         print(idx1)



# with torch.no_grad():
#     for idx1,review1 in enumerate(content_list):
#         for idx2, review2 in enumerate(content_list):
#             if (idx1!=idx2):
#                 encoded_input1 = tokenizer(review1, padding=True, truncation=True, return_tensors='pt')
#                 encoded_input2 = tokenizer(review2, padding=True, truncation=True, return_tensors='pt')

#                 model_output1 = model(**encoded_input1.to(device))
#                 model_output2 = model(**encoded_input2.to(device))
#                 # Perform pooling
#                 sentence_embeddings1 = mean_pooling(model_output1, encoded_input1['attention_mask'])
#                 sentence_embeddings2 = mean_pooling(model_output2, encoded_input2['attention_mask'])
#                 # Normalize embeddings
#                 sentence_embeddings1 = F.normalize(sentence_embeddings1, p=2, dim=1)
#                 sentence_embeddings2 = F.normalize(sentence_embeddings2, p=2, dim=1)

#                 results= evaluate_cos(sentence_embeddings1,sentence_embeddings2)
                
#                 matrix[idx1][idx2]=results
#                 # exit()
#         if idx1%200==0:
#             with open ("matrix"+str(idx1),"wb") as fout:
#                 pickle.dump(matrix,fout)
#             print(idx1)

# # softmax the results
# results = softmax(results)



# # get embedding
# query=get_embed(query,model,tokenizer)
# # get cos similarity
# results= evaluate_cos(query,all_word_embeding)

# # softmax the results
# results = softmax(results)

# top_results500=torch.topk(results,50)  
# ind500=top_results500.indices
# word500=[]
# for ind in ind500:
#     print(gt_name[ind])
#     word500.append(gt_name[ind])

