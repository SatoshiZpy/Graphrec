import numpy as np
import random
import torch
from torch.utils.data import Dataset

class GRDataset(Dataset):
	def __init__(self, data, u_items_list, u_users_list, u_users_items_list, i_users_list, review_list):
		self.users = data["user_id"].tolist()
		self.items = data["item_id"].tolist()
		self.labels = data["label"]
		self.u_items_list = u_items_list
		self.u_users_list = u_users_list
		self.u_users_items_list = u_users_items_list
		self.i_users_list = i_users_list
		self.review_list = review_list

	def __getitem__(self, index):
		rid = self.review_list[index]
		uid = self.users[rid]
		iid = self.items[rid]
		u_items = self.u_items_list[uid]
		u_users = self.u_users_list[uid]
		u_users_items = self.u_users_items_list[uid]
		i_users = self.i_users_list[iid]
		label = self.labels[rid]

		return (uid, iid, label), u_items, u_users, u_users_items, i_users, rid

	def __len__(self):
		return len(self.review_list)
