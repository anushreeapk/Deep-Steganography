
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import requests
from subprocess import call
import os.path
from PIL import Image
from io import BytesIO


# In[2]:


# imagenet_urls_df = pd.read_table('fall11_urls.txt', sep="\t", header=None)
# imagenet_urls_df.columns = ['ID','URL']


# # In[12]:


# a = imagenet_urls_df.sample(20000)


# # In[13]:


# a


# # In[14]:


# a.to_csv(r'imagenet_URLs.txt', header=None, index=None, sep=' ', mode='a')


# In[17]:

count=0
if not os.path.exists('./data/train'):
	os.makedirs('./data/train')

with open("imagenet_URLs.txt","r") as fp:
	for line in fp:
		line = line.split(" ")
		url = line[1]
		Id = line[0]
		
		try:
			request = requests.get(url)
		except:
			print ("Broken Link!")
			continue

		print url, Id
		if request.status_code == 200:
			try:
				img = Image.open(BytesIO(request.content)).convert("RGB")
			except:
				print ("Valid Link, but not valid JPG!")
				continue
			count+=1
			print (count)
			print('Web site exists... Downloading Image')
			call('wget -O data/'+ Id + ".jpg " + url,shell=True)
			if count == 10000:
				print("10000 Images Collected!")
				break



# img_cover_path = "n12754003_5112.jpg"
# a =  Image.open(img_cover_path).convert("RGB")

# except:
				# print "Valid JPG, but not Valid Image"
