# cap5771sp25-project

## Team Members
- Jaiharishan Arunagiri Veerakumar (62333614)
- Sravani Garapati (11936780)

## Brazilian E-Commerce Public Dataset by Olist
The dataset is publically available at [here](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data)


## Importing Data
```
path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

print("Path to dataset files:", path)


# Assuming 'path' variable holds the downloaded directory
for filename in os.listdir(path):
  if filename.endswith(".csv"): # Only move csv files
    src = os.path.join(path, filename)
    dst = os.path.join("/content/data", filename)
    shutil.move(src, dst)
    print(f"Moved {filename} to /content/data")

print("CSV files moved successfully.")
```

- The data will be created in '/data' folder.

## Importing Libraries
```
import kagglehub
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from collections import Counter
from nltk import ngrams
```

## Running the .ipynb file
- After installing the data and importing the libraries, run each code block sequentially to find results.

## File Structure

```
project-root/
├── Readme.md
├── Report/             
│   ├── Milestone1.pdf 
│   ├── Milestone2.pdf     
├── data/               
│   ├── olist_customers_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_orders_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── product_category_name_translation.csv
├── scripts/            
│   ├── main.ipynb
│   ├── main2.ipynb
│   ├── nlp.ipynb
```