import requests
from requests.exceptions import HTTPError
from bs4 import BeautifulSoup
import openai
import os
from openai import OpenAI
import tiktoken
from itertools import islice
import sys
import numpy as np
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import requests

# EMBEDDING ------------------------------------------------------

# Get num tokens
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Set up OpenAI
client = OpenAI()

EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

# Do embeddings with batching

# Vanilla embedding function
def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    response = client.embeddings.create(input=text_or_tokens, model=model)
    return response.data[0].embedding

# Breaks up a sequence into chunks
def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch

# Encodes string into tokens, break into tokens
def chunked_tokens(text, encoding_name, chunk_length):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator

# Get safe embedding
def len_safe_get_embedding(text, model=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING, average=True):
    chunk_embeddings = []
    chunk_lens = []
    for chunk in chunked_tokens(text, encoding_name=encoding_name, chunk_length=max_tokens):
        chunk_embeddings.append(get_embedding(chunk, model=model))
        chunk_lens.append(len(chunk))

    if average:
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings

# EMBEDDING STUFF ------------------------------------------------------

response = requests.get("https://curius.app/api/users/2055/links?page=0")
# print(response.content)
json = response.json()

# for each curius page, embed all the links

user_embeddings_list = []

for i in range(len(json['userSaved'])):

    i = 26

    # Scrape link
    url = json['userSaved'][i]['link']
    print(i, url)
    link_response = requests.get(url)
    print(link_response.status_code)

    # if successful
    if link_response.__bool__():
        html = link_response.text
        print(html)
        # soup = BeautifulSoup(html, 'html.parser')
        soup = BeautifulSoup(html, 'html.parser')

        # Extract text from the parsed HTML
        text = ' '.join(soup.stripped_strings)
        # print(text)

        if text != '':
        # Add embedding for link
            user_embeddings_list.append(len_safe_get_embedding(text, model="text-embedding-3-small"))
        else:
            pass
    else:
        pass

# Convert the list to a NumPy array when ready
embeddings_array = np.array(user_embeddings_list)

# if response.status_code == 200:
#     print("Success!")
# elif response.status_code == 404:
#     print("Not Found.")

# if response.__bool__():
#     print("Success!")
# else:
#     raise Exception(f"Non-success status code: {response.status_code}")

# print(text)
print(num_tokens_from_string(text, "cl100k_base"))

# Example of appending an embedding to the list