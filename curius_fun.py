import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout
from bs4 import BeautifulSoup
from openai import OpenAI
import tiktoken
from itertools import islice
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# EMBEDDING A LINK ------------------------------------------------------

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

# EMBEDDING A USER ------------------------------------------------------

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2), retry=retry_if_exception_type((ConnectionError, Timeout)))
def robust_get(url):
    return requests.get(url)

def process_link(url):
    try:
        link_response = robust_get(url)
        if link_response.status_code == 200:
            html = link_response.text
            soup = BeautifulSoup(html, 'html.parser')
            text = ' '.join(soup.stripped_strings)
            if text:
                return len_safe_get_embedding(text, model="text-embedding-3-small")
    except Exception as e:
        print(f"Failed to process URL {url}: {e}")
    return None

def fetch_and_process_pages(base_url, start_page=0):
    page = start_page
    user_embeddings_list = []

    while True:
        url = f"{base_url}?page={page}"
        response = requests.get(url)
        if response.status_code != 200:
            print("Failed to fetch data:", response.status_code)
            break

        data = response.json()

        if not data['userSaved']:  
            print("No more data available.")
            break

        for item in data['userSaved']:
            link_url = item['link']
            print(f"Processing URL: {link_url}")
            embedding = process_link(link_url)
            if embedding is not None:
                user_embeddings_list.append(embedding)

        page += 1

    if not user_embeddings_list:
        return None

    embeddings_array = np.array(user_embeddings_list)
    average_embedding = np.mean(embeddings_array, axis=0)
    normalized_average_embedding = average_embedding / np.linalg.norm(average_embedding)
    
    return normalized_average_embedding.tolist()

# Example usage
base_url = "https://curius.app/api/users/2414/links"
user_embedding = fetch_and_process_pages(base_url)
print(user_embedding)
