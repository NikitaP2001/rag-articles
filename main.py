from datasets import Dataset
import torch
from transformers import AutoTokenizer, BertModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
import re
from tqdm import tqdm

df = pd.read_csv('Articles.csv', encoding='ISO-8859-1')
corpus = df['Article'].tolist()[:100]

def semantic_chunking(text, max_tokens, model_name='paraphrase-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    sentences = nltk.sent_tokenize(text)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    chunks = []
    current_chunk = []
    current_tokens = 0
    current_embedding = None

    for i, sentence in enumerate(sentences):
        sentence_tokens = len(sentence.split())
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            if current_embedding is None:
                current_embedding = embeddings[i].unsqueeze(0)
            else:
                current_embedding = (current_embedding + embeddings[i].unsqueeze(0)) / 2
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
            current_embedding = embeddings[i].unsqueeze(0)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


# Tokenization: Sentences are tokenized into input IDs and attention masks using the BERT tokenizer.
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
chunked_corpus = []
for article in tqdm(corpus):
    article = article.lower()
    article = re.sub(r'\(.*?\)', '', article)
    chunks = semantic_chunking(article, 100)
    chunked_corpus.extend(chunks)

dataset = Dataset.from_dict({"text": chunked_corpus})
def preprocess(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=128)
tokenized_corpus = dataset.map(preprocess, batched=True)

'''
Embedding Computation: Each tokenized sentence is passed through a pre-trained BERT model.
The output is the [CLS] token embedding (from outputs.last_hidden_state[:, 0, :]), 
which is used as a fixed-length representation of the sentence.
These embeddings are normalized (using FAISS's normalize_L2) and stored in a FAISS index.
'''
bert_model = BertModel.from_pretrained('bert-base-uncased')
def compute_embeddings(tokenized_corpus):
    input_ids = torch.tensor(tokenized_corpus['input_ids'])
    attention_mask = torch.tensor(tokenized_corpus['attention_mask'])
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        # Use the [CLS] token embedding for sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings
corpus_embeddings = compute_embeddings(tokenized_corpus)

'''
Build the similarity search Index: The precomputed normalized embeddings are added to a FAISS index 
(IndexFlatL2), enabling fast nearest-neighbor searches using cosine similarity.
'''
corpus_embeddings_np = corpus_embeddings.cpu().numpy()
corpus_embeddings_np = np.ascontiguousarray(corpus_embeddings_np)
faiss.normalize_L2(corpus_embeddings_np)
index = faiss.IndexFlatL2(corpus_embeddings_np.shape[1])
index.add(corpus_embeddings_np)

def retrieve(query, top_k=4):
    # Tokenize the query
    query_inputs = tokenizer(query, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        query_embedding = bert_model(
            query_inputs['input_ids'], attention_mask=query_inputs['attention_mask']
        ).last_hidden_state[:, 0, :]  # [CLS] token embedding
    query_embedding_np = query_embedding.numpy()
    faiss.normalize_L2(query_embedding_np)
    D, I = index.search(query_embedding_np, top_k)
    return [chunked_corpus[i] for i in I[0]]

generator_model = T5ForConditionalGeneration.from_pretrained('t5-small')
generator_tokenizer = T5Tokenizer.from_pretrained('t5-small')
def generate_answer(query, retrieved_docs):
    context = " ".join(retrieved_docs)
    input_text = f"question: {query} context: {context}"
    inputs = generator_tokenizer(input_text, return_tensors='pt')
    outputs = generator_model.generate(inputs['input_ids'], max_length=50)
    answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

query = "What countries does expirience economic growth?"
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    retrieved_docs = retrieve(query)
    answer = generate_answer(query, retrieved_docs)
    print(f"Query: {query}")
    print(f"Retrieved Documents: {retrieved_docs}")
    print(f"Generated Answer: {answer}")