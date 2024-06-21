import json
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModel

class Contriever_Model():
    def __init__(self,):
        self.device = torch.device("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.model = AutoModel.from_pretrained('facebook/contriever').to(self.device)

    def contriever_mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def get_contriever_embeddings(self, sentences):
        contriver_inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        contriver_outputs = self.model(**contriver_inputs)
        embeddings = self.contriever_mean_pooling(contriver_outputs[0], contriver_inputs['attention_mask']).detach()
        return embeddings

    def load_json_file(self,):
        with open('parsed_search_results.json', 'r', encoding='utf-8') as file:
            return json.load(file)

    def compute_relevance_scores(self, query, contexts):

        query_embedding = self.get_contriever_embeddings([query])[0]
        relevance_scores = []

        for paragraph in contexts:
            # Tokenize the paragraph
            sentences = sent_tokenize(paragraph['content'])
            sentence_embeddings = self.get_contriever_embeddings(sentences)
            # Compute similarity scores and aggregate them
            scores = [torch.dot(query_embedding, sentence_embedding) for sentence_embedding in sentence_embeddings]
            relevance_scores.append(sum(scores).item())
        return relevance_scores

    # Return the top-K most informative contexts relevant to the query
    def get_topK_contexts(self, query, K=7):
        contexts = self.load_json_file()
        relevance_scores = self.compute_relevance_scores(query, contexts)
        # Combine paragraphs and their scores, sort by scores
        paragraphs_with_scores = list(zip(contexts, relevance_scores))
        paragraphs_with_scores.sort(key=lambda x: x[1], reverse=True)
        return paragraphs_with_scores[:K]