# Phi3 Vision WebSearch Agent

## Problem Statement

Phi3 Vision is an advanced Vision-Language Model (VLM) developed by Microsoft, achieving state-of-the-art performance on various benchmarks despite its compact size. However, like all Large Language Models (LLMs) with parametric world knowledge, Phi3 Vision has some limitations:

- Limited capacity to remember extensive information due to a finite number of parameters.
- Inability to answer knowledge-intensive queries that require information beyond the provided image. In such cases, it may either refuse to answer or provide incorrect information (hallucinate). In the below example it does both (the bird is a "Common Yellowthroat")!

<p align="center">
    <img src="https://github.com/NMS05/Phi3-Vision-WebSearch-Agent/blob/main/assets/error2.png" width="800" height="350">
</p>

---

## Solution - Retrieval-Augmented Generation (RAG) with Internet

What if the Phi3 Vision model could search the internet to retrieve relevant content, enhancing its ability to answer knowledge-intensive queries? The primary objective of this repository is to provide a hands-on experience in developing an Internet-based RAG Agent from scratch using the Phi3 Vision model (its more like a rudimentary Perplexity AI implementation for a VLM).


## How the Phi3V Agent works?

The Phi-3 Vision (Phi3V) Agent is built on top of the Phi3V Vision-Language Model. The following steps aim to retrieve external information from the internet and minimize hallucinations as much as possible:

1. **Initial Query Handling:** The Phi3V Agent receives an image (url) and a query as input. If the query is straightforward and can be confidently answered by the Phi3V model, it responds directly without using the internet.
2. **Reverse Image Search:** If internet assistance is needed, the Phi3V Agent performs a reverse image search and retrieves English Wikipedia pages containing the input image The Agent extracts the textual content from these pages and stores it in a JSON file.
3. **Context Retrieval:** Using the input query and optionally related search keywords, the Agent retrieves the top-K relevant paragraphs from the parsed search results. The Contriever model is used for query-context relevance scoring.
4. **Answer Generation:** For each context, the Agent answers the query if the context contains relevant information. If not, it skips to the next context. The Agent also displays the corresponding context and URL of the webpage for transparency.
5. **Self-Check:** The Agent performs a self-check on its answers to further reduce hallucinations.
6. **Answer Aggregation:** Optionally, all answers can be aggregated into one final response.

---

## Results

The results below are cherry picked! Don't expect hallucination free multimodal agents anytime soon.

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/At_Jewel_Changi%2C_Singapore_2023_36.jpg/450px-At_Jewel_Changi%2C_Singapore_2023_36.jpg" width="350" height="300">
</p>

<p align="center">
    <img src="https://github.com/NMS05/Phi3-Vision-WebSearch-Agent/blob/main/assets/gradio_demo_direct.png" width="950" height="350">
</p>

<p align="center">
    <img src="https://github.com/NMS05/Phi3-Vision-WebSearch-Agent/blob/main/assets/gradio_demo.png" width="1000" height="500">
</p>

---

## Installation

Just make sure your CUDA version supports flash-attention and you install the PyTorch version compatible with CUDA. Here, the requirements.txt are for CUDA 12.2

```bash
git clone https://github.com/NMS05/Phi3-Vision-WebSearch-Agent.git
cd Phi3-Vision-WebSearch-Agent
conda create -n phi3v_agent python=3.10
conda activate phi3v_agent
pip install -r requirements.txt
```

---

## Overview of the repo

* [`vlm_servers/`](vlm_servers/) - contains the Phi3 Vision model deployed as a FastAPI server.
* [`server_apis.py`](server_apis.py) - a pythonic interface to access the above VLM servers via requests.
* [`web_scraper.py`](web_scraper.py) - performs reverse image search and saves the results as a json file.
* [`contriever_wrapper.py`](contriever_wrapper.py) - python class to obtain query-context relevance scores with contriever model. 
* [`search_agent.py`](search_agent.py) - This is where all the magic happens. The complete Agentic workflow is defined here.
* [`chat.py`](chat.py) - chat with the Agent via the terminal.
* [`gradio_demo.py`](gradio_demo.py) - a very simple gradio code. Modify it to make it more user friendly.
---

## Running the Phi3V Agent

```bash
cd vlm_servers/
uvicorn Phi3_Vision_Server:app --host 127.0.0.1 --port 8001 # run the fastAPI VLM server in a new terminal

python chat.py # if you want to chat with the Agent via terminal.

python gradio_demo.py # if you need to chat with the Agent via a simple UI
```
* Open any Wikipedia page of interest & copy the image_url.
* Provide the image_url and query as input to the Agent.

---

## Remarks

**Vision-Language Models (as FastAPI servers)**
- Alongside the Phi3 Vision model, the MiniCPM-Llama3-V 2.5 model is provided as an example. You can add or remove models by ensuring each model operates on a different server port.
- The Agent's performance depends more on the model's instruction-following capabilities than its vision-language understanding. Converting a model into an Agent is challenging if it doesn't adhere to instructions.

**Web Search Tools**
- Current search tools are workarounds and not proper Bing or Google Search APIs. Excessive requests in a short time may flag the activity as bot-like, affecting search results.
- For simplicity and reliability, only English Wikipedia pages are currently used. This can be modified to include trusted sites (e.g., NASA, Encyclopedia).
- Due to this limitation, you can only use images from Wikipedia pages at the moment.

**Query-Context Relevance Scoring**
- The Contriever model is effective but may struggle with very complex queries. More powerful relevance scoring models are needed for better results.
- The implementation works best when the query contains keywords present in the contexts.

**Agent**
- This repository includes a single Agent. Since the VLM(s) is hosted as a FastAPI server, multiple Agents can be built with a shared VLM (without needing to train the base VLM).
- The prompts used by the Agent are crafted through multiple iterations of interaction with the Phi3 Vision model. You may need to revise these prompts when using new VLMs.

**Hallucinations Mitigation!**
The Agentic workflow (see How Phi3V Agent works?) involves several steps where hallucinations can occur. Here are potential sources and mitigation strategies:

1. **Poor Search Results:** Irrelevant or noisy websites can cause hallucinations (remember the recent "adding glue to your pizza" drama). Current filtering (English Wikipedia only) reduces this, but limits web information diversity.
2. **Poor Context-Scoring:** The Contriever model may fail to accurately score relevant contexts, leading to missed answers. Increasing the K value in top-K can help, but may increase latency.
3. **VLM Instruction-Following:** VLM sometimes does not follow the user instruction (Faithfulness Hallucination). Better instruction-tuned VLMs with stronger LLM backbones can improve faithfulness.
4. **VLM Hallucinations:** The Phi3 Vision model may hallucinate despite having correct context. It occassionally hallucinates at step-4 (answer not supported by context) and severely hallucinates at step-6 (loses info from individual answers or completely fabricates new info). This is a fundamental problem with most AI models and there are two solutions on the horizon.
    - Training models with specific objectives to provide context-supported responses.
    - Adding steps like self-check and consistency-check in the Agentic workflow to reduce hallucinations.

---

## References

* Search Tools - [Google-Reverse-Image-Search](https://github.com/RMNCLDYO/Google-Reverse-Image-Search/tree/main)  and [ReverseImageSearcher](https://github.com/Vorrik/Google-Reverse-Image-Search/tree/master)
* Agentic workflow draws ideas from [Self-RAG](https://arxiv.org/abs/2310.11511), [Wiki-LLAVA](https://arxiv.org/abs/2404.15406), [Reverse Image Retrieval](https://arxiv.org/abs/2405.18740)
* Code snippets in chat.py borrowed from [PrismaticVLMs](https://github.com/TRI-ML/prismatic-vlms/blob/main/scripts/generate.py)
