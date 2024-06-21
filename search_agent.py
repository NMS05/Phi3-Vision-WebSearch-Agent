from web_scraper import Scraper
from contriever_wrapper import Contriever_Model
from server_apis import My_VLM_APIs

from PIL import Image
import requests
from io import BytesIO
import textwrap3

class vlm_rag_agent():
    def __init__(self, gradio_demo=False):

        # whether to print result (for chat.py) or return a one large response string (for gradio_demo.py)
        self.gradio_demo = gradio_demo
        self.gradio_string = ""

        # initialize the search tool
        self.scraper = Scraper()

        # using the VLMs as APIs (implemented using FastAPI framework) enables faster prototyping and you can also share the same VLM between multiple agents
        self.vlm = My_VLM_APIs(model_name='phi3_vision')

        # contriever model is pretty small and quickly loads into a GPU, so no need to create an API server for this
        self.contriever = Contriever_Model()

        # placeholder for image
        self.image = None
        # used to check if new (image & search results) must be loaded or current (image & search results) is enough!
        self.currently_active_image_url = ""
        self.is_search_results_available_for_currently_active_image_url = False

        # complete set of prompts used by this agent
        self.prompt_store = {
                "tool_use_prompt": "[Instruction] For the given image, strictly answer the following question if only if you are confident that your answer is correct. If you are not confident about your answer (or) the image/question is beyond your understanding, reply with '[SEARCH]' to leverage external search tools.\n[Question] {}",
                "get_contriever_search_keywords": "[Question] {}.\n[Instruction] To help you answer this question, there is an external search tool available. However, the tool cannot understand complex queries and can only accept keywords to retrieve relevant contexts. Now, for the given Image and the corresponding Question, reply with the appropriate keywords that can retieve the most informative contexts.  Strictly reply with NO MORE THAN FIVE keywords.",
                "answer_with_context_prompt": "[Instruction] The following passage contains the response from an external search tool.\n[Search Tool Response] {} If the passage does not provide sufficient information to answer the following question, reply with [SKIP PASSAGE] to skip to the next.\n[Question] {}",
                "self_check": "\n[Context] {}.\n[Response] {}.\n[Instruction] You need to validate if the above response is clearly and unambiguously supported by the prior context. If the response is supported by context reply with [OK]. If the response does not provide sufficient information or is not supported by the context, reply with [NOT SUPPORTED].",
                "concistency_check": "[Question] {}.\n[Instruction] For the above question the following responses were supported by various contexts. You need to aggregate the information in all the following responses (strictly do not add any new information) and reply with a coherent and consistent final response. {}",
            }

    def load_image(self, image_url):
        image = Image.open(BytesIO(requests.get(image_url,timeout=5.0).content))
        image = image.resize((384,384))
        return image
    
    def perform_reverse_image_search(self, image_url, print_search_results=False):
        # get search results
        search_results = self.scraper.get_search_result(image_url)
        if print_search_results:
            for key, value in search_results.items():
                print(f'{key} Title: {value["Title"]}\n{key} URL: {value["URL"]}\n')
        # extract webpage contents from search results - saved by default to 'parsed_search_results.json'
        self.scraper.extract_webpage_contents(search_results)

    # print results directly in terminal (or) format as a string and send to gradio 
    def print_output_or_format_string(self, content):
        if not self.gradio_demo:
            print(textwrap3.fill(content),width=180)
        else:
            self.gradio_string += f"\n{content}"

    """
    This is where the complete Agentic workflow is defined!
    Note that current version does not maintain a chat history (enables with VLMs having limited context window)
    """
    def chat_with_agent(self, image_url, question):

        # 
        #### load the new image every time a new image_url is provided ####
        # 
        if image_url == self.currently_active_image_url:
            pass
        else:
            self.print_output_or_format_string(f"\n Loading new image!")
            self.image = self.load_image(image_url)
            self.currently_active_image_url = image_url
            self.is_search_results_available_for_currently_active_image_url = False
    
        # 
        #### First check if the question can be answered directly or if an external search tool is required! ####
        # 
        query = self.prompt_store["tool_use_prompt"].format(question)
        agent_response = self.vlm.get_response(self.image, query)

        if "SEARCH" in agent_response:

            self.print_output_or_format_string(f"\n Initial Response: {agent_response}")

            # 
            #### get search results only once whenever a new image_url is provided ####
            # 
            if image_url == self.currently_active_image_url and self.is_search_results_available_for_currently_active_image_url == True:
                pass
            else:
                self.print_output_or_format_string(f"\n Performing Reverse Image Search over the internet!")
                self.perform_reverse_image_search(image_url)
                self.is_search_results_available_for_currently_active_image_url = True

            
            # 
            #### (Optional Step, Not Mandatory) from the question, come up with appropriate keywords for the contriever model to retrieve the relevant passages ####
            # 
            query = self.prompt_store["get_contriever_search_keywords"].format(question)
            contriever_keywords = self.vlm.get_response(None, query)
            self.print_output_or_format_string(f"\nSearching with the keywords : {contriever_keywords}")


            # 
            #### get top-K contexts that might potentially support the given guery ####
            # 
            contriever_search_query = f"Question: {question} Keywords: {contriever_keywords}"
            top_K_contexts = self.contriever.get_topK_contexts(query=contriever_search_query, K=10)


            # store the agent_responses for individual contexts. finally aggregate them into one final answer!
            generated_answers_over_multiple_contexts = []


            # for each of the top-10 contexts
            for i, (context, relevance_score) in enumerate(top_K_contexts):
                context = top_K_contexts[i][0]['content']

                # 
                #### instruct vlm to answer the query with the supporting context ####
                # 
                query = self.prompt_store["answer_with_context_prompt"].format(context, question)
                agent_response = self.vlm.get_response(self.image, query)

                # 
                # if currently provided context doesnot contain the relevant information, the model simply skips to the next context
                # 
                if "SKIP PASSAGE" in agent_response:
                    self.print_output_or_format_string(f"\n[SKIP] Context {i+1} - {context[:50]} .... !")
                    continue

                # 
                #### the model replies with an answer supported by the context ####
                ## however, we still need to double check the answer. so, we instruct the model to check again if the answer is correctly supported by the context! ##
                # 
                else:
                    answer_to_be_verified = agent_response
                    query = self.prompt_store["self_check"].format(context, answer_to_be_verified)
                    agent_response = self.vlm.get_response(self.image, query)  # model replies with "OK" or "NOT SUPPORTED" 

                    if "OK" in agent_response:
                        # just making the results easy to read
                        self.print_output_or_format_string(f"\n*Answer* : {answer_to_be_verified}")
                        self.print_output_or_format_string(f"\nSupported by Context {i+1} : {context}")
                        self.print_output_or_format_string(f"\nReference - {top_K_contexts[i][0]['URL']}")

                    elif "NOT SUPPORTED" in agent_response:
                        self.print_output_or_format_string(f"\n[SKIP] Context {i+1} - {context[:50]} .... !")
                        pass
                    
                    else:
                        self.print_output_or_format_string("\n^^^^ Undesired Response ^^^^\n")
            
            """
            #                           ######### aggregate all previous answers to one final answer #########
            #               *** WARNING! - aggregated answers might lose info in individual answers or might contain hallucinations ***
            #                ****** Hallucination is severe for the Phi3-Vision model at this step :(  , So use with caution  ******

            # combine the all previous responses into one large string in the format "[Response-1] Phi3V is an AI model... [Response-2] It was developed by Microsoft..."
            formatted_string = ""
            for i, answers in enumerate(generated_answers_over_multiple_contexts):
                formatted_string += f"\n[Response-{i+1}] {answers}. "

            # create a final query to the model
            query = self.prompt_store["concistency_check"].format(question,formatted_string)
            aggregated_final_answer = self.vlm.get_response(None,query) # image not required for this step

            # just making the results easy to read
            self.print_output_or_format_string("\n"+"="*100+"\n")
            self.print_output_or_format_string(f"**\nFinal Answer** : {aggregated_final_answer}\n")
            """

            if self.gradio_demo:
                gradio_string_copy = self.gradio_string
                self.gradio_string = "" # reset after answering a query
                return gradio_string_copy


        # 
        #### vlm is confident enough to answer directly (might still contain hallucinations...!!!) ####
        # 
        else:
            self.print_output_or_format_string(f"\n*Direct Answer* : {agent_response}")
            if self.gradio_demo:
                gradio_string_copy = self.gradio_string
                self.gradio_string = "" # reset after answering a query
                return gradio_string_copy