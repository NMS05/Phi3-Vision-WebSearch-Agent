import time
import requests
import logging
from bs4 import BeautifulSoup
from typing import Dict
from urllib.parse import quote, urlparse

# pip install google-image-source-search
from google_img_source_search import ReverseImageSearcher

from nltk.tokenize import sent_tokenize
import json

"""
                                                Image Search Tools

Tool 1 : GoogleReverseImageSearch()
     - Performs Visual Entity Search (image_url --> visual entity --> search results about the visual entity)

Tool 2 : ReverseImageSearcher()
     - Performs Exact Match Search (image_url --> all web pages containing the exact image)

"""
class GoogleReverseImageSearch:
    def __init__(self):
        self.base_url = "https://www.google.com/searchbyimage"
        self.headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        self.retry_count = 3 # number of attempts in trying to open a paginated url ( ... &start_index=0) with request.get()
        self.retry_delay = 2 # delay between each such attempt

    def response(self, query, image_url, max_results=10, max_attempts=3, delay=2):

        """
        max_results = the desired number of top-k search results (note that k-results cannot be guaratanteed, sometime they are just one or two)
        max_attempts = the maximum number of attempts in trying to achieve k-results.
        delay = delay between every such attempt
        """

        self._validate_input(query, image_url)
        encoded_query = quote(query)
        encoded_image_url = quote(image_url)

        url = f"{self.base_url}?q={encoded_query}&image_url={encoded_image_url}&sbisrc=cr_1_5_2"

        all_results = []
        start_index = 0
        current_attempts = 0

        while len(all_results) < max_results and current_attempts < max_attempts:
            if start_index != 0:
                time.sleep(delay)

            paginated_url = f"{url}&start={start_index}"

            response = self._make_request(paginated_url)
            if response is None:
                break

            search_results, valid_content = self._parse_search_results(response.text)
            if not valid_content:
                logging.warning("Unexpected HTML structure encountered.")
                break

            for result in search_results:
                if len(all_results) >= max_results:
                    break
                data = self._extract_result_data(result)
                if data and data not in all_results:
                    all_results.append(data)

            start_index += (len(all_results)-start_index)
            current_attempts+=1

        if len(all_results) == 0:
            logging.warning(f"No results were found for the given query: [{query}], and/or image URL: [{image_url}].")
            return "No results found. Please try again with a different query and/or image URL."
        else:
            return all_results[:max_results]

    def _validate_input(self, query: str, image_url: str):
        if not image_url:
            raise ValueError("Image URL not found. Please enter an image URL and try again.")
        if not self._validate_image_url(image_url):
            raise ValueError("Invalid image URL. Please enter a valid image URL and try again.")

    def _validate_image_url(self, url: str) -> bool:
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
        return any(path.endswith(ext) for ext in valid_extensions)

    def _make_request(self, url: str):
        attempts = 0
        while attempts < self.retry_count:
            try:
                response = requests.get(url, headers=self.headers, timeout=5.0)
                if response.headers.get('Content-Type', '').startswith('text/html'):
                    response.raise_for_status()
                    return response
                else:
                    logging.warning("Non-HTML content received.")
                    return None
            except requests.exceptions.HTTPError as http_err:
                logging.error(f"HTTP error occurred: {http_err}")
                attempts += 1
                time.sleep(self.retry_delay)
            except Exception as err:
                logging.error(f"An error occurred: {err}")
                return None
        return None

    def _parse_search_results(self, html_content: str):
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            return soup.find_all('div', class_='g'), True
        except Exception as e:
            logging.error(f"Error parsing HTML content: {e}")
            return None, False

    def _extract_result_data(self, result) -> Dict:
        link = result.find('a', href=True)['href'] if result.find('a', href=True) else None
        title = result.find('h3').get_text(strip=True) if result.find('h3') else None
        return {"link": link, "title": title} if link and title else {}



"""
                                Perform Reverse Image Search and Parse Web Page Contents
"""

class Scraper():
    def __init__(self, ) -> None:
        self.search_tool_1 = GoogleReverseImageSearch()
        self.search_tool_2 = ReverseImageSearcher()

    def get_search_result(self, image_url):
        search_tool1_results = self.search_tool_1.response(query="", image_url=image_url) # returns a python list
        search_tool2_results = self.search_tool_2.search(image_url) # returns a iterable object


        #### Extract English-Wikipedia pages for info retrieval ####

        filtered_results = {} # store Title + URL as a dictionary
        url_history = [] # to avoid redundancy between two search tools
        webpage_number = 0

        for search_item in search_tool1_results:
            if "Wikipedia" in search_item.get('title') and "en.wikipedia.org" in search_item.get('link'):
                filtered_results[f"page_{webpage_number}"] = {
                    "Title": search_item.get('title'),
                    "URL": search_item.get('link')
                }
                url_history.append(search_item.get('link'))
                webpage_number += 1

        for search_item in search_tool2_results:
            if "Wikipedia" in search_item.page_title and "en.wikipedia.org" in search_item.page_url and search_item.page_url not in url_history:
                filtered_results[f"page_{webpage_number}"] = {
                    "Title": search_item.page_title,
                    "URL": search_item.page_url
                }
                url_history.append(search_item.page_url)
                webpage_number += 1

        assert len(filtered_results) != 0, "\n\tCannot find any English Wikipedia pages matching this search\n"
        return filtered_results

    def extract_webpage_contents(self, search_tool_results):

        #### Extract contents (text paragraphs) from Wikipedia pages and save them to a JSON file.

        json_content = []

        for key, value in search_tool_results.items():
            title = value["Title"]
            url = value["URL"]
            response = requests.get(url, timeout=5.0)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            def parse_element(element):
                if element.name == 'p': # text paragraphs
                    content = element.get_text(strip=True)

                    if len(content) < 100: # remove trivial paragraphs containing less than 100 characters
                        pass

                    else:
                        ## use this if you want to use the entire text paragraph as a context ###
                        json_content.append({
                                    'Title': title,
                                    'URL': url,
                                    'content': content
                                })
                        
                        ## use this if you want smaller contexts of roughly 3 sentences each ##
                        # sentences = sent_tokenize(content)
                        # chunks = [sentences[i:i + 3] for i in range(0, len(sentences), 3)] # split to smaller paragraphs of roughly 3 sentences
                        # for chunk in chunks:
                        #     if len(chunk) > 0:
                        #         chunk_content = ' '.join(chunk)
                        #         json_content.append({
                        #             'Title': title,
                        #             'URL': url,
                        #             'content': chunk_content
                        #         })
                        #     else:
                        #         pass

            # Extract text paragraphs from the page
            for element in soup.find_all(['p']): parse_element(element)

        # Save contents to a JSON file named parsed_search_results.json
        with open('parsed_search_results.json', 'w', encoding='utf-8') as file:
            json.dump(json_content, file, ensure_ascii=False, indent=4)
        file.close()