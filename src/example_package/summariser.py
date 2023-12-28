from typing import Any
import os
from serpapi import GoogleSearch
import regex as re
import prompts
from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
GOOGLE_SEARCH_API_KEY = os.environ["GOOGLE_SEARCH_API_KEY"]


class PaperSummariser:

    def __init__(self) -> None:
        self.stop_at = 7
        self.PROMPTS = prompts

    def _summarise_content(self, text) -> Any:
        
        text_chunk_summary = []
        for i in range(self.stop_at):
            text_chunk = text[i].page_content
            page_summary = self._summarise_page(text_chunk)
        text_chunk_summary.append(page_summary)

        all_page_summary ="\n".join(text_chunk_summary)
        final_summary = self._summarise_page(all_page_summary)

        return final_summary

    def _summarise_page(self, text_chunk):
        
        prompt = self.PROMPTS.INSTRUCTION_PROMPT.format(text_chunk=text_chunk)
        messages = [{"role": "system", "content": self.PROMPTS.ROLE_PROMPT},
                {"role": "user", "content": prompt}]
        client = OpenAI()
        response = client.chat.completions.create(
                    messages=messages,
                    model = "gpt-4", 
        )
        return response.choices[0].message.content
    
    def get_paper_info(self, text_chunk):

        prompt = self.PROMPTS.EXTRACTION_PROMPT.format(text_chunk=text_chunk)
        messages = [{"role": "system", "content": self.PROMPTS.ROLE_PROMPT},
                {"role": "user", "content": prompt}]
        client = OpenAI()
        response = client.chat.completions.create(
                messages=messages,
                model = "gpt-4", 
        )
        info = eval(response.choices[0].message.content)

        return info
    
    @staticmethod
    def google_scholar_search(query):
        params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": GOOGLE_SEARCH_API_KEY
    }
        search = GoogleSearch(params)
        results = search.get_dict()
        summary = results["organic_results"][0]["publication_info"]["summary"]
        publish_year = re.search(r'(19|20)\d\d', summary).group(0)
        cited_times = results["organic_results"][0]["inline_links"]["cited_by"]["total"]

        return publish_year, cited_times

    def preprocess(self, filepath):

        loader = PyPDFLoader(filepath)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=0)
        text = text_splitter.split_documents(documents)

        return text
    
    def summarise(self, paper_path):
        summary = {}
        text = self.preprocess(paper_path)

        front_page = text[0]
        info = self.get_paper_info(front_page)
        title = info["title"]
        summary.update(info)

        publish_year, cited_times = self.google_scholar_search(query=title)
        summary.update({"publish_year": publish_year, "cited_times": cited_times})

        final_summary = self._summarise_content(text)
        summary.update({"summary": final_summary})

        return summary

    
