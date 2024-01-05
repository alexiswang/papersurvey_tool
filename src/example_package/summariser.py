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


from typing import Any
from serpapi import GoogleSearch
import regex as re
import prompts
from openai import OpenAI
from llmsherpa.readers import LayoutPDFReader


default_prompts = {"system_prompt": prompts.ROLE_PROMPT,
                   "user_prompt": prompts.INSTRUCTION_PROMPT,
                   "extraction_prompt": prompts.EXTRACTION_PROMPT}


class PaperSummariser:

    def __init__(self, model: str = "gpt-4", prompts: dict = default_prompts, temperature: float = 1) -> None:
        
        self.prompts = prompts
        self.model = model
        self.temperature = temperature
        self.stop_at = 7

    def _summarise_content(self, text_chunks) -> Any:
        
        text_chunk_summary = []
        for text_chunk in text_chunks:
            page_summary = self._summarise_page(text_chunk)
            text_chunk_summary.append(page_summary)
        self.summary_list = text_chunk_summary

        all_page_summary ="\n".join(text_chunk_summary)
        final_summary = self._summarise_page(all_page_summary)

        return final_summary

    def _summarise_page(self, text_chunk):
        
        prompt = self.prompts["user_prompt"].format(text_chunk=text_chunk)
        messages = [{"role": "system", "content": self.prompts["system_prompt"]},
                {"role": "user", "content": prompt}]
        client = OpenAI()
        response = client.chat.completions.create(
                    messages=messages,
                    model = self.model, 
                    temperature=self.temperature
        )
        return response.choices[0].message.content
    
    def get_paper_info(self, text_chunk):
        """ extract the title and the authors from the first page        
        """

        prompt = self.prompts["user_prompt"].format(text_chunk=text_chunk)
        messages = [{"role": "system", "content": self.prompts["system_prompt"]},
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
    
    def split_by_chunk_size(self, file_path, chunk_size=4000):
        """ this does not account for dropping irrelevant chunks such as references and appendices
        """

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        content = "\n".join([doc.page_content for doc in documents])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        text_chunks = text_splitter.split_text(content)

        return text_chunks
    
    def split_by_section(self, file_path):

        llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
        # pdf_url = "../example_paper1.pdf" # also allowed is a file path e.g. /home/downloads/xyz.pdf
        pdf_reader = LayoutPDFReader(llmsherpa_api_url)
        doc = pdf_reader.read_pdf(file_path)
        
        section_list = [section for section in doc.sections() if section.level == 0]
        remove_list = ["references", "appendix", "acknowledgment"]
        selected_sections = [section for section in section_list if not any(remove_section in section.title.lower() for remove_section in remove_list)]

        text_chunks = []
        for section in selected_sections:
            text_to_include = ""
            text = section.to_text(include_children=True, recurse=True)
            text_to_include += text
            text_chunks.append(text_to_include)

        return text_chunks
    
    
    def preprocess(self, file_path, split_by="section"):

        if split_by == "section":
            text_chunks = self.split_by_section(file_path)
        elif split_by == "chunk_size":
            text_chunks = self.split_by_chunk_size(file_path)

        return text_chunks
    
    def summarise(self, paper_path, verbose=False):
        summary = {}
        text_chunks = self.preprocess(paper_path)

        if verbose:
            front_page = text_chunks[0]
            info = self.get_paper_info(front_page)
            title = info["title"]
            summary.update(info)
            publish_year, cited_times = PaperSummariser.google_scholar_search(query=title)
            summary.update({"publish_year": publish_year, "cited_times": cited_times})

        final_summary = self._summarise_content(text_chunks)
        summary.update({"summary": final_summary})

        return summary

