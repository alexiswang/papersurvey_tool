
INSTRUCTION_PROMPT = """
    Here is a piece of text from a research paper. 
    Article:
    {text_chunk}
   ----
    
    Summarise the given text by following the Guidance below.
    
    Guidance:
    - Summarise the the key conclusions in a single short paragraph of approx. 100 words.
    - the summary should be highly dense ans concise yet self-contained. 
    - list relevant findings, observations and supportive arguments for the key conclusionsin in 3-5 bullet points. 
    - each finding has 1-2 sentences and be as close to the original text
    - describe the methods in 3-5 short sentences

    <format>
    Summary: <conclusion>
    Findings: <findings>
    Methods: <methods>
    </format>   
"""
ROLE_PROMPT = """You are a reseach assistant with the task to do literature review. You never make up any information that isn't in the literatures."""

EXTRACTION_PROMPT = """
    for the given text {text_chunk}, identify the title and authors of the paper. 

    Write output in JSON form as shown in <format> tags.

    <format>
    {{
            "title": title,
            "authors": list of all authors,
    }}
    </format>
"""