from evaluation.prompts import RELEVANCY_SCORE_CRITERIA, RELEVANCY_SCORE_STEPS, COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS, CONSISTENCY_SCORE_CRITERIA, CONSISTENCY_SCORE_STEPS, FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS, EVALUATION_PROMPT_TEMPLATE  # noqa
from openai import OpenAI
from bert_score import BERTScorer
from rouge import Rouge


def get_geval_score(
    criteria: str, steps: str, document: str, summary: str, metric_name: str
):
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        metric_name=metric_name,
        document=document,
        summary=summary,
    )

    client = OpenAI()
    response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
    )
    return response.choices[0].message.content


def get_bert_scores(ref_summary: str, eval_summary: str):
    p, r, f = BERTScorer(lang="en", rescale_with_baseline=True).score([ref_summary], [eval_summary])
    return {"p": p,
            "r": r,
            "f": f}


def get_rouge_scores(ref_summary: str, eval_summary: str):
    rouge = Rouge()
    scores = rouge.get_scores(eval_summary, ref_summary)
    return scores


def get_geval_scores(document: str, summary: str):

    evaluation_metrics = {
    "Relevance": (RELEVANCY_SCORE_CRITERIA, RELEVANCY_SCORE_STEPS),
    "Coherence": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
    "Consistency": (CONSISTENCY_SCORE_CRITERIA, CONSISTENCY_SCORE_STEPS),
    "Fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
}
    # scores = {"Evaluation type": [], "Score": []}
    scores = {}
    for eval_type, (criteria, steps) in evaluation_metrics.items():
        result = get_geval_score(criteria, steps, document, summary, eval_type)
        score_num = int(result.strip())
        scores.update({eval_type: score_num})

    return scores


def get_questions(text, n=5):

    closed_end_questions_template = """
    For the given text below, please follow the Guidance to generate {n} questions. 
    
    Text: {text}

    Guidance:
    - questions should be closed-ended that can be answered by 'yes' or 'no'. 
    - questions should be related to the important facts of the text.
    - use distinct information from different parts of the text to generate questions.
    - Return only the questions in JSON as shown in the example output below.

    Example Output: {{questions: [list of questions]}}

    """
    prompt= closed_end_questions_template.format(n=n, text=text)

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def get_answers(text, questions):

    closed_end_answers_template = """
    You are given several questions separated by '\n\n' and a text. 
    Answer each question in 'yes', 'no', or 'idk'.
    For each qusetion, find one or two quotes from the text that are most relevant to answering the question, then print them in numbered order. 
    Quotes should be reletively short. 
    Follow the example output to format your response.

    If there are no relevant quotes, print 'no quotes found'.

    Text: {text}

    Questions: {questions}

    
    Example Output: [{{'question': question, 'answer': answer, 'quotes': [list of quotes]}}]

    """
   
    prompt = closed_end_answers_template.format(text=text, questions=questions)
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def get_inclusion_scores(source_doc, summary, n):

    questions = get_questions(text=source_doc, n=n)
    questions_list = eval(questions)['questions']
    questions_str = "\n\n".join(questions_list)

    qa_source = get_answers(text=source_doc, questions=questions_str)
    qa_source_df = pd.DataFrame(eval(qa_source))
    qa_summary = get_answers(text=summary, questions=questions)
    qa_summary_df = pd.DataFrame(eval(qa_summary))
    comparison_df = pd.merge(qa_source_df, qa_summary_df, on='question', how='inner')
    comparison_df.rename(columns={"answer_x": "answer_source", "answer_y": "answer_summary"}, inplace=True)
    inclusion_score = sum(comparison_df["anwer_source"] == comparison_df["answer_summary"]) / len(comparison_df)

    return inclusion_score, comparison_df
