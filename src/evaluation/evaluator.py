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