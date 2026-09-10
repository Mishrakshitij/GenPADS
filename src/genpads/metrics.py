"""Text metrics reported separately from model likelihood and task success."""


def generation_metrics(references, predictions):
    import sacrebleu
    from nltk.translate.meteor_score import meteor_score
    from nltk.translate.nist_score import corpus_nist
    from rouge_score.rouge_scorer import RougeScorer
    if not references or len(references) != len(predictions):
        raise ValueError("Metrics require nonempty, aligned references and predictions")
    reference_sets = [[item] if isinstance(item, str) else list(item) for item in references]
    if any(not items for items in reference_sets):
        raise ValueError("Every prediction needs at least one reference")
    reference_tokens = [[text.split() for text in items] for items in reference_sets]
    prediction_tokens = [text.split() for text in predictions]
    scorer = RougeScorer(["rouge2"], use_stemmer=True)
    try:
        nist = corpus_nist(reference_tokens, prediction_tokens, n=5)
    except (ValueError, ZeroDivisionError):
        nist = None  # Undefined when the corpus cannot support 5-grams.
    return {
        "bleu": sacrebleu.corpus_bleu(predictions, [
            [items[i] if i < len(items) else None for items in reference_sets]
            for i in range(max(map(len, reference_sets)))
        ]).score / 100,
        "nist": nist,
        "meteor": sum(meteor_score(ref, pred) for ref, pred in zip(reference_tokens, prediction_tokens)) / len(references),
        "rouge2_f1": sum(max(scorer.score(ref, pred)["rouge2"].fmeasure for ref in refs)
                          for refs, pred in zip(reference_sets, predictions)) / len(references),
    }
