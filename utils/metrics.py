
from collections import defaultdict
from dataclasses import dataclass, field
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import BLEUScore

@dataclass
class Metrics:
    """
    Base class for metrics
    """
    scores: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    average_scores: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    rouge: ROUGEScore = field(default_factory=ROUGEScore)
    bleu: BLEUScore = field(default_factory=BLEUScore)

    def compute(
            self,
            preds: list[str],
            target: list[str]
        )-> dict[str, float]:
        """
        Compute the metrics
        """
        rouge_score = self.rouge(preds, target)
        bleu_score = self.bleu(preds, target)

        # Convert the scores to a dictionary
        self.scores['rouge'].append(rouge_score['rouge1_fmeasure'].item())
        self.scores['bleu'].append(bleu_score.item())
        # Update the average scores
        self.update_average('rouge', rouge_score['rouge1_fmeasure'].item())
        self.update_average('bleu', bleu_score.item())

        return self.average_scores

    def update_average(
            self,
            metric: str,
            value: float
        )-> None:
        """
        Update the specified metric
        """
        self.average_scores[metric] *= (len(self.scores[metric])-1)
        self.average_scores[metric] += value
        self.average_scores[metric] /= len(self.scores[metric])
    

def compute_rouge(
        preds: list[str], 
        target: list[str],
    )-> dict[str, float]:
    """
    Compute the ROUGE score
    """
    rouge = ROUGEScore()
    
    return rouge(preds, target)

def compute_bleau(
        preds: list[str], 
        target: list[str],
    )-> dict[str, float]:
    """
    Compute the BLEU score
    """
    bleu = BLEUScore()

    return bleu(preds, target)