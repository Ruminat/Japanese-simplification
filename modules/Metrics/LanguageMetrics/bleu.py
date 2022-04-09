from datasets import load_metric
from modules.Metrics.definitions import TMetricsData

bleu = load_metric("bleu")

def getBleuScore(metricsData: TMetricsData) -> float:
  return bleu.compute(
    predictions=metricsData.translationTokens,
    references=metricsData.tgtTokens
  )["bleu"] * 100
