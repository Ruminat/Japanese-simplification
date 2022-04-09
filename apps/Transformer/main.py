import sys

from definitions import DATASET
from modules.Metrics.LanguageMetrics.bleu import getBleuScore
from modules.Metrics.LanguageMetrics.sari import getSariScore
from modules.Metrics.utils import getMetricsData
from modules.Seq2SeqTransformer.main import Seq2SeqTransformer
from utils import (getTrainedTransformer, initiatePyTorch, loadTransformer,
                   prettyPrintSentencesTranslation)


def startTransformerApp() -> None:
  initiatePyTorch()

  if ("--train" in sys.argv):
    print("\n-- TRAIN MODE --\n")
    transformer = getTrainedTransformer()
  elif ("--load" in sys.argv):
    print("\n-- LOADING THE SAVED MODEL --\n")
    transformer = loadTransformer()
  else:
    print("\n-- DEFAULT (TRAIN) MODE --\n")
    transformer = getTrainedTransformer()

  if ("--no-print" not in sys.argv):
    # -- Testing the model --
    printTransformerTests(transformer)

def printTransformerTests(transformer: Seq2SeqTransformer) -> None:
  print("\nSentences that are not in the dataset\n")

  prettyPrintSentencesTranslation(transformer, [
    "お前はもう死んでいる。",
    "知識豊富な人間は実に馬鹿である。",
    "あたしのこと好きすぎ。",
    "事実上日本の唯一の公用語である。",
    "我思う故に我あり。",
  ])

  print("\nSentences from the dataset\n")

  prettyPrintSentencesTranslation(transformer, [
    "彼は怒りに我を忘れた。",
    "ジョンは今絶頂だ。",
    "ビル以外はみなあつまった。",
    "彼はすぐに風邪をひく。",
    "彼は私のしたことにいちいち文句を言う。",
    "彼女は内気なので、ますます彼女が好きだ。",
  ])

  print("\nMetrics\n")

  print("Calculating the metrics data...")
  metricsData = getMetricsData(transformer, DATASET)
  print("Calculating the metrics...")
  blueScore = getBleuScore(metricsData)
  print(f"BLEU score: {blueScore}")
  sariScore = getSariScore(metricsData)
  print(f"SARI score: {sariScore}")
