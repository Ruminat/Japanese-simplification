import torch
import tqdm
from modules.Dataset.definitions import TJapaneseSimplificationDataset
from torch.utils.data import DataLoader
from modules.Language.utils import formatSentence

from modules.Metrics.definitions import TMetricsData

def getMetricsData(model: torch.nn, dataset: TJapaneseSimplificationDataset) -> TMetricsData:
  result = TMetricsData(
    srcSample = [],
    tgtSample = [],
    srcTokens = [],
    tgtTokens = [],
    translation = [],
    translationTokens = [],
  )
  testSplit = dataset.getTestSplit()
  testDataloader = DataLoader(testSplit)
  for datasetRow in tqdm.tqdm(testDataloader, leave=False):
    try:
      srcSample = formatSentence(datasetRow[dataset.srcSentenceKey][0])
      tgtSample = formatSentence(datasetRow[dataset.tgtSentenceKey][0])

      srcTokens = model.tokenize(srcSample)
      tgtTokens = model.tokenize(tgtSample)

      translation = model.translate(srcSample)
      if (srcSample[-1] == "。"):
        translation += srcSample[-1]
      translationTokens = model.tokenize(translation)

      result.srcSample.append(srcSample)
      result.tgtSample.append([tgtSample])
      result.srcTokens.append(srcTokens)
      result.tgtTokens.append([tgtTokens])
      result.translation.append(translation)
      result.translationTokens.append(translationTokens)
    except Exception as error:
      print("getMetricsData error:", error)
  return result
