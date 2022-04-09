import functools

from datasets import load_dataset
from modules.Dataset.definitions import TJapaneseSimplificationDataset

fileName = "modules/Dataset/wikipediaJa/data/wikipediaJa.csv"

VALIDATION_TEST_PERCENT = 50
# TEST_PERCENT = 5

@functools.cache
def getTrainSplit():
  dataset = load_dataset("csv", data_files=fileName, split=f"train[:1%]")
  print("loaded the train split (wikiJa)", dataset.num_rows)
  return dataset

@functools.cache
def getValidationSplit():
  dataset = load_dataset("csv", data_files=fileName, split=f"train[99%:]")
  print("loaded the validation split (wikiJa)", dataset.num_rows)
  return dataset

wikipediaJaDataset = TJapaneseSimplificationDataset(
  getTrainSplit=getTrainSplit,
  getValidationSplit=getValidationSplit,
  srcSentenceKey="sentence",
  srcSentenceKey="sentence"
)
