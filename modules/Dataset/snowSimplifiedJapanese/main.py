import functools

from datasets import load_dataset, logging
from modules.Dataset.definitions import TJapaneseSimplificationDataset

logging.set_verbosity_error()

SNOW_DATASET = "snow_simplified_japanese_corpus"

SNOW_T15 = "snow_t15"
SNOW_T23 = "snow_t23"

VALIDATION_TEST_PERCENT = 50
# TEST_PERCENT = 5

@functools.cache
def getTrainSplit():
  dataset = load_dataset(SNOW_DATASET, SNOW_T15, split=f"train")
  # dataset = load_dataset(SNOW_DATASET, SNOW_T15, split=f"train[:100]")
  print("loaded the train split (SNOW)", dataset.num_rows)
  return dataset

@functools.cache
def getValidationSplit():
  dataset = load_dataset(SNOW_DATASET, SNOW_T23, split=f"train[:20%]")
  # dataset = load_dataset(SNOW_DATASET, SNOW_T23, split=f"train[200:300]")
  print("loaded the validation split (SNOW)", dataset.num_rows)
  return dataset

@functools.cache
def getTestSplit():
  dataset = load_dataset(SNOW_DATASET, SNOW_T23, split=f"train[20%:40%]")
  # dataset = load_dataset(SNOW_DATASET, SNOW_T23, split=f"train[400:500]")
  print("loaded the test split (SNOW)", dataset.num_rows)
  return dataset

snowSimplifiedJapaneseDataset = TJapaneseSimplificationDataset(
  getTrainSplit=getTrainSplit,
  getValidationSplit=getValidationSplit,
  getTestSplit=getTestSplit,
  srcSentenceKey="original_ja",
  tgtSentenceKey="simplified_ja"
)
