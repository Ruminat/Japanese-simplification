import functools

from datasets import concatenate_datasets, load_dataset
from modules.Dataset.definitions import TJapaneseSimplificationDataset
from modules.Dataset.snowSimplifiedJapanese.main import SNOW_DATASET, SNOW_T23, getValidationSplit

fileName = "modules/Dataset/wikipediaJa/data/wikipediaJp.csv"

VALIDATION_TEST_PERCENT = 50
# TEST_PERCENT = 5

@functools.cache
def getTrainSplit():
  snowDataset = load_dataset(SNOW_DATASET, SNOW_T23, split=f"train[40%:]")
  wikiDataset = load_dataset("csv", data_files=fileName, split=f"train")
  dataset = concatenate_datasets([snowDataset, wikiDataset])
  print("loaded the train split (wikiJa)", dataset.num_rows)
  return dataset

# @functools.cache
# def getValidationSplit():
#   dataset = load_dataset("csv", data_files=fileName, split=f"train[99%:]")
#   print("loaded the validation split (wikiJa)", dataset.num_rows)
#   return dataset

wikipediaJpDataset = TJapaneseSimplificationDataset(
  getTrainSplit=getTrainSplit,
  getValidationSplit=getValidationSplit,
  srcSentenceKey="original_ja",
  tgtSentenceKey="simplified_ja"
)
