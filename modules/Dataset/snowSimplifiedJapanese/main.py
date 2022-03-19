import functools
from datasets import concatenate_datasets, load_dataset

from modules.Dataset.main import MyDataset

SNOW_DATASET_SHITE = "snow_simplified_japanese_corpus"

SNOW_T15 = "snow_t15"
SNOW_T23 = "snow_t23"

VALIDATION_PERCENTAGE = 5

@functools.cache
def getTrainSplit():
  t15Dataset = load_dataset(SNOW_DATASET_SHITE, SNOW_T15, split=f"train[{VALIDATION_PERCENTAGE}%:]")
  t23Dataset = load_dataset(SNOW_DATASET_SHITE, SNOW_T23, split=f"train")
  return concatenate_datasets([t15Dataset, t23Dataset])

@functools.cache
def getValidationSplit():
  return load_dataset(SNOW_DATASET_SHITE, SNOW_T15, split=f"train[:{VALIDATION_PERCENTAGE}%]")

snowSimplifiedJapaneseDataset = MyDataset(getTrainSplit=getTrainSplit, getValidationSplit=getValidationSplit)
