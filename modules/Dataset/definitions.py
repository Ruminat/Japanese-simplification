from dataclasses import dataclass
from typing import Callable, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from modules.Language.definitions import SPACY_JP

TDataset = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]

TDatasetFn = Callable[[], TDataset]

@dataclass
class TDatasetBase:
  getTrainSplit: TDatasetFn
  getValidationSplit: Optional[TDatasetFn] = None
  getTestSplit: Optional[TDatasetFn] = None

  def iterateOverSplits(self):
    for row in self.getTrainSplit():
      yield row
    if (self.getValidationSplit is not None):
      for row in self.getValidationSplit():
        yield row
    if (self.getTestSplit is not None):
      for row in self.getTestSplit():
        yield row

@dataclass
class TSimplificationDataset(TDatasetBase):
  srcSentenceKey: str = ""
  tgtSentenceKey: str = ""

@dataclass
class TJapaneseSimplificationDataset(TSimplificationDataset):
  spacyKey: str = SPACY_JP
