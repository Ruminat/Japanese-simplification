from dataclasses import dataclass
from typing import Optional

from spacy import Vocab

from modules.Dataset.definitions import TJapaneseSimplificationDataset

from definitions import DEFAULT_MODEL_FILENAME, DEVICE
from modules.Language.definitions import TTokenizer


@dataclass
class TSeq2SeqTransformerParameters:
  dataset: TJapaneseSimplificationDataset
  maxEpochs: int = 30
  batchSize: int = 64
  attentionHeadsCount: int = 8
  encoderLayersCount: int = 6
  decoderLayersCount: int = 6
  embeddingSize: int = 512
  feedForwardSize: int = 512
  dropout: float = 0.1
  device: str = DEVICE
  fileName: str = DEFAULT_MODEL_FILENAME
  customTokenizer: Optional[TTokenizer] = None
  customVocab: Optional[Vocab] = None
