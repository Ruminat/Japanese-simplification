from typing import List
from spacy import Vocab
import torch
import torch.nn as nn
from modules.Embedding.main import TokenEmbedding
from modules.Language.definitions import BOS_IDX, BOS_SYMBOL, EOS_SYMBOL, TTextTransformer, TTokenizer
from modules.Language.utils import tensorTransform
from modules.PositionalEncoding.main import PositionalEncoding
from modules.Seq2SeqTransformer.utils import greedyDecode
from torch import Tensor

from torch.nn import Transformer


# The final model to be trained
class Seq2SeqTransformer(nn.Module):
  def __init__(
    self,
    batchSize: int,
    srcLanguage: str,
    tgtLanguage: str,
    numEncoderLayers: int,
    numDecoderLayers: int,
    embeddingSize: int,
    nhead: int,
    tokenize: TTokenizer,
    vocab: dict[str, Vocab],
    dimFeedforward: int = 512,
    dropout: float = 0.1,
    device: torch.device = torch.device("cpu")
  ):
    super(Seq2SeqTransformer, self).__init__()
    self.transformer = Transformer(
      d_model=embeddingSize,
      nhead=nhead,
      num_encoder_layers=numEncoderLayers,
      num_decoder_layers=numDecoderLayers,
      dim_feedforward=dimFeedforward,
      dropout=dropout
    )

    self.srcLanguage = srcLanguage
    self.tgtLanguage = tgtLanguage
    self.tokenize = tokenize
    self.vocab = vocab

    self.batchSize = batchSize
    self.device = device

    srcVocabSize = len(vocab[self.srcLanguage])
    tgtVocabSize = len(vocab[self.tgtLanguage])

    self.generator = nn.Linear(embeddingSize, tgtVocabSize)
    self.srcEmbedding = TokenEmbedding(srcVocabSize, embeddingSize)
    self.tgtEmbedding = TokenEmbedding(tgtVocabSize, embeddingSize)
    self.positionalEncoding = PositionalEncoding(embeddingSize, dropout=dropout)

  def srcTextTransform(self, text: str) -> Tensor:
    return tensorTransform(self.vocab[self.srcLanguage](self.tokenize(text)))

  def tgtTextTransform(self, text: str) -> Tensor:
    return tensorTransform(self.vocab[self.tgtLanguage](self.tokenize(text)))

  def forward(
    self,
    src: Tensor,
    trg: Tensor,
    srcMask: Tensor,
    tgtMask: Tensor,
    srcPaddingMask: Tensor,
    tgtPaddingMask: Tensor,
    memory_key_padding_mask: Tensor
  ):
    srcEmbedding = self.positionalEncoding(self.srcEmbedding(src))
    tgtEmbedding = self.positionalEncoding(self.tgtEmbedding(trg))
    outs = self.transformer(
      srcEmbedding,
      tgtEmbedding,
      srcMask,
      tgtMask,
      None,
      srcPaddingMask,
      tgtPaddingMask,
      memory_key_padding_mask
    )
    return self.generator(outs)

  def encode(self, src: Tensor, srcMask: Tensor):
    return self.transformer.encoder(
      self.positionalEncoding(self.srcEmbedding(src)),
      srcMask
    )

  def decode(self, tgt: Tensor, memory: Tensor, tgtMask: Tensor):
    return self.transformer.decoder(
      self.positionalEncoding(self.tgtEmbedding(tgt)),
      memory,
      tgtMask
    )

  # Method for translating from srcLanguage to tgtLangauge (Japanese -> simplified Japanese)
  def translate(self, srcSentence: str):
    self.eval()
    src = self.srcTextTransform(srcSentence).view(-1, 1)
    numTokens = src.shape[0]
    srcMask = (torch.zeros(numTokens, numTokens)).type(torch.bool)
    tgtTokens = greedyDecode(
      self,
      src,
      srcMask,
      maxLen=numTokens + 5,
      startSymbol=BOS_IDX,
      device=self.device
    ).flatten()
    tokensToLookUp = list(tgtTokens.cpu().numpy())
    tokens = self.vocab[self.tgtLanguage].lookup_tokens(tokensToLookUp)

    return self.tokensToText(tokens)

  # Turns a list of tokens into a single string
  # ["what", "is", "love"] -> "what is love"
  def tokensToText(self, tokens: List[str]) -> str:
    result = ""
    for token in tokens:
      if token == BOS_SYMBOL or token == EOS_SYMBOL:
        continue
      if token == "。":
        break
      result += token
    return result
