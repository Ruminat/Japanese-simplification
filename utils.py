import torch

from definitions import (BATCH_SIZE, BETAS, DATASET, DEFAULT_MODEL_FILENAME,
                         DEVICE, DIM_FEEDFORWARD, EMB_SIZE, EPSILON,
                         LEARNING_RATE, MODELS_DIR, NHEAD, NUM_DECODER_LAYERS,
                         NUM_EPOCHS, SEED, SRC_LANGUAGE, TGT_LANGUAGE,
                         WEIGHT_DECAY)
from modules.Language.definitions import PAD_IDX
from modules.Language.utils import getSpacyTokenizer, getVocabTransform
from modules.Seq2SeqTransformer.main import Seq2SeqTransformer
from modules.Seq2SeqTransformer.utils import (initializeTransformerParameters,
                                              train)


def initiatePyTorch() -> None:
  torch.manual_seed(SEED)
  torch.cuda.empty_cache()
  print(f"Running PyTorch on {DEVICE} with seed={SEED}")

def prettyPrintSentencesTranslation(transformer: Seq2SeqTransformer, sentences: str) -> None:
  for sentence in sentences:
    prettyPrintTranslation(transformer, sentence)

def prettyPrintTranslation(transformer: Seq2SeqTransformer, sourceSentence: str) -> None:
  src = f"«{sourceSentence.strip()}»"
  result = f"«{transformer.translate(sourceSentence).strip()}»"
  print("Translating:", src, "->", result)

def loadTransformer(fileName: str = DEFAULT_MODEL_FILENAME) -> Seq2SeqTransformer:
  print("Loading Transformer...")
  transformer = torch.load(f"{MODELS_DIR}/{fileName}")
  transformer.eval()
  print("Transformer is ready to use")
  return transformer

def getTrainedTransformer(fileName: str = DEFAULT_MODEL_FILENAME) -> Seq2SeqTransformer:
  tokenize = getSpacyTokenizer(DATASET.spacyKey)
  vocab = getVocabTransform(SRC_LANGUAGE, TGT_LANGUAGE, tokenize, DATASET)

  transformer = Seq2SeqTransformer(
    batchSize = BATCH_SIZE,
    srcLanguage = SRC_LANGUAGE,
    tgtLanguage = TGT_LANGUAGE,
    numEncoderLayers = NUM_DECODER_LAYERS,
    numDecoderLayers = NUM_DECODER_LAYERS,
    embeddingSize = EMB_SIZE,
    nhead = NHEAD,
    tokenize = tokenize,
    vocab = vocab,
    dimFeedforward = DIM_FEEDFORWARD,
    device = DEVICE
  )
  print("Created the Transformer model")
  initializeTransformerParameters(transformer)
  print("Initialized parameters")

  transformer = transformer.to(DEVICE)
  lossFn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
  optimizer = torch.optim.Adam(
    transformer.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=BETAS,
    eps=EPSILON
  )
  print("Created lossFn and optimizer")

  print("Training the model...")
  train(transformer, optimizer, lossFn, NUM_EPOCHS, DATASET)
  print("The model has trained well")

  torch.save(transformer, f"{MODELS_DIR}/{fileName}")

  return transformer
