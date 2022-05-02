import tqdm
from modules.Language.definitions import UNK_IDX, TTokensSentence
from modules.Language.utils import getSpacyTokenizer, getVocab
from modules.Dataset.snowSimplifiedJapanese.main import snowSimplifiedJapaneseDataset


print("MIKOCHI")

def hasUnknown(tokens: TTokensSentence) -> bool:
  for token in tokens:
    if (token == UNK_IDX):
      return True
  return False

dataset = snowSimplifiedJapaneseDataset

tokenize = getSpacyTokenizer(dataset.spacyKey)
srcVocab = getVocab(tokenize, dataset, dataset.srcSentenceKey)
tgtVocab = getVocab(tokenize, dataset, dataset.tgtSentenceKey)

goodSentences = []
with open('./playground/wiki/result.txt', encoding="utf8", mode='r') as file:
  # reader.readlines()
  for line in tqdm.tqdm(file.readlines(), leave=False):
    sentence = line.strip()
    srcIndexes = srcVocab(tokenize(sentence))
    # tgtIndexes = tgtVocab(tokenize(sentence))
    if (not hasUnknown(srcIndexes)):
      goodSentences.append(f"{sentence},{sentence}")

print("VOT TAK VOT......", len(goodSentences))

with open('./result.txt', encoding="utf8", mode='w') as file:
  file.write("\n".join(goodSentences))
