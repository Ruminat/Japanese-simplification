import sys
from timeit import default_timer as timer

from apps.Transformer.definitions import (baseModelParams,
                                          fromPretrainedParams,
                                          wikiModelParams)
from modules.Dataset.snowSimplifiedJapanese.main import \
    snowSimplifiedJapaneseDataset
from modules.Metrics.LanguageMetrics.bleu import getBleuScore
from modules.Metrics.LanguageMetrics.sari import getSariScore
from modules.Metrics.utils import getMetricsData
from modules.Seq2SeqTransformer.main import Seq2SeqTransformer
from utils import (fromPretrained, getTrainedTransformer, initiatePyTorch,
                   loadTransformer, prettyPrintSentencesTranslation)


def startTransformerApp() -> None:
  initiatePyTorch()

  if ("--pretrain" in sys.argv):
    print("\n-- PRETRAIN MODE --\n")
    transformer = getTrainedTransformer(wikiModelParams)
  elif ("--from-pretrained" in sys.argv):
    print("\n-- FROM PRETRAINED MODE --\n")
    pretrainedTransformer = loadTransformer(wikiModelParams.fileName)
    transformer = fromPretrained(pretrainedTransformer, fromPretrainedParams)
  elif ("--train" in sys.argv):
    print("\n-- TRAIN MODE --\n")
    transformer = getTrainedTransformer(baseModelParams)
  elif ("--load" in sys.argv):
    print("\n-- LOADING THE SAVED MODEL --\n")
    transformer = loadTransformer(fromPretrainedParams.fileName)
  else:
    print("""
      Couldn't parse the provided command.
      You can type
        `python main.py --help`
      to get the list of available commands.
    """)

  if ("--no-print" not in sys.argv):
    # -- Testing the model --
    printTransformerTests(transformer)

def printTransformerTests(transformer: Seq2SeqTransformer) -> None:
  print("\nSentences that are not in the dataset\n")

  prettyPrintSentencesTranslation(transformer, [
    "お前はもう死んでいる。",
    "知識豊富な人間は実に馬鹿である。",
    "あたしのこと好きすぎ。",
    "事実上日本の唯一の公用語である。",
    "我思う故に我あり。",
  ])

  print("\nSentences from the dataset\n")

  prettyPrintSentencesTranslation(transformer, [
    # "彼は怒りに我を忘れた。",
    # "ジョンは今絶頂だ。",
    # "ビル以外はみなあつまった。",
    # "彼はすぐに風邪をひく。",
    # "彼は私のしたことにいちいち文句を言う。",
    # "彼女は内気なので、ますます彼女が好きだ。",
    # "英仏海峡を泳ぎ渡るのに成功した最初の人はウェッブ船長でした。",
    # "彼女はほほえんで僕のささやかなプレゼントを受け取ってくれた。",
    # "料理はそうおいしくはなかったけれど、その他の点ではパーティーは成功した。",
    # "その絵の値段は１０ポンドです。","その絵の価格はポンドです。",
    "彼を軽く見ないほうがいいよ",
    "家の周りには囲いがしてある",
    "私たちは母校を訪れた",
    "ガソリンを積んだトラックが門に衝突して爆発した",
    "彼は冬の間中スキーに出かけた",
    "記者：例を１つあげてくださいますか",
    "彼らは援助を必要としている",
    "その問題は考慮に値しない",
    "何故私は彼らを撃つ、教えてくれ、彼らが何をしたのだ",
    "彼は医者になり無医村へ行こうと決意した",
    "そのスキャンダルはやがてみんなに知れ渡るだろう",
    "聴衆はコンサートが始まるのをどうにも待ちきれなかった",
    "彼女の濃いブルーの瞳がとても印象的だった",
    "こんなところで君に出会うとは",
    "きのうは和牛が特売だった",
    "彼は大きなあくびをした",
    "君はあの委員会のメンバーですか",
    "もし天気がよければ、ピクニックに行こう",
    "私は彼女の意志にそむいて彼女にピアノをひかせた",
    "つばさの客車は何両ですか",
    "何かで妥協しなくちゃ",
    "父はたいてい日曜日はゴルフをしている",
    "恒久的な平和など幻想に過ぎない",
    "この会社は立派な組織をもっている",
    "健康より貴重のものは何もない",
    "私はその仕事を全力を尽くしてやった",
    "彼女はその豪華な部屋に目のくらむ思いがした",
    "入場料はただだった",
    "君にここで会うとは夢にも思わなかった",
  ])

  print("\nMetrics\n")

  print("Calculating the metrics data...")
  startTime = timer()
  metricsData = getMetricsData(transformer, snowSimplifiedJapaneseDataset)
  endTime = timer()
  print(f"Getting the metrics data took {endTime - startTime}s")
  print("Calculating the metrics...")
  blueScore = getBleuScore(metricsData)
  print(f"BLEU score: {blueScore}")
  sariScore = getSariScore(metricsData)
  print(f"SARI score: {sariScore}")
