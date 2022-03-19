from modules.Dataset.definitions import TDatasetFn


class MyDataset:
  def __init__(self, getTrainSplit: TDatasetFn, getValidationSplit: TDatasetFn):
    self.getTrainSplit = getTrainSplit
    self.getValidationSplit = getValidationSplit
