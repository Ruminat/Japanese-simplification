## Japanese simplificatin system

This is a repository with a Transformer model for simplifying Japanese sentences.
Made by Vlad Furman (https://github.com/Ruminat).

## Setting up the environment and installing dependencies

You need [Python](https://www.python.org/downloads/) and [pyenv](https://github.com/pyenv/pyenv) in order to use this model.

You can install `pyenv` with:
```bash
pip install virtualenv
```

Now you need to create a `pyenv` environment by executing the following command:
```bash
python -m venv env
```

After that you need to activate it:
```bash
env\Scripts\activate
```

And finally install the dependencies:
```bash
pip install -r requirements.txt
```

## How to use the model

The main code is placed in `./main.py`. You can train the model with the following command (don't forget that you need to be in the `pyenv`):
```bash
python main.py --train
```

If you don't want to train the model yourself, you can download a trained model from https://github, place it in the `./build` directory and run:
```bash
python main.py --load
```

Training the model with `python main.py --train` will replace the built model file in `./build`, so you might want to backup the model files.
Or you can change the build file name in `./definitions.py`.

You can start a server with a `/processJapaneseText` URL for simplification:
```bash
python main.py --server
```

You can run `main.py` with the following flags:
- `--help` will print this README.md,
- `--version` or `--v` will print the current app verion.
