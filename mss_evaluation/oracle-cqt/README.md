# Oracle Source Separation Methods

## Installation

Install `pipenv` using `pip install pipenv`. Then run

```
pipenv install
```

in the source folder to install all python requirements. Alternatively you can use `pip install -r requirements.txt` to install the requirements using `pip` instead of `pipenv`.

## Usage

Each Oracle method comes with its a command line argument parser. To run one of the method just numerically

```
python METHOD_NAME.py --eval_dir ./Evaluation_Dir --audio_dir ./Audio_Dir
```

you can omit either `--eval_dir` or `audio_dir` (or both) to not write out the audio_files to disk or do not do the evaluation.
