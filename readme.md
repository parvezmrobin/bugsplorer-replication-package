# Bugsplorer Replication Package

## Environment Setup

First, make sure you have CUDA 11.6 and Python 3.10 installed.
Then, install the required packages by running the following command:
```shell
pip install -r requirements.txt
```

## Download and Prepare the Dataset
Download the Defectors dataset from [here](https://zenodo.org/record/7708984#.ZEYdOXbMJD8)
and LineDP dataset from [here](https://github.com/awsm-research/line-level-defect-prediction/tree/master/Dataset).
Put both of them inside the `dataset` folder.
Then, run the `preprocess_data.py` script inside the `script` folder to prepare the dataset.

## Run the Experiments

### Train A Model
To train a model, run the following command:
```shell
python -m src.bug_prediction.BugPredictorTrainer \
--model_type=roberta \
--config_name=huggingface/CodeBERTa-small-v1 \
--tokenizer_name=huggingface/CodeBERTa-small-v1 \
--encoder_type=line \
--dataset_path=dataset/defectors/line_bug_prediction_splits/random \
--batch_size=16 \
--num_epoch=20 \
--cache_dir=cache/roberta-defectors-line-random \
--checkpoint_dir=checkpoints/roberta-defectors-line-random \
--class_weight=100 
```

### Test the Trained Model
To test the trained model, run the following command:
```shell
python -m src.bug_prediction.BugPredictionTester \
--model_type=roberta \
--config_name=huggingface/CodeBERTa-small-v1 \
--tokenizer_name=huggingface/CodeBERTa-small-v1 \
--model_name=checkpoints/103.train-bugsplorer-linedp-line-time-w_100-gpu_2-b_16 \
--encoder_type=line \
--dataset_path=dataset/defectors/line_bug_prediction_splits/random \
--batch_size=16 \
--cache_dir=cache/roberta-defectors-line-random \
--checkpoint_dir=checkpoints/roberta-defectors-line-random \
--output_path=outputs/roberta-defectors-line-random 
```
