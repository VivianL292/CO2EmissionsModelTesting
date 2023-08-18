# CO2EmissionsModelTesting

Loads SQuAD dataset to train large language models and uses the CodeCarbon package's CO2 emissions tracker to measure CO2 emissions and RAM usage during training. They are then evaluated on random samples from two datasets.

# Random Sample Evaluation Experiment

This repository contains code for conducting a random sample evaluation experiment with the SQuAD and AdversarialQA using Python. The previously-mentioned models trained on SQuAD data, after making inferences on samples from the SQuAD and AdversarialQA dataset, are then evaluated on the cosine similarity, STS (BERT-Embedding), STS (OpenAI-Embedding), and STS (Palm-Embedding) metrics. The experiment involves the following steps:

1. **Random Sampling**: A random set of indices is generated using `numpy` to select a subset of data from the "train" dataset. This is achieved using the `np.random.randint(len(dataset["train"]), size=100)` function, which generates 100 random integers as indices.

2. **Processing Samples**: For each randomly selected index `i`, the corresponding sample from the "train" dataset is retrieved. The sample consists of a context and a question. If either the context or the question has a length less than 2, a default prediction of "N/A" is assigned. The sample is then processed for prediction.

3. **Prediction**: The function `answer_question(prompt, question)` is used to predict an answer based on the given context and question. If the prediction has a length less than 2, it is set to "no output found".

4. **Ground Truth**: The ground truth answer is extracted from the sample's answers and stored as `ground_truth`.

5. **Comparing Answers**: Four different similarity scores are calculated between the predicted answer and the ground truth:
   - `score1`: Cosine similarity between the prediction and ground truth.
   - `score2`: STS (BERT-Embedding) between the prediction and ground truth.
   - `score3`: STS (OpenAI-Embedding) between the prediction and ground truth.
   - `score4`: STS (Palm-Embedding) between the prediction and ground truth.

6. **Collecting Scores**: The calculated similarity scores are collected into separate lists:
   - `scores_1`: List containing `score1` values.
   - `scores_2`: List containing `score2` values.
   - `scores_3`: List containing `score3` values.
   - `scores_4`: List containing `score4` values.

## Usage

To run the experiment on your own dataset, follow these steps:

1. Replace `dataset` with your own dataset object containing the "train" dataset.
2. Implement the `answer_question(prompt, question)` function to provide answer predictions based on context and question.
3. Replace the placeholder API keys in `calculate_sts_openai_score` and `calculate_sts_palm_score` functions with your actual API keys.

To measure CO2 emissions for your own model, replace `model_name` with the appropriate model and `tokenizer` with the appropriate tokenizer.

## Motivation

This experiment was inspired by the need to quantify the CO2 emissions of prominent large language models. It also aims to demonstrate the models' performance on a variety of metrics, allowing for the analysis of the costs and benefits of these models. It is through this evaluation that we may present future courses of action to mitigate the carbon emissions of model training.
