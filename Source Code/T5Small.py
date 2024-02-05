#### Train Version 8.4 ####
import sys
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda")
max_source_length = 4500
max_target_length = 2000

### Load Dataset ###
print("Loading and Cleaning data...")
# trainData = pd.read_csv("./MergedCSVs/train_merged_clean.csv", encoding='gb18030', engine="python")
# trainQuestion = trainData["question"].tolist()
# trainContext = trainData["subtitles"].tolist()
# trainAnswer = trainData["answer"].tolist()
# valData = pd.read_csv("./MergedCSVs/val_merged_clean.csv", encoding='gb18030', engine="python")
# valQuestion = valData["question"].tolist()
# valContext = valData["subtitles"].tolist()
# valAnswer = valData["answer"].tolist()

### Load New Dataset
import json

trainQuestion = []
trainContext = []
trainAnswer = []
with open("./MergedCSVs/train_data.json", "r") as f:
    trainData = json.load(f)  # a list of dictionaries of features
    for sample in trainData:
        trainQuestion.append(sample['question'])
        trainContext.append(sample['subtitle'])
        trainAnswer.append(sample['answer'])

valQuestion = []
valContext = []
valAnswer = []
with open("./MergedCSVs/val_data.json", "r") as f:
    valData = json.load(f)  # a list of dictionaries of features
    for sample in valData:
        valQuestion.append(sample['question'])
        valContext.append(sample['subtitle'])
        valAnswer.append(sample['answer'])

## Clean Dataset (not string and empty)
idx = 0
while idx < len(trainQuestion):
    if (not isinstance(trainQuestion[idx], str)) \
            or (not isinstance(trainContext[idx], str)) \
            or (not isinstance(trainAnswer[idx], str)) \
            or (len(trainQuestion[idx]) == 0) \
            or (len(trainContext[idx]) == 0) \
            or (len(trainAnswer[idx]) == 0):
        del trainQuestion[idx]
        del trainContext[idx]
        del trainAnswer[idx]
        idx = idx - 1
    idx = idx + 1

idx = 0
while idx < len(valQuestion):
    if (not isinstance(valQuestion[idx], str)) \
            or (not isinstance(valContext[idx], str)) \
            or (not isinstance(valAnswer[idx], str)) \
            or (len(valQuestion[idx]) == 0) \
            or (len(valContext[idx]) == 0) \
            or (len(valAnswer[idx]) == 0):
        del valQuestion[idx]
        del valContext[idx]
        del valAnswer[idx]
        idx = idx - 1
    idx = idx + 1

## Make No Answer Subtitles (Negative Samples)
# trainQuestionNegative = []
# trainContextNegative = []
# trainAnswerNegative = []

idx = 0
while idx < len(trainQuestion):
    if trainContext[idx].find(trainAnswer[idx]) < 0:  # answer not found
        # trainQuestionNegative.append(trainQuestion[idx])
        # trainContextNegative.append(trainContext[idx])
        # trainAnswerNegative.append("")
        del trainQuestion[idx]
        del trainContext[idx]
        del trainAnswer[idx]
        idx = idx - 1
    # else:
    #     if trainContext[idx].find(trainAnswer[idx]) > max_source_length * 2:        # context before answer is longer than half of max source length
    #         trainQuestionNegative.append(trainQuestion[idx])
    #         trainContextNegative.append(trainContext[idx][:trainContext[idx].find(trainAnswer[idx])])
    #         trainAnswerNegative.append("")
    #     if len(trainContext[idx]) - trainContext[idx].find(trainAnswer[idx]) - len(trainAnswer[idx]) > max_source_length * 2:           # context after answer is longer than half of max source length
    #         trainQuestionNegative.append(trainQuestion[idx])
    #         trainContextNegative.append(trainContext[idx][:trainContext[idx].find(trainAnswer[idx])])
    #         trainAnswerNegative.append("")
    idx = idx + 1

# print("{} negative samples.".format(len(trainQuestionNegative)))

## Make Sure Subtitles Contain Answer (Positive Samples)
idx = 0
while idx < len(trainQuestion):
    if len(trainContext[idx]) > max_source_length:
        if trainContext[idx].find(trainAnswer[idx]) >= 0 and len(trainAnswer[idx]) < max_source_length:
            leftIdx = trainContext[idx].find(trainAnswer[idx]) - random.randint(0, max_source_length - len(
                trainAnswer[idx]))
            if leftIdx < 0:
                leftIdx = 0
            rightIdx = leftIdx + max_source_length
            trainContext[idx] = trainContext[idx][leftIdx: rightIdx]
        elif trainContext[idx].find(trainAnswer[idx]) >= 0 and len(trainAnswer[idx]) >= max_source_length:
            trainContext[idx] = trainContext[idx][trainContext[idx].find(trainAnswer[idx]): trainContext[idx].find(
                trainAnswer[idx]) + max_source_length]
        else:
            del trainQuestion[idx]
            del trainContext[idx]
            del trainAnswer[idx]
            idx = idx - 1
    idx = idx + 1

idx = 0
while idx < len(valQuestion):
    if len(valContext[idx]) > max_source_length:
        if valContext[idx].find(valAnswer[idx]) >= 0 and len(valAnswer[idx]) < max_source_length:
            leftIdx = valContext[idx].find(valAnswer[idx]) - random.randint(0, max_source_length - len(valAnswer[idx]))
            if leftIdx < 0:
                leftIdx = 0
            rightIdx = leftIdx + max_source_length
            valContext[idx] = valContext[idx][leftIdx: rightIdx]
        elif valContext[idx].find(valAnswer[idx]) >= 0 and len(valAnswer[idx]) >= max_source_length:
            valContext[idx] = valContext[idx][valContext[idx].find(valAnswer[idx]): valContext[idx].find(
                valAnswer[idx]) + max_source_length]
        else:
            del valQuestion[idx]
            del valContext[idx]
            del valAnswer[idx]
            idx = idx - 1
    idx = idx + 1

print("{} positive samples.".format(len(trainQuestion)))

## Mix Positive and Negative Samples
# trainQuestion = trainQuestion + trainQuestionNegative
# trainContext = trainContext + trainContextNegative
# trainAnswer = trainAnswer + trainAnswerNegative

## Concat Question and Context
trainQC = []  # Question and Context are connected by '######'
for idx in range(len(trainQuestion)):
    trainQC.append("question: " + trainQuestion[idx] + "context: " + trainContext[idx])
valQC = []
for idx in range(len(valQuestion)):
    valQC.append("question: " + valQuestion[idx] + "context: " + valContext[idx])

## Shuffle Train Dataset
# trainQCA = list(zip(trainQC, trainAnswer))
# random.shuffle(trainQCA)
# trainQC, trainAnswer = zip(*trainQCA)

print("Data Preprocessing finished, {} samples are valid.".format(len(trainQC)))

### Tokenization ###

print("Tokenizing...")

## T5 ##
from transformers import T5TokenizerFast, BigBirdTokenizerFast
from transformers import AutoTokenizer, LongT5ForConditionalGeneration

# tokenizer = T5TokenizerFast.from_pretrained("lmqg/t5-small-squad-qa")
# tokenizer = T5TokenizerFast.from_pretrained("t5-large")
tokenizer = BigBirdTokenizerFast.from_pretrained("google/bigbird-roberta-large")
# tokenizer = T5TokenizerFast.from_pretrained("Mohan515/t5-small-finetuned-medical")
# tokenizer = T5TokenizerFast.from_pretrained("imxly/t5-copy-med-qa")
# tokenizer = AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")

### Train Model ###
print("Start Training!")

## T5 ##
from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("lmqg/t5-small-squad-qa")
# model = T5ForConditionalGeneration.from_pretrained("Mohan515/t5-small-finetuned-medical")
# model = T5ForConditionalGeneration.from_pretrained("imxly/t5-copy-med-qa")
# model = LongT5ForConditionalGeneration.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")
model.to(device)
model.train()

from transformers.optimization import Adafactor

optimizer = Adafactor(
    model.parameters(),
    lr=1e-4,
    eps=(1e-30, 1e-4),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False,
)

## Run Model
epoches = 3
bestLoss = 100000
bestValLoss = 100000

stepLoss = []           # every 100 samples is a step
epochValLoss = []           # every epoch for val

for epoch in range(epoches):
    print("Epoch {} starts.\n".format(epoch + 1))
    totalLoss = 0.0
    totalStepLoss = 0.0
    sampleIdx = 0
    for idx in range(len(trainQC)):
        input_ids = tokenizer(trainQC[idx], add_special_tokens=True, return_tensors="pt", truncation=True, max_length=max_source_length).input_ids.to(device)
        labels = tokenizer(trainAnswer[idx], return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_target_length).input_ids.to(device)
        if idx % 1000 == 0:
            print("Sample context {}: \n".format(sampleIdx + 1), trainContext[idx])
            print("Sample output {}: \n".format(sampleIdx + 1), tokenizer.decode((model.generate(input_ids=input_ids, decoder_start_token_id=0, max_length=max_target_length))[0], skip_special_tokens=True))
            # print("Sample gold decode {}: \n".format(sampleIdx+1), tokenizer.decode(labels[0]))
            print("Sample gold {}: \n".format(sampleIdx + 1), trainAnswer[idx])
            print("\n")
            sampleIdx = sampleIdx + 1
        loss = model(input_ids=input_ids, labels=labels).loss.to(device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        totalLoss = totalLoss + loss.item()
        totalStepLoss = totalStepLoss + loss.item()
        if idx != 0 and idx % 100 == 0:
            stepLoss.append(totalStepLoss)
            totalStepLoss = 0.0

    ## val model
    model.eval()
    totalLossVal = 0.0
    sampleIdx = 0

    with torch.no_grad():
        for idx in range(len(valQC)):
            input_ids = tokenizer(valQC[idx], add_special_tokens=True, return_tensors="pt", truncation=True, max_length=max_source_length).input_ids.to(device)
            labels = tokenizer(valAnswer[idx], return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_target_length).input_ids.to(device)
            if idx % 100 == 0:
                print("Sample Val context {}: \n".format(sampleIdx + 1), valContext[idx])
                print("Sample Val output {}: \n".format(sampleIdx + 1), tokenizer.decode((model.generate(input_ids=input_ids, decoder_start_token_id=0, max_length=max_target_length))[0]))
                print("Sample Val gold {}: \n".format(sampleIdx + 1), valAnswer[idx])
                print("\n")
                sampleIdx = sampleIdx + 1
            loss = model(input_ids=input_ids, labels=labels).loss.to(device)
            totalLossVal = totalLossVal + loss.item()
        epochValLoss.append(totalLossVal)
    print("Epoch: {} |  Train Loss: {} | Test Loss: {}".format(epoch + 1, totalLoss, totalLossVal))

    if totalLoss < bestLoss:
        if epoch+1 == 10:
            model.save_pretrained("./savePretrainedT5Small6-10")
            bestLoss = totalLoss
            bestValLoss = totalLossVal
            print("Model Saved.")
        if epoch+1 == 15:
            model.save_pretrained("./savePretrainedT5Small6-15")
            bestLoss = totalLoss
            bestValLoss = totalLossVal
            print("Model Saved.")
        if epoch+1 == 20:
            model.save_pretrained("./savePretrainedT5Small6-20")
            bestLoss = totalLoss
            bestValLoss = totalLossVal
            print("Model Saved.")
    print("--------------------------------------------------------------------------------------------------------")

print("Training finished.")

## Draw training details
x_axis = []
for step in range(len(stepLoss)):
    x_axis.append(step+1)
plt.plot(x_axis, stepLoss, 'b*--', alpha=0.5, linewidth=1, label='train')
plt.legend()
plt.xlabel('step')
plt.ylabel('loss')
plt.show()

x_axis = []
for step in range(len(epochValLoss)):
    x_axis.append(step+1)
plt.plot(x_axis, epochValLoss, 'r*-', alpha=0.5, linewidth=1, label='train')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
