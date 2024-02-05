import torch
from transformers import AutoTokenizer, LongformerForSequenceClassification
import pandas as pd
import json
import sys

device = torch.device("cuda")

### Load Dataset ###
print("Loading and Cleaning data...")
trainData = pd.read_csv("./MergedCSVs/train_merged.csv", encoding='utf-8', engine="python")
trainQuestion = trainData["question"].tolist()
trainContext = trainData["subtitles"].tolist()
trainAnswer = trainData["answer"].tolist()
trainTimeGold = trainData["timestamp"].tolist()
timeStampOriginal = trainData["subtitle_timestamps"].tolist()

## clean dataset (not string)
idx = 0
while idx < len(trainQuestion):
    if (not isinstance(trainQuestion[idx], str)) \
            or (not isinstance(trainContext[idx], str)) \
            or (not isinstance(trainAnswer[idx], str)):
        del trainQuestion[idx]
        del trainContext[idx]
        del trainAnswer[idx]
        del trainTimeGold[idx]
        del timeStampOriginal[idx]
        idx = idx - 1
    idx = idx + 1
for row in range(len(trainQuestion)):  # Unify to lower case
    trainQuestion[row] = trainQuestion[row].lower()
    trainContext[row] = trainContext[row].lower()

## clean timestamp data
trainTimeStamp = []  # to list of dict
for row in range(len(timeStampOriginal)):
    timeStampRow = timeStampOriginal[row]
    for loc in range(len(timeStampRow)):
        if timeStampRow[loc] == "\"":  # change to """
            timeStampRow = timeStampRow[:loc] + "\'" + timeStampRow[(loc + 1):]
        if loc == 0 or loc == (len(timeStampRow) - 1):
            continue
        elif loc == 1 or loc == (len(timeStampRow) - 2):
            timeStampRow = timeStampRow[:loc] + "\"" + timeStampRow[(loc + 1):]
        elif timeStampRow[loc] == "'":
            if (timeStampRow[loc + 1] == "(") \
                    or ((timeStampRow[loc - 1] == ")") & (timeStampRow[loc + 1] == ":")) \
                    or ((timeStampRow[loc - 2] == ":") & (timeStampRow[loc - 3] == "\"")) \
                    or ((timeStampRow[loc + 1] == ",") & (timeStampRow[loc + 4] == "(")):
                timeStampRow = timeStampRow[:loc] + "\"" + timeStampRow[(loc + 1):]
        elif (timeStampRow[loc] == "\\") & (timeStampRow[loc + 1] != "n"):
            timeStampRow = timeStampRow[:loc] + "/" + timeStampRow[(loc + 1):]
    trainTimeStamp.append(json.loads(timeStampRow))
for row in range(len(trainTimeStamp)):
    for key in list(trainTimeStamp[row].keys()):  # remove strange substrings
        trainTimeStamp[row][key] = trainTimeStamp[row][key].lower()
        trainTimeStamp[row][key] = trainTimeStamp[row][key].replace("/xa0", " ")

## make labels
trainLabel = []
for row in range(len(trainTimeGold)):
    stampList = []
    stampList.append(int(trainTimeGold[row][1: trainTimeGold[row].find(",")]))
    stampList.append(int(trainTimeGold[row][trainTimeGold[row].find(",") + 2: len(trainTimeGold[row]) - 1]))
    trainLabel.append(stampList)

trainSubtitleLabel = []
dirty = 0
row = 0
while row < len(trainLabel):
    dirtyStart = 0  # -1 if Start is found, 0 not processed yet, 1 if start is not found
    dirtyEnd = 0  # -1 if end is found, 0 not processed yet, 1 if end is not found
    startIdx = 0
    endIdx = 0
    contextTemp = trainContext[row]  # store the context of row
    cutLength = 0  # record how long context has been canceled
    for key in list(trainTimeStamp[row].keys()):
        startEnd = []
        startEnd.append(float(key[1: key.find(",")]))
        startEnd.append(float(key[key.find(",") + 2: len(key) - 1]))
        if ((dirtyStart == 0) & (trainLabel[row][0] >= startEnd[0]) & (trainLabel[row][0] <= startEnd[1])) or (
                (dirtyEnd == 0) & (trainLabel[row][1] >= startEnd[0]) & (trainLabel[row][1] <= startEnd[1])):  # start or end time is found
            if (dirtyStart == 0) & (trainLabel[row][0] >= startEnd[0]) & (
                    trainLabel[row][0] <= startEnd[1]):  # start time is found
                cutLength = len(trainContext[row]) - len(contextTemp)
                startIdx = contextTemp.find(trainTimeStamp[row][key])
                if startIdx < 0:  # start can not be found
                    dirtyStart = 1
                    startIdx = 0
                    break
                else:
                    startIdx = startIdx + cutLength
                    dirtyStart = -1
                    contextTemp = contextTemp[(contextTemp.find(trainTimeStamp[row][key]) + len(trainTimeStamp[row][key])):]
            if (dirtyEnd == 0) & (trainLabel[row][1] >= startEnd[0]) & (
                    trainLabel[row][1] <= startEnd[1]):  # end time is found
                cutLength = len(trainContext[row]) - len(contextTemp)
                endIdx = contextTemp.find(trainTimeStamp[row][key])
                if endIdx < 0:  # end can not be found
                    dirtyEnd = 1
                    endIdx = len(trainContext[row])
                else:
                    endIdx = endIdx + cutLength + len(trainTimeStamp[row][key])
                    dirtyEnd = -1
                break
        else:  # middle or no use part
            if (contextTemp.find(trainTimeStamp[row][key]) >= 0) & (contextTemp.find(trainTimeStamp[row][key]) < 5):
                contextTemp = contextTemp[(contextTemp.find(trainTimeStamp[row][key]) + len(trainTimeStamp[row][key])):]  # cut off no use substring in the front of context
    if (dirtyStart == 1) or (dirtyEnd == 1):  # none of start and end is found
        dirty = dirty + 1
        del trainQuestion[row]
        del trainContext[row]
        del trainAnswer[row]
        del trainLabel[row]
        del trainTimeStamp[row]
    else:
        if dirtyStart == 0:
            startIdx = 0
        if dirtyEnd == 0:
            endIdx = len(trainContext[row])
        trainSubtitleLabel.append([startIdx, endIdx])
        row = row + 1

## make batches
# batchSize = 16
max_subtitle_length = 1900
trainQCBatches = []
trainQCBatch = []
trainLabelBatches = []
trainLabelBatch = []

batchLength = 0
for row in range(len(trainSubtitleLabel)):
    if len(trainContext[row]) > max_subtitle_length:
        for piece in range(int(len(trainContext[row]) / max_subtitle_length) + 1):
            if piece == int(len(trainContext[row]) / max_subtitle_length):
                contextPiece = trainContext[row][piece * max_subtitle_length:]
            else:
                contextPiece = trainContext[row][(piece * max_subtitle_length): ((piece + 1) * max_subtitle_length)]
            QCPiece = trainQuestion[row] + "\n######\n" + contextPiece
            trainQCBatch.append(QCPiece)

            if (piece * max_subtitle_length < trainSubtitleLabel[row][0]) and (piece * max_subtitle_length + len(contextPiece) > trainSubtitleLabel[row][0]):
                trainLabelBatch.append(1)
            elif (piece * max_subtitle_length < trainSubtitleLabel[row][1]) and (piece * max_subtitle_length + len(contextPiece) > trainSubtitleLabel[row][1]):
                trainLabelBatch.append(1)
            else:
                trainLabelBatch.append(0)

            # batchLength = batchLength + 1
            # if batchLength == batchSize:                 ## current QC batch is full
            #     trainQCBatches.append(trainQCBatch)
            #     trainLabelBatches.append(trainLabelBatch)
            #     trainQCBatch = []
            #     trainLabelBatch = []
            #     batchLength = 0

# if len(trainQCBatch) != 0:              ## some cases remain
#     trainQCBatches.append(trainQCBatch)
#     trainLabelBatches.append(trainLabelBatch)

print("Data loading finished, {} answers not found, {} valid cases.".format(dirty, len(trainSubtitleLabel)))

### Tokenization ###

print("Tokenizing...")

tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')

### Train Model ###
model = LongformerForSequenceClassification.from_pretrained('./savePretrainedJudgeQA-100', num_labels=2)
model.to(device)
model.train()

from transformers.optimization import Adafactor, AdafactorSchedule

optimizer = Adafactor(
    model.parameters(),
    lr=1e-2,
    eps=(1e-30, 1e-2),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False,
)

## Run Model
print("Start training!")
epoches = 100
sampling = 10000
bestLoss = 10000000

max_context_length = 2000

for epoch in range(epoches):
    totalLoss = 0.0
    sampleIdx = 0
    for idx in range(len(trainQCBatch)):
        inputs = tokenizer(trainQCBatch[idx], return_tensors="pt", padding='max_length', truncation=True, max_length=max_context_length).to(device)
        if idx % sampling == 0:
            prediction = model(**inputs).logits.argmax().item()
            print("Sample {}:\n    Prediction:\n {}\n    Label:\n {}\n".format(sampleIdx + 1, prediction, trainLabelBatch[idx]))
            sampleIdx = sampleIdx + 1
        labels = torch.tensor(trainLabelBatch[idx]).to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totalLoss = totalLoss + loss.item()
    print("{} epoch  |  loss: {}".format(epoch + 1, totalLoss))
    if totalLoss < bestLoss:
        model.save_pretrained("./savePretrainedJudgeQA-100")
        bestLoss = totalLoss
        print("Model saved.")
    print("----------------------------------------------------------------------------------------------------")

## save model
