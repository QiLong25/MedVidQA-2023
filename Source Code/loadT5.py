### Test Model ###

import torch
import pandas as pd
device = torch.device("cuda")
from transformers import T5TokenizerFast
from transformers import T5ForConditionalGeneration
from transformers import BertTokenizerFast, BertModel
import json
import sys
import torch.nn.functional as F

### Generate Output ###
testIdx = 2

valData = pd.read_csv("./MergedCSVs/val_merged.csv", encoding='utf-8', engine="python")
valQuestion = valData["question"].tolist()[testIdx]
valContext = valData["subtitles"].tolist()[testIdx]
valGoldAnswer = valData["answer"].tolist()[testIdx]
valTimeGold = valData["timestamp"].tolist()[testIdx]
timeStampOriginal = valData["subtitle_timestamps"].tolist()[testIdx]
valQC = valQuestion[testIdx] + " ###### " + valContext[testIdx]

tokenizer = T5TokenizerFast.from_pretrained("t5-small")
t_model = T5ForConditionalGeneration.from_pretrained("./savePretrainedT5").to(device)
t_model.eval()

max_source_length = 2000
max_target_length = 2000

# simTokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
simTokenizer = T5TokenizerFast.from_pretrained("t5-small")
# simModel = BertModel.from_pretrained("bert-base-uncased").to(device)
simModel = T5ForConditionalGeneration.from_pretrained("./savePretrainedT5").to(device)
simModel.eval()

simScoreStart = {}
simScoreSelect = {}

with torch.no_grad():
    if len(valQC) > max_source_length:
        textGenerate = ""
        for segment in range(int(len(valQC) / max_source_length) + 1):
            if segment == int(len(valQC) / max_source_length):
                segQC = valQuestion + ' ###### ' + valQC[segment * max_source_length: len(valQC)]
            else:
                segQC = valQuestion + ' ###### ' + valQC[segment * max_source_length: (segment + 1) * max_source_length]

            input_ids = tokenizer(segQC, add_special_tokens=True, return_tensors="pt", padding='max_length', truncation=True, max_length=max_source_length).input_ids.to(device)
            embeddingQC = simTokenizer(segQC, return_tensors='pt', max_length=max_source_length, truncation=True).to(device)
            repreQC = simModel(**embeddingQC)
            poQC = repreQC['pooler_output']

            output = t_model.generate(input_ids=input_ids, decoder_start_token_id=0, max_length=max_target_length)
            output = tokenizer.decode(output[0])
            embeddingOutput = simTokenizer(output, return_tensors='pt', max_length=max_target_length, truncation=True).to(device)
            repreOutput = simModel(**embeddingOutput)
            poOutput = repreOutput['pooler_output']

            if F.cosine_similarity(poQC, poOutput, dim=1).item() > 0.9:
                textGenerate = textGenerate + output

        print("Context: \n", valContext)
        print("\n")
        print("Question: \n", valQuestion)
        print("\n")
        print("Model Prediction: \n", textGenerate)
        print("\n")
        print("Gold Answer: \n", valGoldAnswer)
        print("\n")

    else:
        input_ids = tokenizer(valQC, add_special_tokens=True, return_tensors="pt", padding='max_length', truncation=True, max_length=max_source_length).input_ids.to(device)
        output = t_model.generate(input_ids=input_ids, decoder_start_token_id=0, max_length=max_target_length)
        textGenerate = tokenizer.decode(output[0])
        print("Context: \n", valContext)
        print("\n")
        print("Question: \n", valQuestion)
        print("\n")
        print("Model Prediction: \n", textGenerate)
        print("\n")
        print("Gold Answer: \n", valGoldAnswer)
        print("\n")

## clean timestamp data
valTimeStamp = {}
for loc in range(len(timeStampOriginal)):
    if timeStampOriginal[loc] == "\"":               # change to """
        timeStampOriginal = timeStampOriginal[:loc] + "\'" + timeStampOriginal[(loc+1):]
    if loc == 0 or loc == (len(timeStampOriginal)-1):
        continue
    elif loc == 1 or loc == (len(timeStampOriginal)-2):
        timeStampOriginal = timeStampOriginal[:loc] + "\"" + timeStampOriginal[(loc + 1):]
    elif loc < 3 or loc > (len(timeStampOriginal)-5):
        continue
    elif timeStampOriginal[loc] == "'":
        if (timeStampOriginal[loc+1] == "(")\
            or ((timeStampOriginal[loc-1] == ")") & (timeStampOriginal[loc+1] == ":"))\
            or ((timeStampOriginal[loc-2] == ":") & (timeStampOriginal[loc-3] == "\""))\
            or ((timeStampOriginal[loc+1] == ",") & (timeStampOriginal[loc+4] == "(")):
            timeStampOriginal = timeStampOriginal[:loc] + "\"" + timeStampOriginal[(loc + 1):]
    elif (timeStampOriginal[loc] == "\\") & (timeStampOriginal[loc+1] != "n"):
        timeStampOriginal = timeStampOriginal[:loc] + "/" + timeStampOriginal[(loc + 1):]
    valTimeStamp = json.loads(timeStampOriginal)

## make labels
valLabel = []
valLabel.append(int(valTimeGold[1 : valTimeGold.find(",")]))
valLabel.append(int(valTimeGold[valTimeGold.find(",")+2 : len(valTimeGold)-1]))

## find time stamp
simTokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
simModel = BertModel.from_pretrained("bert-base-uncased").to(device)
simModel.eval()

if len(textGenerate) > max_target_length:
    timeStampsSet = []
    for segment in range(int(len(textGenerate) / max_target_length) + 1):
        if segment == int(len(textGenerate) / max_target_length):
            segAns = textGenerate[segment * max_target_length: len(textGenerate)]
        else:
            segAns = textGenerate[segment * max_target_length: (segment + 1) * max_target_length]
        embeddingAns = simTokenizer(segAns, return_tensors='pt', max_length=max_target_length, truncation=True).to(device)
        repreAns = simModel(**embeddingAns)
        poAns = repreAns['pooler_output']

        simScoreStart = {}
        simScoreSelect = {}

        with torch.no_grad():
            for key in list(valTimeStamp.keys()):
                embeddingStartAns = simTokenizer(segAns[:len(valTimeStamp[key])], return_tensors='pt', max_length=max_target_length, truncation=True).to(device)
                repreStartAns = simModel(**embeddingStartAns)
                poStart = repreStartAns['pooler_output']

                embedding_stamp = simTokenizer(valTimeStamp[key], return_tensors='pt', max_length=max_target_length, truncation=True).to(device)
                repreStamp = simModel(**embedding_stamp)
                po1 = repreStamp['pooler_output']
                cosine_scoresStart = F.cosine_similarity(poStart, po1, dim=1)
                # cosine_scoresEnd = F.cosine_similarity(poEnd, po1, dim=1)
                simScoreStart[key] = cosine_scoresStart.item()
                # simScoreEnd[key] = cosine_scoresEnd.item()

            simScoreStart = sorted(simScoreStart.items(),key = lambda x:x[1],reverse = True)
            # simScoreEnd = sorted(simScoreEnd.items(),key = lambda x:x[1],reverse = True)
            startTime = float(simScoreStart[0][0][1 : simScoreStart[0][0].find(",")])
            startKey = simScoreStart[0][0]

        with torch.no_grad():
            subtitleSelect = valTimeStamp[startKey]
            for key in list(valTimeStamp.keys())[list(valTimeStamp.keys()).index(startKey):]:
                for i in range(list(valTimeStamp.keys()).index(key) - list(valTimeStamp.keys()).index(startKey)):
                    subtitleSelect = subtitleSelect + valTimeStamp[list(valTimeStamp.keys())[list(valTimeStamp.keys()).index(startKey) + i]]
                embeddingSelectAns = simTokenizer(subtitleSelect, return_tensors='pt', max_length=max_target_length, truncation=True).to(device)
                repreSelectAns = simModel(**embeddingSelectAns)
                poSelect = repreSelectAns['pooler_output']
                cosine_scoresSelect = F.cosine_similarity(poSelect, poAns, dim=1)
                simScoreSelect[key] = cosine_scoresSelect.item()

        simScoreSelect = sorted(simScoreSelect.items(),key = lambda x:x[1],reverse = True)
        timeStampPair = []
        timeStampPair.append(float(simScoreStart[0][0][1 : simScoreStart[0][0].find(",")]))
        timeStampPair.append(float(simScoreSelect[0][0][simScoreSelect[0][0].find(",")+2 : len(simScoreSelect[0][0])-1]))

    print("Stamp prediction: ")
    for pair in timeStampsSet:
        print("{} - {}".format(pair[0], pair[1]))

else:
    embeddingAns = simTokenizer(textGenerate, return_tensors='pt', max_length=max_target_length, truncation=True).to(device)
    repreAns = simModel(**embeddingAns)
    poAns = repreAns['pooler_output']

    simScoreStart = {}
    simScoreSelect = {}

    with torch.no_grad():
        for key in list(valTimeStamp.keys()):
            embeddingStartAns = simTokenizer(textGenerate[:len(valTimeStamp[key])], return_tensors='pt',
                                             max_length=max_target_length, truncation=True).to(device)
            repreStartAns = simModel(**embeddingStartAns)
            poStart = repreStartAns['pooler_output']

            embedding_stamp = simTokenizer(valTimeStamp[key], return_tensors='pt', max_length=max_target_length,
                                           truncation=True).to(device)
            repreStamp = simModel(**embedding_stamp)
            po1 = repreStamp['pooler_output']
            cosine_scoresStart = F.cosine_similarity(poStart, po1, dim=1)
            # cosine_scoresEnd = F.cosine_similarity(poEnd, po1, dim=1)
            simScoreStart[key] = cosine_scoresStart.item()
            # simScoreEnd[key] = cosine_scoresEnd.item()

        simScoreStart = sorted(simScoreStart.items(), key=lambda x: x[1], reverse=True)
        # simScoreEnd = sorted(simScoreEnd.items(),key = lambda x:x[1],reverse = True)
        startTime = float(simScoreStart[0][0][1: simScoreStart[0][0].find(",")])
        startKey = simScoreStart[0][0]

    with torch.no_grad():
        subtitleSelect = valTimeStamp[startKey]
        for key in list(valTimeStamp.keys())[list(valTimeStamp.keys()).index(startKey):]:
            for i in range(list(valTimeStamp.keys()).index(key) - list(valTimeStamp.keys()).index(startKey)):
                subtitleSelect = subtitleSelect + valTimeStamp[
                    list(valTimeStamp.keys())[list(valTimeStamp.keys()).index(startKey) + i]]
            embeddingSelectAns = simTokenizer(subtitleSelect, return_tensors='pt', max_length=max_target_length, truncation=True).to(device)
            repreSelectAns = simModel(**embeddingSelectAns)
            poSelect = repreSelectAns['pooler_output']
            cosine_scoresSelect = F.cosine_similarity(poSelect, poAns, dim=1)
            simScoreSelect[key] = cosine_scoresSelect.item()

    simScoreSelect = sorted(simScoreSelect.items(), key=lambda x: x[1], reverse=True)
    timeStampPair = []
    timeStampPair.append(float(simScoreStart[0][0][1: simScoreStart[0][0].find(",")]))
    timeStampPair.append(float(simScoreSelect[0][0][simScoreSelect[0][0].find(",") + 2: len(simScoreSelect[0][0]) - 1]))

print("Stamp prediction: {} - {}", timeStampPair[0], timeStampPair[1])

print("Stamp gold: {} - {}".format(valLabel[0], valLabel[1]))

























