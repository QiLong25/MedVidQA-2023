# MedVidQA-2023
Summer Research at Applied-AI Lab, NC State University

![model](https://github.com/QiLong25/MedVidQA-2023/assets/143149589/aac91c52-c60c-4c18-9e48-544dac3afe03)

## Challenge
![MedVidQA](https://medvidqa.github.io/img/vcval.PNG)

[MedVidQA](https://medvidqa.github.io/)

**Task: Video Corpus Visual Answer Localization (VCVAL)**

Given a medical query and a collection of videos, the task aims to retrieve the appropriate video from the video collection and then locate the temporal segments (start and end timestamps) in the video where the answer to the medical query is being shown, or the explanation is illustrated in the video.

The VCVAL task consists of two subtasks: (a) Video Retrieval, and (b) Temporal Segment Prediction.

## Abstract
[TRECVID Paper](https://www-nlpir.nist.gov/projects/tvpubs/tv.pubs.23.org.html)

In this paper, we present our solution to the MedVidQA 2023 Task 1: Video Corpus Visual Answer Localization. We used the training and testing datasets provided by the MedVidQA 2023 competition. For our run-1: we utilized a subtitle-questions cosine similarity score to rank the videos and then implemented a **T5 model**. For our run-2: we adjusted our ranking system to return the top three subtitle-answer similarity from the outputs of our **BigBird model**. For run-3: we used methods almost identical to run-1 except the T5 model was adjusted under different constraints. We found that run-1 had the best performance in leveraging both the selection of the relevant videos and IOU score. Although, we did notice that the IOU in run-2 could be stronger on some queries, yet the ranking did not encompass a broader selection of all of the possible relevant videos. The results of run-3 were of lower performance when compared to our previous runs since the fine-tuning of the T5 model was not at an optimal level. One of the issues we had with solving this task were the capabilities of our GPUs and the length of the training time. Our datasets were quite large and this put significant strain on our models.
 
***Index Terms*—Timestamp Location, Text Generation, T5 Model, Natural Language Processing, Instructional Videos, Sub-title Fragment Localization**

## Work Undertaken (Abstract)
With the development of artificial intelligence, AI-aid learning is getting increasingly popular. Extracting essential information from instructional videos is one of them. Facing the challenge of locating the timestamp of instructional medical videos to answer specific medical questions, our research group proposes several applicable Natural Language Processing models. I propose a three-stage model based on text-to-text model. At the **first stage**, text similarity is calculated between questions and video subtitles to select the most related videos that probably contain answer to the question. At the **second stage**, a text-to-text conditional generation model T5 is finetuned to generate textual answer to the question based on video subtitles. At the **last stage**, embedding cosine similarity is calculated to locate the subtitle fragment and subsequently locate the timestamp of that fragment. Experiments have been carried out to compare different generation models and test the model’s effectiveness on solving the task. The average IOU can reach 0.5877 on test dataset. **It is an innovative temptation to apply “text-to-text” language model for answer locating and medical instruction extraction.**
