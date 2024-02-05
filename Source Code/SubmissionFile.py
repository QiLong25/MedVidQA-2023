import pandas as pd

subtitles_df = pd.read_csv("final_id_subtitles.csv")
scores_df = pd.read_csv("top_1000_scores.csv")
scores_df =scores_df.iloc[:, :2] # This only looks at the top 1 ranked videos

merged_data = []

for index, row in scores_df.iterrows():
    question = row["questions"]

    for rank in range(1, 2):
        cell_content = row[rank]  # This is the tuple of video_id and score

        video_id, score = eval(cell_content)

        subtitles = subtitles_df[subtitles_df["video_id"] == video_id]["subtitles"].values[0]

        new_row = {"questions": question, "video_id": video_id, "video_score": score, "subtitles": subtitles}
        merged_data.append(new_row)

merged_df = pd.DataFrame(merged_data)
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5TokenizerFast
import torch

model_folder = os.path.join(os.path.dirname(__file__), "model")
model = T5ForConditionalGeneration.from_pretrained(model_folder)
tokenizer = T5TokenizerFast.from_pretrained("t5-small")


def get_answer(question, subtitles):
    input_text = f"Answer: {subtitles} Context: {question}"
    inputs = tokenizer.encode_plus(input_text, max_length=512, truncation=True, return_tensors="pt")

    # Generate the answer using the T5 model with beam search and temperature scaling
    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'],
            num_beams=5,  # Use beam search
            temperature=0.7,  # Adjust temperature for scaling
            max_length=150  # Set max length of generated answers
        )

    # Decode the generated output to get the answer
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


answers = []
count = 1
for index, row in merged_df.iterrows():
    print(count)
    question = row["questions"]
    subtitles = row["subtitles"]

    answer = get_answer(question, subtitles)
    answers.append({"video_id": row["video_id"], "question": question, "answer": answer})
    count += 1

answers_df = pd.DataFrame(answers)
answers_df.to_csv('answers.csv', index = False)

###################################################################################################################

from youtube_transcript_api import YouTubeTranscriptApi

import pandas as pd

answers_df = pd.read_csv('answers.csv')


# Function to fetch subtitles with rounded timestamps
def get_subtitles_with_rounded_timestamps(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        subtitles_dict = {}

        for entry in transcript:
            start_time = round(entry['start'])
            end_time = round(entry['start'] + entry['duration'])
            text = entry['text']

            if text not in subtitles_dict:
                subtitles_dict[text] = []

            subtitles_dict[text].append((start_time, end_time))

        return subtitles_dict

    except Exception as e:
        print(f"Error fetching subtitles for video_id {video_id}: {e}")
        return {}


# Add a new column to the DataFrame for timestamp_subtitles
answers_df['timestamp_subtitles'] = answers_df['video_id'].apply(get_subtitles_with_rounded_timestamps)

# Save the DataFrame with the new column to timestamp_answers.csv
answers_df.to_csv('timestamp_answers.csv', index=False)

##################################################################################################################
import pandas as pd
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("timestamp_answers.csv")

# Define a context window size (in seconds) around the predicted answer
context_window_size = 25  # You can adjust this as needed


# Function to preprocess subtitles
def preprocess_subtitles(subtitles):
    cleaned_subtitles = {}
    for sentence, timestamps in subtitles.items():
        cleaned_sentence = re.sub(r'[^\w\s]', '', sentence).lower()
        cleaned_subtitles[cleaned_sentence] = timestamps
    return cleaned_subtitles


# Function to find start and end timestamps for a given answer in cleaned subtitles
def find_context_window(answer, subtitles):
    tfidf_vectorizer = TfidfVectorizer()
    subtitles_text = list(subtitles.keys())
    tfidf_matrix = tfidf_vectorizer.fit_transform(subtitles_text)
    answer_tfidf = tfidf_vectorizer.transform([answer])
    similarities = cosine_similarity(answer_tfidf, tfidf_matrix).flatten()

    max_similarity_index = similarities.argmax()
    closest_subtitle = list(subtitles.keys())[max_similarity_index]
    closest_subtitle_timestamps = subtitles[closest_subtitle]

    # Find the average of start times and end times
    start_times = [start for start, _ in closest_subtitle_timestamps]
    end_times = [end for _, end in closest_subtitle_timestamps]

    avg_start = sum(start_times) / len(start_times)
    avg_end = sum(end_times) / len(end_times)

    start = max(0, avg_start - context_window_size)
    end = avg_end + context_window_size

    return start, end


# Iterate through the dataframe rows and find context windows for each answer
for index, row in df.iterrows():
    predicted_answer = row["answer"]
    timestamp_subtitles = ast.literal_eval(row["timestamp_subtitles"])
    cleaned_subtitles = preprocess_subtitles(timestamp_subtitles)  # Cleaned subtitles
    start, end = find_context_window(predicted_answer, cleaned_subtitles)
    df.at[index, "start_timestamp"] = int(start)
    df.at[index, "end_timestamp"] = int(end)

# Save the updated dataframe with start and end timestamps
df.to_csv("timestamp_answers_with_context.csv", index=False)

###############################################################################################

# Add a relevant score

# We will use the cosine score from earlier

scores_df = pd.read_csv("top_1000_scores.csv")
scores_df =scores_df.iloc[:, :2] # This only looks at the top 1 ranked videos

scores = []

for index, row in scores_df.iterrows():
    question = row["questions"]

    for rank in range(1, 2):
        cell_content = row[rank]  # This is the tuple of video_id and score

        video_id, score = eval(cell_content)
        score = round(score, 2)


        new_row = {"relevant_score": score}
        scores.append(new_row)

scores_df = pd.DataFrame(scores)

existing_df = pd.read_csv("timestamp_answers_with_context.csv")

# Concatenate the dataframes horizontally (axis=1)
merged_df = pd.concat([existing_df, scores_df], axis=1)

# Save the merged dataframe back to CSV
merged_df.to_csv("scores_timestamp_answers_with_context.csv", index=False)

##########################################################################################

# Making JSON file

import pandas as pd
import json

df = pd.read_csv("scores_timestamp_answers_with_context.csv")

# Create an empty list to store the JSON entries
json_data = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    question_id = f"Q{index + 1}"
    relevant_video = {
        "video_id": row["video_id"],
        "relevant_score": round(row["relevant_score"], 2),
        "answer_start_second": round(row["start_timestamp"], 2),
        "answer_end_second": round(row["end_timestamp"], 2)
    }

    # Create the JSON entry for the current question
    json_entry = {
        "question_id": question_id,
        "relevant_videos": [relevant_video]
    }

    # Append the JSON entry to the list
    json_data.append(json_entry)

# Save the JSON data to a file
with open("submission.json", "w") as json_file:
    json.dump(json_data, json_file, indent=4)










