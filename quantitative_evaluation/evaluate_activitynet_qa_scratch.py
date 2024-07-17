# import openai
import os
import json
import ast
from multiprocessing.pool import Pool

class Opt:
    def __init__(self):
        self.root_dir = '/media/user/9d078323-0374-41e4-b6fe-bf9903a83d23/VLM/Eval/ActivityNet_Test-1-3_videos'
        self.output_dir = self.root_dir
        self.pred_path = os.path.join(self.root_dir, 'video_chatgpt_activitynet_qa_preds.json')
        self.output_json = os.path.join(self.root_dir, 'video_chatgpt_activitynet_qa_preds_result.json')
        self.num_tasks = 1

def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = Opt()

    file = open(args.pred_path)
    pred_contents = json.load(file)

    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        video_id = sample['id']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['video_name']
        question = sample['question']
        answer = sample['answer']
        pred = sample['pred']
        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set

    # Calculate average score and accuracy
    yes_count = 0
    no_count = 0
    correct_count = 0
    total_count = 0
    for _, result in prediction_set.items():
        # Computing accuracy
        answer = result['a'].lower()
        pred = result['pred'].lower()
        total_count += 1

        if answer == "yes":
            if "yes" in pred:
                correct_count += 1
            yes_count += 1
        elif answer == "no":
            if "no" in pred:
                correct_count += 1
            no_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Correct count:", correct_count)
    print("Total count:", total_count)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()

