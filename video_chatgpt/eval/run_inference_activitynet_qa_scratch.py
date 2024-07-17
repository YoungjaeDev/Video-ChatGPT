'''
python video_chatgpt/eval/run_inference_activitynet_qa.py \
    --video_dir ./ \
    --gt_file_question test_q.json \
    --gt_file_answers test_a.json \
    --output_dir ./ \
    --model-name mmaaz60/LLaVA-7B-Lightening-v1-1 \
    --output_name video_chatgpt_activitynet_qa_preds \
    --projection_path Video-ChatGPT-7B/video_chatgpt-7B.bin
'''

import os 
import json
from tqdm import tqdm
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

opd = os.path.dirname
sys.path.append(opd(opd(opd(os.path.abspath(__file__)))))

from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer


class Opt:
    def __init__(self):
        self.root_dir = '/media/user/9d078323-0374-41e4-b6fe-bf9903a83d23/VLM/Eval/ActivityNet_Test-1-3_videos'
        self.video_dir = os.path.join(self.root_dir, 'all_test')
        self.gt_file_question = os.path.join(self.root_dir, 'test_q.json')
        self.gt_file_answers = os.path.join(self.root_dir, 'test_a.json')
        self.output_dir = self.root_dir
        self.output_name = 'video_chatgpt_activitynet_qa_preds'
        self.model_name = 'mmaaz60/LLaVA-7B-Lightening-v1-1'
        self.conv_mode = 'video-chatgpt_v1'
        self.projection_path = 'Video-ChatGPT-7B/video_chatgpt-7B.bin'

def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name,
                                                                                        args.projection_path)
    # model: 32006

    with open(args.gt_file_question) as file:
        gt_questions = json.load(file)
    with open(args.gt_file_answers) as file:
        gt_answers = json.load(file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    index = 0
    for sample in tqdm(gt_questions):
        video_name = sample['video_name']
        question = sample['question']
        id = sample['question_id']
        answer = gt_answers[index]['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}

        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"v_{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        video_frames = load_video(video_path)

        try:
            
            output = video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len)
            sample_set['pred'] = output
            output_list.append(sample_set)
        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")

    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)

if __name__ == "__main__":
    args = Opt()
    run_inference(args)