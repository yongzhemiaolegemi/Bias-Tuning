from transformers import AutoModelForCausalLM, AutoTokenizer
from  template import EvaluationConfig
import json
from tqdm import tqdm
#################### Configuration ####################
model_name = "/home/beaver/models/Qwen2.5-1.5B-Instruct"
# evaluation_list = ['gsm8k', 'mbpp']
evaluation_list = ['mbpp']
batch_size = 4


#######################################################
model_name_only = model_name.split('/')[-1]

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

for evaluation_name in evaluation_list:
    # new_data_list = []
    eval_config = EvaluationConfig(evaluation_name)
    data_dir, get_batch_prompt, judge = eval_config.get_all()
    # Load the dataset
    with open(data_dir, 'r') as f:
        data = json.load(f)
    # Split the dataset into batches
    new_data = []
    pbar = tqdm(range(0, len(data), batch_size))
    for item in pbar:
        batch_data = data[item:item+batch_size]
        batch_prompt = get_batch_prompt(batch_data)
        # Generate the response
        messages_batch = [
            [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}]
            for prompt in batch_prompt
        ]
        texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
        model_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(responses)
        for response, data_item in zip(responses, batch_data):
            is_correct, filtered_prediction, filtered_gt = judge(response, data_item)

            print(is_correct, filtered_prediction)
            data_item['is_correct'] = is_correct
            data_item['filtered_prediction'] = filtered_prediction
            data_item['response'] = response
            new_data.append(data_item)
        output_dir = f"{model_name_only}_{evaluation_name}.json"
        with open(output_dir, 'w') as f:
            json.dump(new_data, f, indent=4)
        
        # exit()
