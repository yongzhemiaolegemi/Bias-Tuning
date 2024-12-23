import json
#################### Configuration ####################
result_dir = 'Qwen2.5-1.5B-Instruct_mbpp.json'

#######################################################
with open(result_dir, 'r') as f:
    data = json.load(f)
count_correct = 0
count_total = 0
count_false = 0
count_error = 0
for item in data:
    count_total += 1
    if item['is_correct']=='true':
        count_correct += 1
    if item['is_correct']=='false':
        count_false += 1
    if item['is_correct']=='error':
        count_error += 1
print(count_correct, count_total, count_false, count_error)
print(count_correct/count_total,count_false/count_total,count_error/count_total) 