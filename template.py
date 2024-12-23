from system_prompt import system_prompt_math_qa, system_prompt_code 
import re
class EvaluationConfig:
    def __init__(self, evaluation_name):
        self.evaluation_name = evaluation_name
        self.gsm8k_dir = "raw_data/gsm8k.json"
        self.mbpp_dir = "raw_data/mbpp.json"


    def get_all(self):
        if self.evaluation_name == 'gsm8k':
            return self.gsm8k_dir, self.get_batch_prompt_gsm8k, self.judge_gsm8k
        elif self.evaluation_name == 'mbpp':
            return self.mbpp_dir, self.get_batch_prompt_mbpp, self.judge_mbpp
        else:
            print("Invalid evaluation name: ", self.evaluation_name)
            exit()


    def get_batch_prompt_gsm8k(self, batch_data):
        batch_prompt = []
        for data in batch_data:
            prompt = system_prompt_math_qa + data['question']
            batch_prompt.append(prompt)
        return batch_prompt
    

    def get_batch_prompt_mbpp(self, batch_data):
        batch_prompt = []
        for data in batch_data:
            prompt = system_prompt_code + data['text'] +data['def_code']
            batch_prompt.append(prompt)
        return batch_prompt
    

    def judge_gsm8k(self, raw_prediction, raw_gt):
        
        matches = re.findall(r'\[(.*?)\]', raw_prediction)
        filtered_prediction = "none"    
        if len(matches) > 0:
            filtered_prediction = matches[-1]
        filtered_gt = raw_gt['direct_answer']
        if filtered_gt==filtered_prediction:
            is_correct = 'true'
        elif filtered_prediction=='none':
            is_correct = 'error'
        else:
            is_correct = 'false'
        return is_correct, filtered_prediction, filtered_gt
    
    
    def judge_mbpp(self, raw_prediction, raw_gt):
        is_pass = 'false'
        full_code = raw_prediction
        # go to 'def':
        if full_code.find('def') != -1:
            full_code = full_code[full_code.find('def'):]
        # delete the ' in the end:
        full_code = full_code.rstrip("```")
        test_cases = raw_gt['test_list']
        filtered_gt = raw_gt['code']
        import signal
        def timeout_handler(signum, frame):
            # raise TimeoutError("Test case execution exceeded time limit.")
            pass
            return
        TIME_LIMIT = 5
        try:
            # Execute the function definition code
            namespace = {}
            exec(full_code, namespace)
            
            # Flag to track if all test cases pass
            all_tests_passed = True

            # Execute all test cases
            for test_case in test_cases:
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(TIME_LIMIT)
                    exec(test_case, namespace)
                    is_pass = 'true'
                    signal.alarm(0)
                except Exception as e:
                    # print(f'Error in test case at index {i}: {test_case}')
                    # print(f'Error message: {e}')
                    is_pass = 'false'
                    break  # Exit the loop on first error in the test case
            del namespace
            # If all test cases passed, increment the count


        except Exception as e:
            # print(f'Error executing function definition at index {i}: {e}')
            is_pass = 'error'
        return is_pass, full_code, filtered_gt
    
        
