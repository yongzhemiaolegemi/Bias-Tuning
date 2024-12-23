system_prompt_math_qa = '''
Solve the following question by thinking step by step and provide the answer in the end in [ ]. The final answer should be a number without any additional text.\n
An example:
What is 2 + 3?
Let's solve this step by step:
1. Add 2 and 3 together.
2. The result is 5.
Therefore, the answer is [5].
IMPORTANT: Your output should always end with:

Therefore, the answer is [your answer].(your answer should be a number)

Now, let's solve the question:
'''
system_prompt_code = '''
Repeat the following function definition and complete the code inside the function. Output the entire Python code as a plain string that can be directly executed using `exec()`. Do not add any extra content.\n
'''

