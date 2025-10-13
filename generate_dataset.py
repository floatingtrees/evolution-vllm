import json
from random import randint

num_datapoints = 30
datalist = []
max_num = 10000
for i in range(num_datapoints):
    num1 = randint(1, max_num)
    num2 = randint(1, max_num)
    boxed_answer = "Box your answer like this: \\boxed\{answer\}"
    question = f"What is {num1} multiplied by {num2}? {boxed_answer}"
    datalist.append({"question": question, "answer": num1 * num2})
with open("demo_dataset.json", "w") as f:
    json.dump(datalist, f, indent=2)
