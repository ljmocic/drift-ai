import os
arr = [x for x in os.listdir() if x.endswith(".out")]
print(arr)