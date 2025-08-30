import pickle

with open("dic_pred0.pkl", "rb") as f:
    dic = pickle.load(f)

# with open("dic_format0.pkl", "rb") as f:
#     dic = pickle.load(f)

from pprint import pprint

with open("dic_pred0.txt", "w") as f:
    pprint(dic, stream=f)

with open("dic_perf0.pkl", "rb") as f:
    dic = pickle.load(f)

# with open("dic_format0.pkl", "rb") as f:
#     dic = pickle.load(f)

from pprint import pprint

with open("dic_perf0.txt", "w") as f:
    pprint(dic, stream=f)