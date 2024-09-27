import pickle

with open(r'data\updated_features\file.pkl', 'rb') as handle:
    b = pickle.load(handle)

print(b)