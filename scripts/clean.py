import pickle
with open('cv19index/resources/xgboost/model.pickle', 'rb') as f:
    unpickled = pickle.load(f)
del unpickled['predictions']
with open('cv19index/resources/xgboost/model.pickle', 'wb') as f:
    pickle.dump(unpickled, f)