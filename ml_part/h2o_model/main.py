import h2o
from h2o.automl import H2OAutoML

# Start the H2O cluster (locally)
h2o.init()

TRAIN_CSV_PATH = r''

# Import a sample binary outcome train/test set into H2O
train = h2o.import_file(r"data\H2O_Auto_ML\train_logP_data.csv")
test = h2o.import_file(r"data\H2O_Auto_ML\test_logP_data.csv")

# Identify predictors and response
x = train.columns
y = "logP"
x.remove(y)

# train[y] = train[y].asfactor()
# test[y] = test[y].asfactor()

# Run AutoML for 20 base models
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb.head(rows=lb.nrows))