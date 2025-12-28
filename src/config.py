TRAIN_CSV = "../data/processed/train.csv"
VAL_CSV = "../data/processed/val.csv"
TEST_CSV = "../data/processed/test.csv"

EXPERIMENT_NAME = "customer-churn-merged-split"
OUT_DIR = "../artifacts"

TARGET = "Churn"
ID_COL = "CustomerID"

RANDOM_STATE = 42

CAT_COLS = ["Gender", "Subscription Type", "Contract Length"]
NUM_COLS = [
    "Age","Tenure","Usage Frequency",
    "Support Calls","Payment Delay",
    "Total Spend","Last Interaction"
]

DROP_FEATURES = []

