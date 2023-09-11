import torch
import transformers


MAX_LEN = 128
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 12
DATASET_PATH = '/content/entity-annotated-corpus/ner_dataset.csv'
MODEL_PATH = '/content/bert_model_state/'

LEARN_RATE = 3e-6
CHECKPOINT = 'bert-base-uncased'
TOKENIZER = transformers.AutoTokenizer.from_pretrained(CHECKPOINT)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 666
