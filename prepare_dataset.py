import pandas as pd
import joblib
from sklearn import preprocessing
from sklearn import model_selection
import config


def process_data(dataset_path=config.DATASET_PATH):
    df = pd.read_csv(dataset_path, encoding = 'latin-1')
    df.loc[:, 'Sentence #'] = df['Sentence #'].fillna(method='ffill')

    enc_pos = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()

    df.loc[:, 'POS'] = enc_pos.fit_transform(df['POS'])
    df.loc[:, 'Tag'] = enc_tag.fit_transform(df['Tag'])

    sentences = df.groupby('Sentence #')['Word'].apply(list).values
    pos = df.groupby('Sentence #')['POS'].apply(list).values
    tag = df.groupby('Sentence #')['Tag'].apply(list).values
    
    meta_data = {
        'enc_pos': enc_pos,
        'enc_tag': enc_tag
    }
    
    joblib.dump(meta_data, 'meta.bin')
    
    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))
    
    (
        train_sentences,
        test_sentences,
        train_pos,
        test_pos,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(sentences, pos, tag, random_state=config.RANDOM_SEED, test_size=0.2)
    
    (
        valid_sentences,
        test_sentences,
        valid_pos,
        test_pos,
        valid_tag,
        test_tag
    ) = model_selection.train_test_split(test_sentences, test_pos, test_tag, random_state=config.RANDOM_SEED, test_size=0.5)
    
    
    return {
        'train_sentences': train_sentences, 'train_pos': train_pos, 'train_tag': train_tag,
        'valid_sentences': valid_sentences, 'valid_pos': valid_pos, 'valid_tag': valid_tag,
        'test_sentences': test_sentences, 'test_pos': test_pos, 'test_tag': test_tag,   
    }, num_pos, num_tag
    
