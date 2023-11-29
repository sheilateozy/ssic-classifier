from typing import List, Tuple
import re
import pandas as pd
import pickle
from transformers import BertTokenizer
import torch
from src.text_backtranslator import back_translate


class DataPreprocessor:
    '''
    Attributes
    ----------
    config: dict
        Project configurations in /config/main.yaml

    data: pd.DataFrame()
        Original dataset

    texts: pd.Series
        Raw text features for classfication task

    tokenized_texts: List[torch.Tensor]
        Tokenized text features for classfication task
    '''

    def __init__(self, config: dict):
        self.config = config

    def _process_text(self) -> None:
        '''
        Attributes Instantiated
        -----------------------
        tokenized_texts: List[torch.Tensor]
            Tokenized text features for classfication task
        '''
        
        #step 1: clean text
        def helper(text: str):
            if not text:
                return ''

            email_regex = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
            replace_nec_regex = re.compile(r'\s+(n.e.c.|n.e.c)')
            remove_fixed_start_regex = re.compile(r'the principal activit.+?\s*(of the company)*\s*(is)*\s+') 
            text = email_regex.sub(r'', text)
            text = replace_nec_regex.sub(r' nec', text)
            text = remove_fixed_start_regex.sub(r' ', text)

            return text

        clean_texts = list(self.texts.apply(lambda text: helper(text)))

        #step 2: tokenize text
        tokenizer = BertTokenizer.from_pretrained(self.config['pretrained_model']['path'])
        self.tokenized_texts = [tokenizer(text, padding='max_length', truncation=True, max_length=self.config['pretrained_model']['tokenizer_max_length'], return_tensors='pt') for text in clean_texts]


class TrainingDataPreprocessor(DataPreprocessor):
    '''
    Attributes
    ----------
    config: dict
        Project configurations in /config/main.yaml

    data: pd.DataFrame()
        Original dataset

    texts: pd.Series
        Raw text features for classfication task

    tokenized_texts: List[torch.Tensor]
        Tokenized text features for classfication task

    label_encoding_map: dict
        Maps the encoded labels to the raw labels, in the format {raw label: encoded label}

    labels: pd.Series
        Raw target for classification task

    encoded_labels: List[int]
        Target encoded into integers
    '''

    def __init__(self, config: dict, data: pd.DataFrame):
        super().__init__(config)
        self.data = data

    def _augment_text(self):
        '''
        Obtain additional training text data using back-translation


        Attributes Instantiated
        -----------------------
        texts: pd.Series
            Raw text features for classfication task

        labels: pd.Series
            Raw target for classification task
        '''

        #obtain additional training text data using back translation
        new_texts = self.data['desc'].apply(lambda text: back_translate(text))
        new_texts_df = pd.DataFrame(new_texts, columns=['desc'])
        
        #add additional texts to self.data
        new_texts_df['ssic'] = self.data['ssic']  #add ssic col into new_texts_df so it has same format as self.data
        self.data = pd.concat([self.data, new_texts_df])

        #instantiate attributes: texts, labels
        self.texts = self.data['desc']  #cols are already renamed to standard format from DataSplitter
        self.labels = self.data['ssic']  #cols are already renamed to standard format from DataSplitter

    def _balance_training_set(self) -> None:
        '''
        Through oversampling
        Ensures that each label category has the same number of rows
        By oversampling smaller categories with replacement until it has the same number of rows as the largest label category
        '''
        
        #set min_n to the maximum number of rows in any label category
        min_n = self.labels.groupby(self.labels).size().max()

        #number of samples needed for each label category to reach min_n
        samples_req_per_label_cat = dict(self.labels.groupby(self.labels).size().apply(lambda x: max(min_n - x, 0)))  #a dict

        #add samples to both features and label
        #first make a df for both features and label
        df = pd.DataFrame([self.texts, self.labels]).T
        df.columns = ['text', 'label']

        samples_to_add = df.groupby(df['label'], group_keys=False)\
                            .apply(lambda g: g.sample(n=samples_req_per_label_cat[g.name], replace=len(g) < samples_req_per_label_cat[g.name]))
        self.labels = pd.concat([self.labels, samples_to_add['label']]).reset_index(drop=True)
        self.texts = pd.concat([self.texts, samples_to_add['text']]).reset_index(drop=True)

    def _shuffle_rows(self) -> None:
        '''
        Shuffle training data to improve model performance:
        - Reduce order bias: Shuffling the dataset ensures that the training examples are presented to the model in a random order, preventing any potential biases introduced by the order of the samples. In some cases, the data may be sorted or clustered by class, and shuffling helps to break such patterns.
        - Better convergence: When using optimization algorithms which update the model's parameters based on a mini-batch of samples, shuffling the dataset ensures that each mini-batch consists of a diverse set of examples. This helps the optimization algorithm to converge faster and more accurately.
        - Robustness to class imbalance: For imbalanced data, shuffling can help ensure that each mini-batch contains a more balanced representation of classes. This prevents the model from overfitting to the majority class and improves its performance on the minority class.
        '''

        #use the same random seed to shuffle both tokenized_texts and encoded_labels so that the order of rows will be the same in both
        self.texts = pd.Series(self.texts).sample(frac=1, random_state=self.config['random_seed']).reset_index(drop=True)
        self.labels = pd.Series(self.labels).sample(frac=1, random_state=self.config['random_seed']).reset_index(drop=True)

    def _encode_label(self) -> None:
        '''
        Attributes Instantiated
        -----------------------
        encoded_labels: List[int]
            Target encoded into integers for classfication task

        label_encoding_map: dict
            Maps the encoded labels to the raw labels, in the format {encoded label: raw label (ssic)}
        '''

        self.encoded_labels = list(self.labels.astype('category').cat.codes)
        label_encoding_map = dict(enumerate(self.labels.astype('category').cat.categories))
        label_encoding_map = dict((v, k) for k, v in label_encoding_map.items())

        self.label_encoding_map = label_encoding_map  

        #save label_encoding_map
        with open(self.config['mapping']['label_encoding_map']['path'], 'wb') as file:
            pickle.dump(label_encoding_map, file, protocol=pickle.HIGHEST_PROTOCOL)
 
    def process(self) -> Tuple[List[str], List[int]]:
        '''
        Returns
        -------
        tokenized_texts: List[torch.Tensor]
            Tokenized text features for classfication task

        encoded_labels: List[int]
            Target encoded into integers for classfication task
        '''

        self._augment_text()
        self._balance_training_set()
        self._shuffle_rows()
        self._process_text()
        self._encode_label()
        return self.tokenized_texts, self.encoded_labels


class TestDataPreprocessor(DataPreprocessor):
    '''
    Attributes
    ----------
    config: dict
        Project configurations in /config/main.yaml

    data: pd.DataFrame()
        Original dataset

    texts: pd.Series
        Raw text features for classfication task

    tokenized_texts: List[torch.Tensor]
        Tokenized text features for classfication task

    label_encoding_map: dict
        Maps the encoded labels to the raw labels, in the format {raw label: encoded label}

    labels: pd.Series
        Raw target for classification task

    encoded_labels: List[int]
        Target encoded into integers
    '''

    def __init__(self, config: dict, data: pd.DataFrame):
        super().__init__(config)
        self.data = data
        self.texts = data['desc']  #cols are already renamed to standard format from DataSplitter
        self.labels = data['ssic']  #cols are already renamed to standard format from DataSplitter

    def _encode_label(self) -> None:
        '''
        Attributes Instantiated
        -----------------------
        encoded_labels: List[int]
            Target encoded into integers for classfication task

        label_encoding_map: dict
            Maps the encoded labels to the raw labels, in the format {encoded label: raw label (ssic)}
        '''

        #use label encoding map that was already saved
        #this means that training set must always be processed before test set
        with open(self.config['mapping']['label_encoding_map']['path'], 'rb') as file:
            label_encoding_map = pickle.load(file)
        self.label_encoding_map = label_encoding_map
        self.encoded_labels = list(self.labels.apply(lambda x: label_encoding_map[x]))

    def process(self) -> Tuple[List[str], List[int]]:
        '''
        Returns
        -------
        tokenized_texts: List[torch.Tensor]
            Tokenized text features for classfication task

        encoded_labels: List[int]
            Target encoded into integers for classfication task
        '''

        self._process_text()
        self._encode_label()
        return self.tokenized_texts, self.encoded_labels


class MultiInferenceDataPreprocessor(DataPreprocessor):
    '''
    Attributes
    ----------
    config: dict
        Project configurations in /config/main.yaml

    data: pd.DataFrame()
        Original dataset

    texts: pd.Series
        Raw text features for classfication task

    tokenized_texts: List[torch.Tensor]
        Tokenized text features for classfication task
    '''
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.data = pd.read_excel(config['data']['inference_data']['path'])
        self.texts = self.data[config['data']['inference_data']['text_col_name']]  #DataSplitter not used on inference data so col names are not in standard format, unlike train/test/val data
        #no labels in data

    def process(self) -> List[str]:
        '''
        Returns
        -------
        tokenized_texts: List[torch.Tensor]
            Tokenized text features for classfication task
        '''

        self._process_text()
        return self.tokenized_texts
    

class SingleInferenceDataPreprocessor(DataPreprocessor):
    '''
    Attributes
    ----------
    config: dict
        Project configurations in /config/main.yaml

    texts: pd.Series
        Raw text features for classfication task

    tokenized_texts: List[torch.Tensor]
        Tokenized text features for classfication task
    '''
    
    def __init__(self, config: dict, text: str):
        super().__init__(config)
        self.texts = pd.Series([text])

    def process(self) -> List[str]:
        '''
        Returns
        -------
        tokenized_texts: List[torch.Tensor]
            Tokenized text features for classfication task
        '''

        self._process_text()
        return self.tokenized_texts