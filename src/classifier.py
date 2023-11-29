from typing import List, Optional
import pandas as pd
from transformers import BertForSequenceClassification
import torch
import pickle
import numpy as np
from sklearn.metrics import classification_report


class SingleInferenceClassifier:
    '''
    Used for predictions on text inputted via Streamlit app


    Attributes
    ----------
    config: dict
        Project configurations in /config/main.yaml

    tokenized_texts_for_pred: List[torch.Tensor]
        Tokenized texts to obtain predictions for

    model: BertForSequenceClassification
        Ensemble model to be used for prediction
    '''


    #class attributes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #instance attributes
    def __init__(self,
                config: dict,
                tokenized_texts_for_pred: List[torch.Tensor],
                model: BertForSequenceClassification):
        self.config = config
        self.tokenized_texts_for_pred = tokenized_texts_for_pred
        self.model = model

    def get_predictions(self, top_n: int = 3) -> pd.DataFrame:
        '''
        Obtains predictions and prediction probabilities for SSIC code only


        Parameters
        ----------
        top_n: int
            Number of predictions to be obtained per text.
            Method: For every text, obtain its prediction probability for all possible label classes.
                    Return the top n label classes with the highest probabilities.


        Returns
        -------
        preds_df: pd.DataFrame
            Predictions and prediction probabilities sorted from highest to lowest probability
        '''

        #get the 1 tokenized text
        tokenized_text = self.tokenized_texts_for_pred[0]

        #get prediction probs for all possible label classes
        input_ids = tokenized_text['input_ids'].to(self.device)
        attention_mask = tokenized_text['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            all_pred_probs = torch.softmax(logits, dim=-1)  #shape: (1, number of label classes)

        #get mappings for both desired labels (ssic/ subsector)
        #load label encoding map
        with open(self.config['mapping']['label_encoding_map']['path'], 'rb') as file:
            label_encoding_map = pickle.load(file)  #dict
        map_to_ssic = label_encoding_map

        #load ssic to subsector map
        mapping = pd.read_csv(self.config['mapping']['ssic_to_subsector_map']['path'], dtype=str)
        mapping_dict = dict(zip(mapping[self.config['mapping']['ssic_to_subsector_map']['ssic_col_name']], mapping[self.config['mapping']['ssic_to_subsector_map']['subsector_col_name']]))
        mapping_dict = {int(k): v for k, v in mapping_dict.items()}

        reverse_label_encoding_map = dict((v, k) for k, v in label_encoding_map.items())
        map_to_subsector = {}
        for k in reverse_label_encoding_map.keys():
            map_to_subsector[k] = mapping_dict[reverse_label_encoding_map[k]]

        #get predictions for ssic
        #for each text, get the top_n predicted ssics and their prediction probs
        top_n_pred_probs, top_n_pred_encoded_labels = torch.topk(all_pred_probs, top_n, dim=-1)
        top_n_pred_ssics = [str(reverse_label_encoding_map[encoded_label]) for encoded_label in top_n_pred_encoded_labels[0].cpu().numpy()]
        top_n_pred_probs = top_n_pred_probs[0].tolist()

        #create preds_df
        preds_df = pd.DataFrame([top_n_pred_ssics, top_n_pred_probs]).T
        preds_df.columns=['SSIC Code Prediction', 'Probability']
        
        #get top subsector from top ssic
        top_subsector = map_to_subsector[int(top_n_pred_encoded_labels[0][0])]
        
        return preds_df, top_subsector
    

class MultiInferenceClassifier:
    '''
    Used for predictions on test data and new inference data only


    Attributes
    ----------
    config: dict
        Project configurations in /config/main.yaml

    by_subsector: bool
        Whether to obtain predictions for Subsector label (True) or SSIC label (False)

    dataset_for_pred: Optional[pd.DataFrame] = None
        Dataset to obtain predictions for
        Predictions and prediction probabilities are added as new columns onto this dataset

    tokenized_texts_for_pred: List[torch.Tensor]
        Tokenized texts to obtain predictions for

    model: BertForSequenceClassification
        Ensemble model to be used for prediction
    '''


    #class attributes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #instance attributes
    def __init__(self,
                config: dict,
                by_subsector: bool,
                tokenized_texts_for_pred: List[torch.Tensor],
                model: BertForSequenceClassification,
                dataset_for_pred: Optional[pd.DataFrame] = None):
        self.config = config
        self.by_subsector = by_subsector
        self.tokenized_texts_for_pred = tokenized_texts_for_pred
        self.model = model

        if dataset_for_pred is not None:
            self.dataset_for_pred = dataset_for_pred
        else:
            self.dataset_for_pred = pd.read_excel(config['data']['inference_data']['path'])


    def get_predictions(self, top_n: int = 3) -> pd.DataFrame:
        '''
        Obtains predictions and prediction probabilities


        Parameters
        ----------
        top_n: int
            Number of predictions to be obtained per text.
            Method: For every text, obtain its prediction probability for all possible label classes.
                    Return the top n label classes with the highest probabilities.


        Returns
        -------
        preds_df: pd.DataFrame
            Dataset from 'self.dataset_for_pred' with predictions and prediction probabilities added as new columns
        '''


        #for every text, get prediction probs for all possible label classes
        input_ids = torch.cat([tokenized_text['input_ids'] for tokenized_text in self.tokenized_texts_for_pred], dim=0).to(self.device)
        attention_mask = torch.cat([tokenized_text['attention_mask'] for tokenized_text in self.tokenized_texts_for_pred], dim=0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            all_pred_probs = torch.softmax(logits, dim=-1)  #shape: (length of evaluation set, number of label classes)

        #get mappings for both desired labels (ssic/ subsector)
        #load label encoding map
        with open(self.config['mapping']['label_encoding_map']['path'], 'rb') as file:
            label_encoding_map = pickle.load(file)  #dict
        map_to_ssic = label_encoding_map

        #load ssic to subsector map
        mapping = pd.read_csv(self.config['mapping']['ssic_to_subsector_map']['path'], dtype=str)
        mapping_dict = dict(zip(mapping[self.config['mapping']['ssic_to_subsector_map']['ssic_col_name']], mapping[self.config['mapping']['ssic_to_subsector_map']['subsector_col_name']]))
        mapping_dict = {int(k): v for k, v in mapping_dict.items()}

        reverse_label_encoding_map = dict((v, k) for k, v in label_encoding_map.items())
        map_to_subsector = {}
        for k in reverse_label_encoding_map.keys():
            map_to_subsector[k] = mapping_dict[reverse_label_encoding_map[k]]

        #get predictions in the desired label (ssic/ subsector)
        #desired label is subsector
        if self.by_subsector: 
            #for each text, get the top_n predicted subsectors and their prediction probs
            top_n_pred_probs, top_n_pred_encoded_labels = torch.topk(all_pred_probs, top_n, dim=-1)
            top_n_pred_labels = [[map_to_subsector[encoded_label] for encoded_label in row] for row in top_n_pred_encoded_labels.cpu().numpy()]
            top_n_pred_probs = top_n_pred_probs.tolist()

        #desired label is ssic
        else:  
            #for each text, get the top_n predicted ssics and their prediction probs
            top_n_pred_probs, top_n_pred_encoded_labels = torch.topk(all_pred_probs, top_n, dim=-1)
            top_n_pred_labels = [[reverse_label_encoding_map[encoded_label] for encoded_label in row] for row in top_n_pred_encoded_labels.cpu().numpy()]
            top_n_pred_probs = top_n_pred_probs.tolist()

        #create dictionaries to store the top_n predictions and their prediction probs
        top_n_pred_labels_dict = {f'top_{n+1}': [] for n in range(top_n)}
        top_n_pred_probs_dict = {f'top_{n+1}': [] for n in range(top_n)}

        #populate the dictionaries
        for row_labels, row_probs in zip(top_n_pred_labels, top_n_pred_probs):
            for n, (label, prob) in enumerate(zip(row_labels, row_probs)):
                top_n_pred_labels_dict[f'top_{n+1}'].append(label)
                top_n_pred_probs_dict[f'top_{n+1}'].append(prob)

        #add dictionary information to original dataset
        preds_df = self.dataset_for_pred.copy()
        for n in range(top_n):
            preds_df[f'rank{n+1}_pred'] = top_n_pred_labels_dict[f'top_{n+1}']
            preds_df[f'rank{n+1}_prob'] = top_n_pred_probs_dict[f'top_{n+1}']

        return preds_df

    def get_classification_report(self, preds_df: pd.DataFrame, for_top_n: int) -> pd.DataFrame: 
        '''
        Only used when true labels exist
        Not used on new inference data (without true labels)

        Get evaluation metrics for prediction
            For each label class prediction
            - Precision
            - Recall
            - F1-score

            For overall prediction
            - Macro Average Precision
            - Macro Average Recall
            - Macro Average F1-score
            - Accuracy


        Parameters
        ----------
        preds_df: pd.DataFrame
            Dataset from 'self.dataset_for_pred' with predictions and prediction probabilities added as new columns
            Output of get_predictions() method in this class

        for_top_n: int
            How many top n predictions to consider when determining if prediction is correct: Prediction is considered correct if true label matches one of the top_n predictions
            - If for_top_n=1, rank1_pred must be same as true for model prediction to be considered correct
            - If for_top_n=2, either rank1_pred or rank2_pred must be same as true for model prediction to be considered correct
            - And so on


        Returns
        -------
        classification_report: pd.DataFrame
            Evaluation metrics for prediction
        '''

        #if desired label is subsector
        if self.by_subsector:
            #get true subsector labels from true ssic labels
            ssic_to_subsector_map = pd.read_csv(self.config['mapping']['ssic_to_subsector_map']['path'], dtype=str)
            ssic_col_name = self.config['mapping']['ssic_to_subsector_map']['ssic_col_name']
            subsector_col_name = self.config['mapping']['ssic_to_subsector_map']['subsector_col_name']
            ssic_to_subsector_map = dict(zip(ssic_to_subsector_map[ssic_col_name], ssic_to_subsector_map[subsector_col_name]))  #dict: {'ssic': 'subsector'}
            ssic_to_subsector_map = {int(k): v for k, v in ssic_to_subsector_map.items()}
            preds_df['subsector'] = preds_df['ssic'].apply(lambda x: ssic_to_subsector_map[x])

            #instantiate true_label_col_name as 'subsector'
            true_label_col_name = 'subsector'

        #if desired label is ssic
        else:
            #instantiate true_label_col_name as 'ssic'
            true_label_col_name = 'ssic'

        #add new col, correct_pred, to preds_df: 1 if prediction is correct, 0 otherwise
        preds_df['correct_pred'] = 0
        for i in range(for_top_n):
            preds_df['correct_pred'] = np.where(preds_df[true_label_col_name] == preds_df[f'rank{i+1}_pred'], 1, preds_df['correct_pred'])

        #add new col, final_pred, which contains the correct predicted class if the prediction is correct and the top 1 predicted class otherwise
        condition = [preds_df['correct_pred'] == 1, preds_df['correct_pred'] == 0]
        values = [preds_df[true_label_col_name], preds_df['rank1_pred']]
        preds_df['final_pred'] = np.select(condition, values, default=np.nan)
        
        #compute evaluation metrics for each label class
        metrics_per_class = classification_report(y_true=preds_df[true_label_col_name], y_pred=preds_df['final_pred'], output_dict=True)
        unique_classes = preds_df[true_label_col_name].unique().tolist()
        unique_classes.sort()
        unique_classes = [str(x) for x in unique_classes]
        unique_classes.append('macro avg')
        
        #create new df, report, that contains evaluation metrics for each label class prediction and for overall prediction
        class_list = []
        count = []
        precision = []
        recall = []
        f1_score = []
        
        for c in unique_classes:
            class_list.append(c)
            precision.append(round(metrics_per_class[c]['precision'], 2))
            recall.append(round(metrics_per_class[c]['recall'], 2))
            f1_score.append(round(metrics_per_class[c]['f1-score'], 2))
            count.append(round(metrics_per_class[c]['support'], 2))
        
        overall_acc = round(metrics_per_class['accuracy'], 2)
        class_list.append(f'Overall Prediction Accuracy: {overall_acc*100: .1f}%')   
        count.append('')
        precision.append('')
        recall.append('')
        f1_score.append('')
        
        report = pd.DataFrame({'Class': class_list, 
                                        'Count': count, 
                                        'Precision': precision, 
                                        'Recall': recall, 
                                        'F1_Score': f1_score})
        
        return report

