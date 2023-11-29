from typing import List, Optional
import pandas as pd
import numpy as np
import optuna
from transformers import BertForSequenceClassification
import torch
from tqdm import tqdm
import joblib
from copy import deepcopy


class SSICRecommenderDataset(torch.utils.data.Dataset):
    '''
    Custom dataset class compatible with PyTorch's DataLoader
    Used to efficiently load and preprocess data as required for model training and evaluation
    This class is required as torch.utils.data.Dataset is an abstract class and cannot be used directly without creating a subclass with specific implementations
    
    Attributes
    ----------
    tokenized_texts: List[torch.Tensor]
        Texts converted into tensors, to be used as inputs to the model

    encoded_labels: List[int]
        The target of the classification task
    '''

    def __init__(self, tokenized_texts: List[torch.Tensor], encoded_labels: List[int]):
        self.tokenized_texts = tokenized_texts 
        self.encoded_labels = encoded_labels

    def __len__(self) -> int:
        '''
        Required method in order to work with PyTorch's DataLoader

        Returns
        -------
        The total number of rows in the dataset
        '''
         
        return len(self.encoded_labels)
    
    def __getitem__(self, index):
        '''
        Required method in order to work with PyTorch's DataLoader

        Returns
        -------
        A tuple of (X, y) where
            X: torch.Tensor
                The tokenized_text at the specified index

            y: torch.Tensor
                The label (target) at the specified index
        '''
        X = self.tokenized_texts[index]
        y = torch.tensor(self.encoded_labels[index])
        return (X, y)


class ModelEvaluator:
    '''
    Obtains model performance in terms of specified loss function on a specified dataset


    Class Attributes
    ----------------
    device: torch.device
        The type of device to run on, either CPU or GPU


    Instance Attributes
    ----------
    model: BertForSequenceClassification
        The model to be evaluated

    eval_texts: List[torch.Tensor]
        Tokenized texts for evaluating the model

    eval_labels: List[int]
        Target encoded into integers for evaluating the model

    loss_function: Optional[torch.nn.Module] = None
        Loss function for gradient descent, used in both model training and model evaluation
        Defaults to CrossEntropyLoss as this is a classification task
    '''

    #class attributes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #instance attributes
    def __init__(self, model: BertForSequenceClassification, eval_texts: List[torch.Tensor], eval_labels: List[int], loss_function: torch.nn.Module):
        self.model = model
        self.eval_texts = eval_texts
        self.eval_labels = eval_labels
        self.loss_function = loss_function

    def evaluate(self) -> float:
        #create DataLoader object for data for evaluation
        val = SSICRecommenderDataset(tokenized_texts=self.eval_texts, encoded_labels=self.eval_labels)
        val_dataloader = torch.utils.data.DataLoader(dataset=val, batch_size=len(val))

        #set model to eval mode
        model = self.model.eval()

        #evaluate model on val set
        #initialize total val loss to 0
        total_loss_val = 0

        with torch.no_grad():  #disable gradient computation to speed up computation
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(self.device)
                mask = val_input['attention_mask'].squeeze(1).to(self.device)
                input_id = val_input['input_ids'].squeeze(1).to(self.device)

                output = model(input_id, mask)

                #get loss
                batch_loss = self.loss_function(output.logits, val_label.long())
                total_loss_val += batch_loss.item()      

        return total_loss_val / len(val_dataloader)


class ModelTuner:
    '''
    Class Attributes
    ----------------
    device: torch.device
        The type of device to run on, either CPU or GPU
    
    
    Instance Attributes
    -------------------
    config: dict
        Project configurations in /config/main.yaml
    
    train_texts: List[torch.Tensor]
        Tokenized texts for training the ensemble model

    train_labels: List[int]
        Target encoded into integers for training the ensemble model

    val_texts: List[torch.Tensor]
        Tokenized texts for evaluating the model (Validation set used to determine the optimal ensemble model)
        Note that this is different from the test set, which is held out until the optimal ensemble model has been identified, to determine its performance on new unseen data

    val_labels: List[int]
        Target encoded into integers for evaluating the model

    loss_function: Optional[torch.nn.Module] = None
        Loss function for gradient descent, used in both model training and model evaluation
        Defaults to CrossEntropyLoss as this is a classification task
    '''

    #class attributes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #instance attributes
    def __init__(self,
                config: dict, 
                train_texts: List[torch.Tensor], train_labels: List[int], 
                val_texts: List[torch.Tensor], val_labels: List[int],
                loss_function: Optional[torch.nn.Module] = None):
        self.config = config
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.val_texts = val_texts
        self.val_labels = val_labels
        if loss_function is None:
            self.loss_function = torch.nn.CrossEntropyLoss().to(ModelEnsembler.device)  #default loss_function is CrossEntropyLoss
        else:
            self.loss_function = loss_function.to(ModelEnsembler.device)

    def tune_hyperparams(self, num_trials: int = 10) -> List[BertForSequenceClassification]:
        '''
        Fine-tunes pre-trained BERT model using Optuna optimization framework
        Uses hyperparameter search space defined in project configuration file


        Parameters
        ----------
        num_trials: int, default = 10
            Number of optuna trials to run hyperparameter tuning for


        Returns
        -------
        All fine-tuned models sorted from best performing to worst: List[BertForSequenceClassification]
        '''
    
        def objective(trial: optuna.Trial) -> float:
            '''
            Returns
            -------
            Model performance in terms of loss: float
            '''

            #get hyperparam values to be used in this trial
            hyperparam_search_space = self.config['model_tuning']['hyperparam_search_space']
            batch_size = trial.suggest_int('batch_size', hyperparam_search_space['batch_size_range'][0], hyperparam_search_space['batch_size_range'][1])
            learning_rate = trial.suggest_float('learning_rate', hyperparam_search_space['learning_rate_range'][0], hyperparam_search_space['learning_rate_range'][1])
            epochs = trial.suggest_int('epochs', hyperparam_search_space['epoch_range'][0], hyperparam_search_space['epoch_range'][1])
            weight_decay = trial.suggest_float('weight_decay', hyperparam_search_space['weight_decay_range'][0], hyperparam_search_space['weight_decay_range'][1])
            frozen_layers = hyperparam_search_space['frozen_layers']
            np.random.seed(self.config['random_seed'])
            torch.manual_seed(self.config['random_seed'])

            #create a DataLoader object for training set
            train = SSICRecommenderDataset(tokenized_texts=self.train_texts, encoded_labels=self.train_labels)
            train_dataloader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)  #shuffle the dataset at each epoch to reduce the impact of the order of samples on the model's learning

            #define the pre-trained model to be fine-tuned
            model = BertForSequenceClassification.from_pretrained(self.config['pretrained_model']['path'], num_labels=len(set(self.train_labels)), output_attentions=False, output_hidden_states=False)
            
            #freeze the first {frozen_layers} number of model layers during fine-tuning: these layers will not have their weights fine-tuned
            #{frozen_layers} set to 0 in proj config bc: it's common to fine-tune all layers of BERT without freezing any layers, especially for tasks like text classification. 
            #the reason is that all layers of BERT contain valuable information for understanding the semantics of text, and freezing any layers might prevent the model from fully adapting to the new task.
            for name, param in model.named_parameters():
                for i in range(frozen_layers):
                    if name.startswith(f'bert.encoder.layer.{i}'):
                        param.requires_grad = False

            #move model to GPU or CPU accordingly
            model = model.to(self.device)

            #define the optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            #for each epoch, train model and log performance on val set
            for epoch_num in range(epochs):
                #set model to training mode
                model.train()
                
                #initialize total training loss to 0
                total_loss_train = 0

                #iterate through batches of training set and perform forward and backward passes
                for train_input, train_label in tqdm(train_dataloader):
                    #move input data, input labels and attention mask to GPU or CPU accordingly
                    train_label = train_label.to(self.device)
                    mask = train_input['attention_mask'].squeeze(1).to(self.device)
                    input_id = train_input['input_ids'].squeeze(1).to(self.device)

                    #forward pass: obtain model output using input data and attention masks
                    output = model(input_id, mask)

                    #compute this batch's loss using loss function and logits from model output
                    batch_loss = self.loss_function(output.logits, train_label.long())
                    total_loss_train += batch_loss.item()  #update total training loss as sum of each batch's loss

                    #backward pass: update model weights using this batch's loss
                    model.zero_grad()  #zero the gradients of model parameters to avoid accumulation
                    batch_loss.backward()  #backward pass: compute gradients with respect to this batch's loss
                    optimizer.step()  #update model weights using optimizer (AdamW)

                #for this epoch, evaluate trained model performance on val set
                avg_loss = ModelEvaluator(model=model, eval_texts=self.val_texts, eval_labels=self.val_labels, loss_function=self.loss_function).evaluate()

                #for this epoch, get average loss over both training and val set
                print(f'Epoch {epoch_num+1} | Train Loss: {total_loss_train / len(train_dataloader): .2f}\
                        | Val Loss: {avg_loss: .2f}')

            #save trained model for this trial (= 1 hyperparam set tried out)
            directory = self.config['model_tuning']['finetuned_models_directory']
            file_name = 'trial_' + str(trial.number)
            model.save_pretrained(directory + '/' + file_name)

            #save self.study object to file
            joblib.dump(self.study, directory + '/' + 'optuna_tuning_study.pkl')

            #return model performance
            return avg_loss  #the last epoch's avg_loss

        #use objective() function defined above for optuna optimization
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(func=objective, n_trials=num_trials)

        #need to save self.study object to file again else the last trial will not be captured?
        #not sure if this solves it tho
        joblib.dump(self.study, directory + '/' + 'optuna_tuning_study.pkl')

        #get all optuna trials tried out, sorted from best performing to worst
        sorted_trial_nums = self.study.trials_dataframe().sort_values(by=['value'], ascending=True)['number']
        sorted_finetuned_model_files = list(sorted_trial_nums.apply(lambda x: 'trial_' + str(x)))

        #save sorted_hyperparam_model_files to file
        joblib.dump(sorted_finetuned_model_files, directory + '/' + 'sorted_finetuned_model_files.pkl')

        return sorted_finetuned_model_files


class ModelEnsembler:
    '''
    Class Attributes
    ----------------
    device: torch.device
        The type of device to run on, either CPU or GPU
    
    
    Instance Attributes
    -------------------
    config: dict
        Project configurations in /config/main.yaml

    val_texts: List[torch.Tensor]
        Tokenized texts for evaluating the model (Validation set used to determine the optimal ensemble model)
        Note that this is different from the test set, which is held out until the optimal ensemble model has been identified, to determine its performance on new unseen data

    val_labels: List[int]
        Target encoded into integers for evaluating the model

    loss_function: Optional[torch.nn.Module] = None
        Loss function for gradient descent, used in both model training and model evaluation
        Defaults to CrossEntropyLoss as this is a classification task
    '''

    #class attributes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #instance attributes
    def __init__(self,
                config: dict, 
                val_texts: List[torch.Tensor], val_labels: List[int],
                loss_function: Optional[torch.nn.Module] = None):
        self.config = config
        self.val_texts = val_texts
        self.val_labels = val_labels
        if loss_function is None:
            self.loss_function = torch.nn.CrossEntropyLoss().to(ModelEnsembler.device)  #default loss_function is CrossEntropyLoss
        else:
            self.loss_function = loss_function.to(ModelEnsembler.device)

    def create_model_soup(self, sorted_finetuned_model_files: List[str]) -> BertForSequenceClassification:
        '''
        Notes
        -----
        Ensembles models (= a model soup) by averaging weights of multiple fine-tuned large pre-trained models, see https://arxiv.org/abs/2203.05482
        
        Method to determine the optimal model ensemble:
        - Sort all the fine-tuned models by performance, from best to worst
        - Instiantiate the model soup as the best-performing fine-tuned model
        - Determines whether or not to add the next fine-tuned model to the soup by checking if adding it improves the performance of the soup
        

        Parameters
        ----------
        sorted_finetuned_model_files: List[str]
            Filenames of models finetuned using Optuna, sorted by best performing model to worst


        Returns
        -------
        The optimal ensembled model: BertForSequenceClassification
        '''
        copied_sorted_finetuned_model_files = sorted_finetuned_model_files.copy()
        best_model_file = copied_sorted_finetuned_model_files.pop(0)  #removes the first model (which is the best model bc the list is sorted) from copied_sorted_finetuned_model_files in-place
        
        #instiantiate best model as this first model
        finetuned_models_directory = self.config['model_tuning']['finetuned_models_directory']
        best_model = BertForSequenceClassification.from_pretrained(finetuned_models_directory + '/' + best_model_file)
        best_model_weights = deepcopy(best_model.state_dict())  #a dict

        #initialize best_soupas as a dictionary with the same keys as best_model_weights, but whose values are all empty lists
        #model weights will later be added to these empty dict values
        best_soup = {key:[] for key in best_model_weights}

        #add the best model weights into best_soup
        for k, v in best_model_weights.items():
            best_soup[k].append(v)

        #iterate through all the remaining hyperparam models
        for model_file in copied_sorted_finetuned_model_files:  #copied_sorted_finetuned_model_files has the first model (which is the best model) removed already
            candidate_model = BertForSequenceClassification.from_pretrained(finetuned_models_directory + '/' + model_file)
            candidate_model_weights = deepcopy(candidate_model.state_dict())

            #instantiate candidate_soup as best_soup
            candidate_soup = deepcopy(best_soup)
            #add this candidate model's weights to candidate soup's weights
            #so the only diff between best_soup and candidate_soup is that candidate_soup has this iteration's candidate_model added to it
            for k, v in candidate_model_weights.items():
                candidate_soup[k].append(v)

            #get weights of best soup and candidate soup by averaging the weights of the models in each soup
            #via dict comprehension: 
            #for every k, v in best_soup.items() where v is a list of model weights,
            #if v is non-empty ie. there are weight tensors in the list,
            #get the element-wise average of each weight tensor
            #and ensure that this averaged tensor has the same data type as the original tensors in the list v
            best_soup_weights = {k:(torch.sum(torch.stack(v), axis=0) / len(v)).type(v[0].dtype) for k, v in best_soup.items() if len(v) != 0}
            candidate_soup_weights = {k:(torch.sum(torch.stack(v), axis=0) / len(v)).type(v[0].dtype) for k, v in candidate_soup.items() if len(v) != 0}

            #update the weights of best_model and candidate_model
            best_model.load_state_dict(best_soup_weights)
            candidate_model.load_state_dict(candidate_soup_weights)       

            #evaluate best model and candidate model
            best_loss = ModelEvaluator(model=best_model, eval_texts=self.val_texts, eval_labels=self.val_labels, loss_function=self.loss_function).evaluate()
            candidate_loss = ModelEvaluator(model=candidate_model, eval_texts=self.val_texts, eval_labels=self.val_labels, loss_function=self.loss_function).evaluate()

            #if performance improves
            if candidate_loss < best_loss:
                #add this iteration's candidate model to best soup
                for k, v in candidate_model_weights.items():
                    best_soup[k].append(v)
                print(f'Added {model_file} to best soup. New soup loss: {candidate_loss: .2f}, Old soup loss: {best_loss: .2f}')
            else:
                print(f'Rejected {model_file}. Current soup loss: {best_loss: .2f}, Rejected soup loss: {candidate_loss: .2f}')

        #get final best soup
        best_soup_weights = {k:(torch.sum(torch.stack(v), axis=0) / len(v)).type(v[0].dtype) for k, v in best_soup.copy().items() if len(v) != 0}
        best_model.load_state_dict(best_soup_weights.copy())

        #save final soup
        best_model.save_pretrained(self.config['model_ensemble']['ensemble_model_directory'])
        return best_model
