from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter():
    '''
    This class is only used for model training
    Splits dataset into train, test, and val for model training and evaluation

    Attributes
    ----------
    config: dict
        Project configurations in /config/main.yaml

    data: pd.DataFrame
        Data for model training

    test: pd.DataFrame
        Test set for classification task, used to obtain model performance on unseen data

    val: pd.DataFrame
        Validaton set for classification task, used to determine optimal model

    train: pd.DataFrame
        Training set for classification task
    '''

    def __init__(self, config: dict):
        self.config = config

        data = pd.read_csv(config['data']['training_data']['path'])
        #standardize colnames
        data = data.rename(columns={config['data']['training_data']['text_col_name']: 'desc', 
                                    config['data']['training_data']['label_col_name']: 'ssic',
                                    config['data']['training_data']['source_col_name']: 'source',
                                    config['data']['training_data']['is_synthetic_col_name']: 'is_syn'})
        self.data = data

    def _remove_text_col_duplicates(self) -> None:
        data = self.data
        data['desc'] = data['desc'].apply(lambda x: x.lower())
        self.data = data.drop_duplicates(subset='desc')

    def _get_test_set(self, test_rows_per_ssic: int = 20, noisily: bool = True) -> None:
        '''
        Creates test set = min(20, mpa_yes) for each label category


        Parameters
        ----------
        test_rows_per_ssic: int = 20
            Number of rows per label category inside mpa_yes to be left out for test set

        noisily: bool = True
            Whether to print out number of rows in test set per label category


        Attributes Instantiated
        -----------------------
        test: pd.DataFrame
            Test set for classification task
        '''

        data = self.data
        mpa_yes = data.loc[(data['source'] == 'mpa_yes') & (data['is_syn'] == 0)]
        test = mpa_yes.groupby('ssic').apply(lambda x: x.sample(min(test_rows_per_ssic, int(x.shape[0]/2)))).droplevel(level=0)  #drop the first level of the multi-level index
        self.test = test
        
        if noisily:
            print('number of test rows per ssic')
            print(test.groupby('ssic').size())  #to see the ssics with very little test rows

    def _get_val_set(self) -> None:
        '''
        Attributes Instantiated
        -----------------------
        val: pd.DataFrame
            Validaton set for classification task, used to determine optimal model
        '''
        test = self.test

        test, val = train_test_split(test, test_size=0.5, stratify=test['ssic'], )
        self.test = test
        self.val = val

    def _get_training_set(self) -> None:
        '''
        Attributes Instantiated
        -----------------------
        train: pd.DataFrame
            Training set for classification task
        '''

        data = self.data

        test_index = self.test.index
        train = data.loc[~data.index.isin(test_index)]
        self.train = train

    def _balance_training_set(self, upper_bound: int = 100, add_mpa_ambi: bool = True) -> None:
        '''
        Through undersampling
        Set threshold = 100, so if SSIC has > 100 rows, prioritize (ssic book + alpha index) > mpa_yes > nlp augmented desc
        Required due to heavily imbalanced training set: If there are certain ssics that have many rows, model will be trained biased towards them
        '''

        train = self.train

        ssic_counts = train.groupby('ssic').size()
        large_ssics = ssic_counts.loc[lambda x: x > upper_bound].index

        overall_res = train.loc[~train['ssic'].isin(large_ssics)]  #small ssics
        for ssic in large_ssics:
            ssic_df = train.loc[train['ssic'] == ssic]

            alpha_index_ssic_book = ssic_df.loc[((ssic_df['source'] == 'alpha_index') | (ssic_df['source'] == 'ssic_book')) & (ssic_df['is_syn'] == 0)]
            mpa_yes = ssic_df.loc[(ssic_df['source'] == 'mpa_yes') & (ssic_df['is_syn'] == 0)]
            mpa_ambi = ssic_df.loc[(ssic_df['source'] == 'mpa_ambi') & (ssic_df['is_syn'] == 0)]
            alpha_index_ssic_book_syn = ssic_df.loc[((ssic_df['source'] == 'alpha_index') | (ssic_df['source'] == 'ssic_book')) & (ssic_df['is_syn'] == 1)]
            mpa_yes_syn = ssic_df.loc[(ssic_df['source'] == 'mpa_yes') & (ssic_df['is_syn'] == 1)]
            mpa_ambi_syn = ssic_df.loc[(ssic_df['source'] == 'mpa_ambi') & (ssic_df['is_syn'] == 1)]

            ssic_res = pd.DataFrame()
            if add_mpa_ambi:
                for data in [alpha_index_ssic_book, mpa_yes, mpa_ambi, alpha_index_ssic_book_syn, mpa_yes_syn, mpa_ambi_syn]:  #in order of priority
                    ssic_res = pd.concat([ssic_res, data], axis=0)
                    if len(ssic_res) > upper_bound:
                        diff = len(ssic_res) - upper_bound
                        ssic_res = ssic_res.iloc[:-diff]
                        break
            else:
                for data in [alpha_index_ssic_book, mpa_yes, alpha_index_ssic_book_syn, mpa_yes_syn]:  #in order of priority
                    ssic_res = pd.concat([ssic_res, data], axis=0)
                    if len(ssic_res) > upper_bound:
                        diff = len(ssic_res) - upper_bound
                        ssic_res = ssic_res.iloc[:-diff]
                        break

            overall_res = pd.concat([overall_res, ssic_res], axis=0)

        self.train = overall_res

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        Returns
        -------
        test: pd.DataFrame
            Test set for classification task, used to obtain model performance on unseen data

        val: pd.DataFrame
            Validaton set for classification task, used to determine optimal model

        train: pd.DataFrame
            Training set for classification task
        '''

        self._remove_text_col_duplicates()
        self._get_test_set()
        self._get_val_set()
        self._get_training_set()
        self._balance_training_set()

        return self.train, self.test, self.val