from src.get_config import get_config
from src.split_data import DataSplitter
from src.preprocess_data import TrainingDataPreprocessor, TestDataPreprocessor
from src.train_ensemble_model import ModelTuner, ModelEnsembler
from src.classifier import Classifier


def main():
    config = get_config(path='/config/main.yaml')

    #split data
    splitter = DataSplitter(config=config)
    train, test, val = splitter.split()

    #preprocess and tokenize data
    train_preprocessor = TrainingDataPreprocessor(config=config, data=train)
    train_texts, train_labels = train_preprocessor.process()

    test_preprocessor = TestDataPreprocessor(config=config, data=test)
    test_texts, test_labels = test_preprocessor.process()

    val_preprocessor = TestDataPreprocessor(config=config, data=val)
    val_texts, val_labels = val_preprocessor.process()

    #fine-tune multiple BERT models
    tuner = ModelTuner(config=config,
                   train_texts=train_texts, train_labels=train_labels,
                   val_texts=val_texts, val_labels=val_labels)

    sorted_finetuned_model_files = tuner.tune_hyperparams(num_trials=5)
    
    #ensemble the fine-tuned models via model soup methodology
    ensembler = ModelEnsembler(config=config,
                           val_texts=val_texts, val_labels=val_labels)

    ensembled_model = ensembler.create_model_soup(sorted_finetuned_model_files)

    #evaluate ensembled model on test set
    #1) evaluate based on SSIC predictions
    ssic_classifier = Classifier(config=config, by_subsector=False, dataset_for_pred=test, tokenized_texts_for_pred=test_texts, model=ensembled_model)
    ssic_preds_df = ssic_classifier.get_predictions(top_n=3)
    ssic_classification_report_top1 = ssic_classifier.get_classification_report(preds_df=ssic_preds_df, for_top_n=1)

    #2) evaluate based on subsector predictions
    subsector_classifier = Classifier(config=config, by_subsector=True, dataset_for_pred=test, tokenized_texts_for_pred=test_texts, model=ensembled_model)
    subsector_preds_df = subsector_classifier.get_predictions(top_n=3)
    subsector_classification_report_top1 = subsector_classifier.get_classification_report(preds_df=subsector_preds_df, for_top_n=1)

    #save evaluation metrics
    ssic_classification_report_top1.to_csv(config['model_evaluation']['ssic_eval_path'], index=False)
    subsector_classification_report_top1.to_csv(config['model_evaluation']['subsector_eval_path'], index=False)


if __name__ == "__main__":
    main()
