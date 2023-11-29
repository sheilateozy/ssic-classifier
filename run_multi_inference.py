from src.get_config import get_config
from src.preprocess_data import MultiInferenceDataPreprocessor
from transformers import BertForSequenceClassification
from src.classifier import MultiInferenceClassifier


def main():
    config = get_config(path='/config/main.yaml')

    #preprocess and tokenize text
    inf_preprocessor = MultiInferenceDataPreprocessor(config=config)
    inf_texts = inf_preprocessor.process()
    
    #load ensembled model
    ensembled_model = BertForSequenceClassification.from_pretrained(config['model_ensemble']['ensemble_model_directory'])
        
    #use ensembled model for inference
    #1) obtain SSIC predictions
    ssic_classifier = MultiInferenceClassifier(config=config, by_subsector=False, tokenized_texts_for_pred=inf_texts, model=ensembled_model)
    ssic_preds_df = ssic_classifier.get_predictions(top_n=3)

    #2) obtain subsector predictions
    subsector_classifier = MultiInferenceClassifier(config=config, by_subsector=True, tokenized_texts_for_pred=inf_texts, model=ensembled_model)
    subsector_preds_df = subsector_classifier.get_predictions(top_n=3)

    #save predictions
    ssic_preds_df.to_csv(config['model_prediction']['ssic_preds_path'], index=False)
    subsector_preds_df.to_csv(config['model_prediction']['subsector_preds_path'], index=False)


if __name__ == "__main__":
    main()
