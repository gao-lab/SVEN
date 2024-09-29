## Codes for SVEN

This folder contains core codes for SVEN package.

#### Files description
| Files | Description |
| --- | --- |
| ```blocks.py``` | Blocks in SVEN annotaion models: ```conv_block```, ```Rconv_block``` and  ```dilated_residual```. |
| ```data.py``` | Codes for data processing. ```SeqDataset```: build dataset in training annotation models; ```onehot_code```: one-hot encoding; ```load_data```: build dataset in customizing SVEN prediction models.|
| ```layers.py``` | Codes for customized layers used in SVEN annotation models. ```StochasticShift```: layer for stochastic shifting input sequences; ```StochasticReverseComplement```: layer for stochastic getting reverse complement sequence. | 
| ```metrics.py``` | Codes for customized metrics used in training SVEN annotation models. ```PearsonR```: calculate Pearson correlation. |
| ```models.py``` | Codes for SVEN annotation models. Class-oriented holistic models: ```acc_hol_model```, ```his_hol_model```, ```tf_hol_model```; feature-oriented separate models: ```sep_f1_model```,```sep_f2_model```. |
| ```predict.py``` | Codes for running SVEN annotation module. ```run_hol_prediction```: run class-oriented holistic models; ```run_sep_prediction```: run feature-oriented separate models; ```model_predict_fast```: run SVEN prediction module with fast mode; ```model_predict_full```: run SVEN prediction module with full mode. |
| ```train.py``` | Codes for customizing SVEN. ```enformer_predict```: run SVEN annotation module with Enformer model; ```enformer_transform```: transform features predicted by Enformer model; ```train_xgb```: train gradient boosting tree models; ```train_elasticNet```: train elasticNet models. |
| ```utils.py``` | Other functions used in SVEN. |
