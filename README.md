[//]: # (# autopilot_clean)

Clean repo for patient individual platelet demands (24-hr scope) in AutoPilot

Task: This study aimed to create a personalized support system for PC demand management. Based on a deep learning-computed risk score, this system aims to predict a patient's PC requirements within the next 24 hours. To achieve this, we utilized data from various clinical information technology (IT) systems.

End-to-end pipeline options: 
1. Loading + transformation of multimodal data and training of rnn-based models: nn_training_pipeline()
2. Utilizing the trained models for live predictions: live_pipeline()
3. Training and evaluation of rf-xgb model on pre-processed data: launch_ml_training()
4. Explaining the rf-xgb model: explain_train.main(config)
5. Explaining the rnn-based models: explain_rnn_pipline()
6. Cohort Analysis and evaluation of rnn models: publish_pipline()

## Installation
docker compose build autopilot_predict

or

poetry install