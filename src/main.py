import logging

from src.data.data_loader import DataPreparator
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.train_pipeline import Trainer

if __name__ == '__main__':
    model_path = '../src/model/artifacts/model.pkl'
    num_epochs = 11
    data_loader = DataPreparator()
    train_data, test_data, loaders = data_loader.get_data()

    model_trainer = Trainer()

    for epoch in range(1, num_epochs):
        model_trainer.train(epoch, data_loader.loaders)
        model_trainer.test(data_loader.loaders)

    model_trainer.save(model_path)

    predict_pipeline = PredictPipeline(model_path)
    data, target = test_data[5]
    data = data.unsqueeze(0).to(model_trainer.device)
    pred = predict_pipeline.predict(data)
    print(f'Prediction: {pred}, target: {target}')

