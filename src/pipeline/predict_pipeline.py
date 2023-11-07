from src.utils import load_object


class PredictPipeline:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = load_object(model_path)

    def predict(self, img):
        output = self.model(img)
        prediction = output.argmax(dim=1, keepdim=True).item()
        return prediction