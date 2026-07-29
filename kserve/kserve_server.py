import joblib
import os

from kserve import Model, ModelServer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "champion_model.pkl"
)


class GridSightModel(Model):

    def __init__(self, name):
        super().__init__(name)
        self.ready = False

    def load(self):

        print("Loading GridSight champion model...")

        champion = joblib.load(MODEL_PATH)

        import predictor.predictor as predictor

        predictor.champion = champion
        predictor.delay_model = champion["delay_model"]
        predictor.cost_model = champion["cost_model"]
        predictor.encoders = champion["encoders_and_features"]

        self.ready = True

        print("Champion model loaded successfully.")

    def predict(self, payload, headers=None):

        from predictor.predictor import predict_for_project

        instances = payload.get("instances", [])

        predictions = []

        for instance in instances:
            predictions.append(
                predict_for_project(instance)
            )

        return {
            "predictions": predictions
        }


if __name__ == "__main__":

    model = GridSightModel("gridsight")

    model.load()

    ModelServer().start([model])
