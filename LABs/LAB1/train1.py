"""Part 1: try the linear and logistic models on small example data."""

import numpy as np

from model import LinearRegressionModel, LogisticRegressionModel


def main() -> None:
    features = np.array(
        [[-2, -1], [-1, -1], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 1], [2, 1]],
        dtype=np.float64,
    )
    linear_targets = 2 * features[:, 0] - features[:, 1] + 1
    logistic_targets = (features[:, 0] + features[:, 1] > 0).astype(np.float64)

    linear_model = LinearRegressionModel(features.shape[1])
    linear_model.fit(features, linear_targets)
    linear_predictions = linear_model.predict(features)
    print(f"Linear MSE: {np.mean((linear_predictions - linear_targets) ** 2):.4f}")

    logistic_model = LogisticRegressionModel(features.shape[1])
    logistic_model.fit(features, logistic_targets)
    logistic_predictions = logistic_model.predict(features)
    accuracy = np.mean((logistic_predictions >= 0.5) == logistic_targets)
    print(f"Logistic accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
