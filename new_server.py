import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.server.strategy import FedAvg
import os
import tensorflow as tf
from tensorflow.keras.models import save_model
from tensorflow.keras.optimizers import Adam
from flwr.common import Parameters

# Make TensorFlow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enable memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define a function to save the global model
def save_global_model(model):
    save_path = "global_model_final_50_rounds.h5"
    save_model(model, save_path)
    print(f"Global model saved to {save_path}")


def weighted_average(metrics):
    total_examples = 0
    federated_metrics = {k: 0 for k in metrics[0][1].keys()}
    for num_examples, m in metrics:
        for k, v in m.items():
            federated_metrics[k] += num_examples * v
        total_examples += num_examples
    return {k: v / total_examples for k, v in federated_metrics.items()}


# Define a custom FedAvg strategy that saves the global model after the final round
class CustomFedAvg(FedAvg):
    def __init__(self, model, num_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.num_rounds = num_rounds

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        # Call the superclass method to aggregate the weights
        aggregated_weights, metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_weights is not None:
            # Extract the weights from the aggregated Parameters object
            weights = fl.common.parameters_to_ndarrays(aggregated_weights)
            # Convert weights to a Keras model
            self.model.set_weights(weights)
            # Save the model if this is the last round
            if rnd == self.num_rounds:
                save_global_model(self.model)
        
        # Clear session to free up memory
        tf.keras.backend.clear_session()
        
        return aggregated_weights, metrics

# Initialize the model
def initialize_model():
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(7, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Set the number of rounds
num_rounds = 50

# Create the model and strategy
model = initialize_model()
strategy = CustomFedAvg(
    model=model,
    num_rounds=num_rounds,
    min_available_clients=5,
    min_evaluate_clients=5,
    min_fit_clients=5,
    fit_metrics_aggregation_fn=weighted_average,
    evaluate_metrics_aggregation_fn=weighted_average
)

# Start Flower server for a specified number of rounds of federated learning
history = fl.server.start_server(
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=num_rounds)
)

final_round, acc = history.metrics_distributed["accuracy"][-1]
print(f"After {final_round} rounds of training, the accuracy of the global model is {acc:.3%}")
