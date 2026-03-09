# """
# Loss functions for SACNN training
# """

# import tensorflow as tf
# from tensorflow import keras
# from keras import backend as K

# def euclidean_loss(y_true, y_pred):
#     """
#     Euclidean loss for density map regression
#     L2 loss between predicted and ground truth density maps
    
#     Args:
#         y_true: Ground truth density map 
#         y_pred: Predicted density map
    
#     Returns:
#         Euclidean loss value
#     """
#     return K.mean(K.square(y_pred - y_true))


# def relative_count_loss(y_true, y_pred):
#     """
#     Relative count loss 
    
#     L = ((pred_count - true_count) / (true_count + 1))^2
    
#     Args:
#         y_true: Ground truth count [B, 1]
#         y_pred: Predicted count [B, 1]
    
#     Returns:
#         Relative count loss value
#     """
#     # Ensure shapes match
#     y_true = K.flatten(y_true)
#     y_pred = K.flatten(y_pred)
    
#     # Calculate relative error
#     # Add 1 to denominator to prevent division by zero
#     relative_error = (y_pred - y_true) / (y_true + 1.0)
    
#     # Return mean squared relative error
#     return K.mean(K.square(relative_error))


# def combined_loss(density_weight=1.0, count_weight=0.1):
#     """
#     Combined loss function for SACNN
#     Weighted sum of density map loss and count loss
    
#     Args:
#         density_weight: Weight for density map loss
#         count_weight: Weight for count loss
    
#     Returns:
#         Combined loss function
#     """
    
#     def loss(y_true, y_pred):
#         """
#         y_true and y_pred are dictionaries with keys:
#         - 'density_map': Density map tensor
#         - 'count': Count tensor
#         """
        
#         # Extract components
#         true_density = y_true['density_map']
#         pred_density = y_pred['density_map']
#         true_count = y_true['count']
#         pred_count = y_pred['count']
        
#         # Calculate individual losses
#         density_loss = euclidean_loss(true_density, pred_density)
#         count_loss = relative_count_loss(true_count, pred_count)
        
#         # Weighted combination
#         total_loss = density_weight * density_loss + count_weight * count_loss
        
#         return total_loss
    
#     return loss


# def adaptive_loss(epoch_threshold=50):
#     """
#     Adaptive loss that adds count loss after certain epochs

    
#     Args:
#         epoch_threshold: Epoch after which to add count loss
    
#     Returns:
#         Loss function that adapts based on training progress
#     """
    
#     class AdaptiveLoss(tf.keras.losses.Loss):
#         def __init__(self, name='adaptive_loss'):
#             super().__init__(name=name)
#             self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
#             self.epoch_threshold = epoch_threshold
            
#         def call(self, y_true, y_pred):
#             # Density loss (always active)
#             density_loss = euclidean_loss(y_true['density_map'], y_pred['density_map'])
            
#             # Count loss (conditional)
#             count_loss = relative_count_loss(y_true['count'], y_pred['count'])
            
#             # Weight for count loss based on epoch
#             count_weight = tf.cond(
#                 self.current_epoch >= self.epoch_threshold,
#                 lambda: 0.1,  # After threshold
#                 lambda: 0.0   # Before threshold
#             )
            
#             return density_loss + count_weight * count_loss
        
#         def update_epoch(self, epoch):
#             """Update current epoch"""
#             self.current_epoch.assign(epoch)
    
#     return AdaptiveLoss()


# # Metrics for evaluation
# def mae_count(y_true, y_pred):
#     """
#     Mean Absolute Error for count prediction
    
#     Args:
#         y_true: Ground truth count
#         y_pred: Predicted count
    
#     Returns:
#         MAE value
#     """
#     return K.mean(K.abs(y_pred - y_true))


# def mse_count(y_true, y_pred):
#     """
#     Mean Squared Error for count prediction
    
#     Args:
#         y_true: Ground truth count
#         y_pred: Predicted count
    
#     Returns:
#         MSE value
#     """
#     return K.mean(K.square(y_pred - y_true))


# def rmse_count(y_true, y_pred):
#     """
#     Root Mean Squared Error for count prediction
    
#     Args:
#         y_true: Ground truth count
#         y_pred: Predicted count
    
#     Returns:
#         RMSE value
#     """
#     return K.sqrt(K.mean(K.square(y_pred - y_true)))


# class DensityMapLoss(tf.keras.losses.Loss):
#     """
#     Custom loss class for density map regression
#     """
    
#     def __init__(self, name='density_map_loss'):
#         super().__init__(name=name)
    
#     def call(self, y_true, y_pred):
#         """
#         Calculate L2 loss between density maps
        
#         Args:
#             y_true: Ground truth density map
#             y_pred: Predicted density map
        
#         Returns:
#             Loss value
#         """
#         return tf.reduce_mean(tf.square(y_pred - y_true))


# class CountLoss(tf.keras.losses.Loss):
#     """
#     Custom loss class for count prediction
#     """
    
#     def __init__(self, relative=True, name='count_loss'):
#         super().__init__(name=name)
#         self.relative = relative
    
#     def call(self, y_true, y_pred):
#         """
#         Calculate count loss
        
#         Args:
#             y_true: Ground truth count
#             y_pred: Predicted count
        
#         Returns:
#             Loss value
#         """
#         if self.relative:
#             # Relative count loss
#             relative_error = (y_pred - y_true) / (y_true + 1.0)
#             return tf.reduce_mean(tf.square(relative_error))
#         else:
#             # Absolute count loss (MSE)
#             return tf.reduce_mean(tf.square(y_pred - y_true))


# def get_loss_functions(use_adaptive=False):
#     """
#     Get loss functions for model compilation
    
#     Args:
#         use_adaptive: Whether to use adaptive loss
    
#     Returns:
#         Dictionary of loss functions
#     """
    
#     if use_adaptive:
#         return {
#             'density_map': DensityMapLoss(),
#             'count': CountLoss(relative=True)
#         }
#     else:
#         return {
#             'density_map': euclidean_loss,
#             'count': relative_count_loss
#         }


# def get_metrics():
#     """
#     Get metrics for model compilation
    
#     Returns:
#         Dictionary of metrics
#     """
    
#     return {
#         'density_map': ['mse'],
#         'count': [mae_count, mse_count, rmse_count]
#     }


# if __name__ == "__main__":
#     # Test losses
#     print("Testing loss functions...")
    
#     # Create dummy data
#     batch_size = 2
#     y_true_density = tf.random.normal((batch_size, 24, 24, 1))
#     y_pred_density = tf.random.normal((batch_size, 24, 24, 1))
#     y_true_count = tf.random.uniform((batch_size, 1), minval=1, maxval=100)
#     y_pred_count = tf.random.uniform((batch_size, 1), minval=1, maxval=100)
    
#     # Test individual losses
#     print("Euclidean loss:", euclidean_loss(y_true_density, y_pred_density).numpy())
#     print("Relative count loss:", relative_count_loss(y_true_count, y_pred_count).numpy())
    
#     # Test metrics
#     print("\nMetrics:")
#     print("MAE:", mae_count(y_true_count, y_pred_count).numpy())
#     print("MSE:", mse_count(y_true_count, y_pred_count).numpy())
#     print("RMSE:", rmse_count(y_true_count, y_pred_count).numpy())
    


# for kaggle
"""
Loss functions for SACNN training
"""

import tensorflow as tf


def euclidean_loss(y_true, y_pred):
    """
    Euclidean loss for density map regression
    L2 loss between predicted and ground truth density maps

    Args:
        y_true: Ground truth density map
        y_pred: Predicted density map

    Returns:
        Euclidean loss value
    """
    return tf.reduce_mean(tf.square(y_pred - y_true))


def relative_count_loss(y_true, y_pred):
    """
    Relative count loss

    L = ((pred_count - true_count) / (true_count + 1))^2

    Args:
        y_true: Ground truth count [B, 1]
        y_pred: Predicted count [B, 1]

    Returns:
        Relative count loss value
    """
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    # Add 1 to denominator to prevent division by zero
    relative_error = (y_pred - y_true) / (y_true + 1.0)

    return tf.reduce_mean(tf.square(relative_error))


def combined_loss(density_weight=1.0, count_weight=0.1):
    """
    Combined loss function for SACNN
    Weighted sum of density map loss and count loss

    Args:
        density_weight: Weight for density map loss
        count_weight: Weight for count loss

    Returns:
        Combined loss function
    """

    def loss(y_true, y_pred):
        """
        y_true and y_pred are dictionaries with keys:
        - 'density_map': Density map tensor
        - 'count': Count tensor
        """

        true_density = y_true['density_map']
        pred_density = y_pred['density_map']
        true_count = y_true['count']
        pred_count = y_pred['count']

        density_loss = euclidean_loss(true_density, pred_density)
        count_loss = relative_count_loss(true_count, pred_count)

        total_loss = density_weight * density_loss + count_weight * count_loss

        return total_loss

    return loss


def adaptive_loss(epoch_threshold=50):
    """
    Adaptive loss that adds count loss after certain epochs

    Args:
        epoch_threshold: Epoch after which to add count loss

    Returns:
        Loss function that adapts based on training progress
    """

    class AdaptiveLoss(tf.keras.losses.Loss):
        def __init__(self, name='adaptive_loss'):
            super().__init__(name=name)
            self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
            self.epoch_threshold = epoch_threshold

        def call(self, y_true, y_pred):
            density_loss = euclidean_loss(y_true['density_map'], y_pred['density_map'])
            count_loss = relative_count_loss(y_true['count'], y_pred['count'])

            count_weight = tf.cond(
                self.current_epoch >= self.epoch_threshold,
                lambda: 0.1,
                lambda: 0.0
            )

            return density_loss + count_weight * count_loss

        def update_epoch(self, epoch):
            """Update current epoch"""
            self.current_epoch.assign(epoch)

    return AdaptiveLoss()


# Metrics for evaluation
def mae_count(y_true, y_pred):
    """
    Mean Absolute Error for count prediction

    Args:
        y_true: Ground truth count
        y_pred: Predicted count

    Returns:
        MAE value
    """
    return tf.reduce_mean(tf.abs(y_pred - y_true))


def mse_count(y_true, y_pred):
    """
    Mean Squared Error for count prediction

    Args:
        y_true: Ground truth count
        y_pred: Predicted count

    Returns:
        MSE value
    """
    return tf.reduce_mean(tf.square(y_pred - y_true))


def rmse_count(y_true, y_pred):
    """
    Root Mean Squared Error for count prediction

    Args:
        y_true: Ground truth count
        y_pred: Predicted count

    Returns:
        RMSE value
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


class DensityMapLoss(tf.keras.losses.Loss):
    """
    Custom loss class for density map regression
    """

    def __init__(self, name='density_map_loss'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        """
        Calculate L2 loss between density maps

        Args:
            y_true: Ground truth density map
            y_pred: Predicted density map

        Returns:
            Loss value
        """
        return tf.reduce_mean(tf.square(y_pred - y_true))


class CountLoss(tf.keras.losses.Loss):
    """
    Custom loss class for count prediction
    """

    def __init__(self, relative=True, name='count_loss'):
        super().__init__(name=name)
        self.relative = relative

    def call(self, y_true, y_pred):
        """
        Calculate count loss

        Args:
            y_true: Ground truth count
            y_pred: Predicted count

        Returns:
            Loss value
        """
        if self.relative:
            relative_error = (y_pred - y_true) / (y_true + 1.0)
            return tf.reduce_mean(tf.square(relative_error))
        else:
            return tf.reduce_mean(tf.square(y_pred - y_true))


def get_loss_functions(use_adaptive=False):
    """
    Get loss functions for model compilation

    Args:
        use_adaptive: Whether to use adaptive loss

    Returns:
        Dictionary of loss functions
    """

    if use_adaptive:
        return {
            'density_map': DensityMapLoss(),
            'count': CountLoss(relative=True)
        }
    else:
        return {
            'density_map': euclidean_loss,
            'count': relative_count_loss
        }


def get_metrics():
    """
    Get metrics for model compilation

    Returns:
        Dictionary of metrics
    """

    return {
        'density_map': ['mse'],
        'count': [mae_count, mse_count, rmse_count]
    }


if __name__ == "__main__":
    print("Testing loss functions...")

    batch_size = 2
    y_true_density = tf.random.normal((batch_size, 24, 24, 1))
    y_pred_density = tf.random.normal((batch_size, 24, 24, 1))
    y_true_count = tf.random.uniform((batch_size, 1), minval=1, maxval=100)
    y_pred_count = tf.random.uniform((batch_size, 1), minval=1, maxval=100)

    print("Euclidean loss:", euclidean_loss(y_true_density, y_pred_density).numpy())
    print("Relative count loss:", relative_count_loss(y_true_count, y_pred_count).numpy())

    print("\nMetrics:")
    print("MAE:", mae_count(y_true_count, y_pred_count).numpy())
    print("MSE:", mse_count(y_true_count, y_pred_count).numpy())
    print("RMSE:", rmse_count(y_true_count, y_pred_count).numpy())