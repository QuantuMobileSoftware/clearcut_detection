from catalyst.dl.core import MetricCallback

from metrics import multi_class_dice


class MultiClassDiceCallback(MetricCallback):
    """
    Dice metric callback.
    """

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "dice",
            eps: float = 1e-7,
            threshold: float = None,
            activation: str = "Sigmoid"
    ):
        """
        :param input_key: input key to use for dice calculation;
            specifies our `y_true`.
        :param output_key: output key to use for dice calculation;
            specifies our `y_pred`.
        """
        super().__init__(
            prefix=prefix,
            metric_fn=multi_class_dice,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation
        )
