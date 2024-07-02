"""
backend.py
"""


class Backend:
    """backend base"""

    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.numpy_input = None
        self.config = None

    def version(self):
        """version"""
        raise NotImplementdError("Backend:version")

    def name(self):
        """name"""
        raise NotImplementdError("Backend:name")

    def load(self, config_arg, inputs=None, outputs=None):
        """load func"""
        raise NotImplementdError("Backend:load")

    def predict(self, feed):
        """predict func"""
        raise NotImplementdError("Backend:predict")

    def warmup(self):
        """warmup func"""
        raise NotImplementdError("Backend:warmup")

    def get_performance_metrics(self):
        """get_performance_metrics func"""
        pass
