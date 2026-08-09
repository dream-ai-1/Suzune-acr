# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Suzune Neural Type Definitions for Type Safety & Tensor Validation
"""
class NeuralType:
    def __init__(self, axes=None, element_type=None, optional=False):
        self.axes = axes
        self.element_type = element_type
        self.optional = optional

class AudioSignal(NeuralType):
    pass

class SpectrogramType(NeuralType):
    pass

class LogprobsType(NeuralType):
    pass

class LabelsType(NeuralType):
    pass

class LengthsType(NeuralType):
    pass
