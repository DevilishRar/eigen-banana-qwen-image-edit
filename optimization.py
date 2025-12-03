"""
"""

from typing import Any
from typing import Callable
from typing import ParamSpec
from torchao.quantization import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

# Try to import spaces (HuggingFace Spaces specific)
try:
    import spaces
except ImportError:
    # Create dummy implementations for non-HF environments (like Kaggle)
    from contextlib import contextmanager
    
    class spaces:
        @staticmethod
        def GPU(duration=None):
            def decorator(func):
                return func
            return decorator
        
        @staticmethod
        @contextmanager
        def aoti_capture(module):
            class DummyCall:
                def __init__(self):
                    self.args = ()
                    self.kwargs = {}
            yield DummyCall()
        
        @staticmethod
        def aoti_compile(exported, configs):
            return None
        
        @staticmethod
        def aoti_apply(compiled, module):
            pass

import torch
from torch.utils._pytree import tree_map


P = ParamSpec('P')


TRANSFORMER_IMAGE_SEQ_LENGTH_DIM = torch.export.Dim('image_seq_length')
TRANSFORMER_TEXT_SEQ_LENGTH_DIM = torch.export.Dim('text_seq_length')

TRANSFORMER_DYNAMIC_SHAPES = {
    'hidden_states': {
        1: TRANSFORMER_IMAGE_SEQ_LENGTH_DIM,
    },
    'encoder_hidden_states': {
        1: TRANSFORMER_TEXT_SEQ_LENGTH_DIM,
    },
    'encoder_hidden_states_mask': {
        1: TRANSFORMER_TEXT_SEQ_LENGTH_DIM,
    },
    'image_rotary_emb': ({
        0: TRANSFORMER_IMAGE_SEQ_LENGTH_DIM,
    }, {
        0: TRANSFORMER_TEXT_SEQ_LENGTH_DIM,
    }),
}


INDUCTOR_CONFIGS = {
    'conv_1x1_as_mm': True,
    'epilogue_fusion': False,
    'coordinate_descent_tuning': True,
    'coordinate_descent_check_all_directions': True,
    'max_autotune': True,
    'triton.cudagraphs': True,
}


def optimize_pipeline_(pipeline: Callable[P, Any], *args: P.args, **kwargs: P.kwargs):

    @spaces.GPU(duration=1500)
    def compile_transformer():

        with spaces.aoti_capture(pipeline.transformer) as call:
            pipeline(*args, **kwargs)

        dynamic_shapes = tree_map(lambda t: None, call.kwargs)
        dynamic_shapes |= TRANSFORMER_DYNAMIC_SHAPES

        # quantize_(pipeline.transformer, Float8DynamicActivationFloat8WeightConfig())
        
        exported = torch.export.export(
            mod=pipeline.transformer,
            args=call.args,
            kwargs=call.kwargs,
            dynamic_shapes=dynamic_shapes,
        )

        return spaces.aoti_compile(exported, INDUCTOR_CONFIGS)

    spaces.aoti_apply(compile_transformer(), pipeline.transformer)
