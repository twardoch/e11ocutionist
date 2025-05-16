"""e11ocutionist: A text processing pipeline for enhancing speech synthesis markup.

Created by Adam Twardoch
"""

from .__version__ import __version__

# Import main classes and types for convenience
try:
    from .e11ocutionist import E11ocutionistPipeline, PipelineConfig, ProcessingStep
    from .chunker import process_document as chunker_process
    from .entitizer import process_document as entitizer_process
    from .orator import process_document as orator_process
    from .tonedown import process_document as tonedown_process
    from .elevenlabs_converter import process_document as elevenlabs_converter_process
    from .elevenlabs_synthesizer import synthesize_with_all_voices
    from .neifix import transform_nei_content

    __all__ = [
        "E11ocutionistPipeline",
        "PipelineConfig",
        "ProcessingStep",
        "__version__",
        "chunker_process",
        "elevenlabs_converter_process",
        "entitizer_process",
        "orator_process",
        "synthesize_with_all_voices",
        "tonedown_process",
        "transform_nei_content",
    ]
except ImportError:
    # Handle the case where the modules are not yet available
    __all__ = ["__version__"]
