from .architecture_c import CustomLightweightTryOn
from .discriminator import PatchGANDiscriminator

def __getattr__(name):
    if name == "ControlNetTryOnModel":
        from .architecture_a import ControlNetTryOnModel
        return ControlNetTryOnModel
    elif name == "IPAdapterTryOnModel":
        from .architecture_b import IPAdapterTryOnModel
        return IPAdapterTryOnModel
    elif name == "SOTADiffusionTryOn":
        from .architecture_d import SOTADiffusionTryOn
        return SOTADiffusionTryOn
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["ControlNetTryOnModel", "IPAdapterTryOnModel", "CustomLightweightTryOn", "SOTADiffusionTryOn", "PatchGANDiscriminator"]
