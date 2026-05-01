from functools import partial
import torch.nn as nn
import timm.models as tmodels

import timm.models.resnet as R
from timm.models import register_model
from timm.models.resnet import ResNet as TimmResNet

@register_model
def ecaresnet101t(pretrained: bool = False, **kwargs) -> R.ResNet:
    """Constructs an ECA-ResNet-101-T model.
    Like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels in the deep stem and ECA attn.
    """
    model_args = dict(
        block=R.Bottleneck, layers=[3, 4, 23, 3], stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'))
    return R._create_resnet('ecaresnet101t', pretrained, **dict(model_args, **kwargs))

@register_model
def ecaresnetblur101d(pretrained: bool = False, **kwargs) -> R.ResNet:
    """Constructs a ECA-ResNet-101-D model with blur anti-aliasing
    """
    model_args = dict(
        block=R.Bottleneck, layers=[3, 4, 23, 3], 
        aa_layer=R.BlurPool2d,
        stem_width=32, 
        stem_type='deep', avg_down=True,
        # drop_rate = 0.2,
        # drop_path_rate = 0.1,
        block_args=dict(attn_layer='eca'))
    return R._create_resnet('ecaresnetblur101d', pretrained, **dict(model_args, **kwargs))

@register_model
def ecaresnetblur152d(pretrained: bool = False, **kwargs) -> R.ResNet:
    """Constructs a ResNet-152-D model with eca.
    """
    model_args = dict(
        block=R.Bottleneck, layers=[3, 8, 36, 3], 
        aa_layer=R.BlurPool2d,
        stem_width=32, 
        stem_type='deep', 
        avg_down=True,
        block_args=dict(attn_layer='eca'))
    return R._create_resnet('ecaresnetblur152d', pretrained, **dict(model_args, **kwargs))

@register_model
def ecaresnetblur200d(pretrained: bool = False, **kwargs) -> R.ResNet:
    """Constructs a ResNet-152-D model with eca.
    """
    model_args = dict(
        block=R.Bottleneck, layers=[3, 24, 36, 3], 
        aa_layer=R.BlurPool2d,
        stem_width=32, 
        stem_type='deep', 
        avg_down=True,
        block_args=dict(attn_layer='eca'))
    return R._create_resnet('ecaresnetblur152d', pretrained, **dict(model_args, **kwargs))

@register_model
def ecaresnetaa101d(pretrained: bool = False, **kwargs) -> R.ResNet:
    """Constructs a ECA-ResNet-101-D model w/ avgpool anti-aliasing
    """
    model_args = dict(
        block=R.Bottleneck, layers=[3, 4, 23, 3], 
        aa_layer=nn.AvgPool2d,
        stem_width=32, 
        stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return R._create_resnet('ecaresnetaa101d', pretrained, **dict(model_args, **kwargs))

@register_model
def ecaresnet152d(pretrained: bool = False, **kwargs) -> R.ResNet:
    """Constructs a ResNet-152-D model with eca.
    """
    model_args = dict(
        block=R.Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return R._create_resnet('ecaresnet152d', pretrained, **dict(model_args, **kwargs))


def timm_create_model(in_channels, num_classes,
                variant="resnet50d",
                **kwargs,
                ):
  part_create = partial(tmodels.create_model, variant,
                            in_chans=in_channels,
                            num_classes=num_classes,)
  if variant.find("swin") != -1 or variant.find("vit") != -1:
    assert kwargs.get("top_k") is not None, "Must contain top_k params"
    assert kwargs.get("max_len") is not None, "Must contain max_len params"

    img_size = (kwargs["max_len"], kwargs["top_k"])
    
    return part_create(img_size=img_size)
  else:
    return part_create()