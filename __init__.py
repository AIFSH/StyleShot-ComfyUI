import os
now_dir = os.path.dirname(os.path.abspath(__file__))
import torch
import numpy as np
from PIL import Image
import cuda_malloc

from huggingface_hub import snapshot_download
from diffusers import StableDiffusionPipeline,UNet2DConditionModel, ControlNetModel,StableDiffusionAdapterPipeline, T2IAdapter

import cv2
from .styleshot.annotator.hed import SOFT_HEDdetector
from .styleshot.annotator.lineart import LineartDetector
from .styleshot.ip_adapter.ip_adapter import StyleShot,StyleContentStableDiffusionControlNetPipeline,StableDiffusionControlNetPipeline

prtrained_dir = os.path.join(now_dir,"prtrained_models")
device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"



def load_weights(use_case="text_driven",preprocessor = "Contour"):
    base_model_path = "runwayml/stable-diffusion-v1-5"
    transformer_block_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    
    if preprocessor == "Lineart":
        # detector = LineartDetector()
        styleshot_model_path = "Gaojunyao/StyleShot_lineart"
    elif preprocessor == "Contour":
        # detector = SOFT_HEDdetector()
        styleshot_model_path = "Gaojunyao/StyleShot"
    else:
        raise ValueError("Invalid preprocessor")

    if not os.path.isdir(base_model_path):
        base_model_path = snapshot_download(base_model_path,
                                            allow_patterns=["*.fp16.safetensors","*.json"],
                                            local_dir=os.path.join(prtrained_dir,base_model_path.split("/")[-1]))
        print(f"Downloaded model to {base_model_path}")
    if not os.path.isdir(transformer_block_path):
        transformer_block_path = snapshot_download(transformer_block_path, 
                                                   # allow_patterns=["*.safetensors","*.json"],
                                                   ignore_patterns=["open_clip*","*.bin"],
                                                   local_dir=os.path.join(prtrained_dir,transformer_block_path.split("/")[-1]))
        print(f"Downloaded model to {transformer_block_path}")
    if not os.path.isdir(styleshot_model_path):
        styleshot_model_path = snapshot_download(styleshot_model_path, 
                                                 local_dir=os.path.join(prtrained_dir,styleshot_model_path.split("/")[-1]))
        print(f"Downloaded model to {styleshot_model_path}")

    ip_ckpt = os.path.join(styleshot_model_path, "pretrained_weight/ip.bin")
    style_aware_encoder_path = os.path.join(styleshot_model_path, "pretrained_weight/style_aware_encoder.bin")

    if use_case == "text_driven":
        pipe = StableDiffusionPipeline.from_pretrained(base_model_path,use_safetensors=True,variant="fp16")
        styleshot = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path)
    if use_case == "image_driven":
        unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet",variant="fp16")
        content_fusion_encoder = ControlNetModel.from_unet(unet)
        
        pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(base_model_path,use_safetensors=True,controlnet=content_fusion_encoder)
        styleshot = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path)

    if use_case == "t2i-adapter":
        adapter_model_path = "TencentARC/t2iadapter_depth_sd15v2"
        if not os.path.isdir(adapter_model_path):
            adapter_model_path = snapshot_download(adapter_model_path, 
                                                   ignore_patterns=["*.png"],
                                                   local_dir=os.path.join(prtrained_dir,adapter_model_path.split("/")[-1]))
            print(f"Downloaded model to {adapter_model_path}")
        adapter = T2IAdapter.from_pretrained(adapter_model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionAdapterPipeline.from_pretrained(base_model_path, adapter=adapter,use_safetensors=True,variant="fp16")
    
        styleshot = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path)
    
    if use_case == "controlnet":
        controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
        if not os.path.isdir(controlnet_model_path):
            controlnet_model_path = snapshot_download(controlnet_model_path, 
                                                      allow_patterns=["*.json","*.fp16.safetensors"],
                                                      local_dir=os.path.join(prtrained_dir,controlnet_model_path.split("/")[-1]))
            print(f"Downloaded model to {controlnet_model_path}")
        controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet,use_safetensors=True,variant="fp16")
    
        styleshot = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path)
    
    return styleshot

class TextNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True})}}
    RETURN_TYPES = ("TEXT",)
    FUNCTION = "encode"

    CATEGORY = "AIFSH_StyleShot"

    def encode(self, text):
        return (text, )
    
class StyleShotNode:
    def __init__(self) -> None:
        self.use_case = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "style":("IMAGE",),
                "prompt":("TEXT",),
                "use_case":(["text_driven","image_driven","controlnet","t2i-adapter"],)
            },
            "optional":{
                "content":("IMAGE",),
                "preprocessor":(["Contour", "Lineart"],),
                "condition":("IMAGE",)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_StyleShot"

    def comfyimage2Image(self,comfyimage):
        comfyimage = comfyimage.numpy()[0] * 255
        image_np = comfyimage.astype(np.uint8)
        image = Image.fromarray(image_np)
        return image

    def generate(self,style,prompt,use_case,
                 content=None,preprocessor="Contour",condition=None):
        if self.use_case != use_case:
            self.use_case = use_case
            self.styleshot = load_weights(self.use_case,preprocessor)
        style_image = self.comfyimage2Image(style)
        if self.use_case == "image_driven":
            annotator_ckpts_path = os.path.join(prtrained_dir,"Annotators")
            snapshot_download("lllyasviel/Annotators",
                      allow_patterns=["sk_model*","ControlNetHED*"],
                      local_dir=annotator_ckpts_path)
        
            if preprocessor == "Lineart":
                detector = LineartDetector(annotator_ckpts_path)
            else:
                detector = SOFT_HEDdetector(annotator_ckpts_path)
            
            content_image = self.comfyimage2Image(content)
            content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
            content_image = detector(content_image)
            content_image = Image.fromarray(content_image)
    
        else:
            content_image = None
        
        if use_case in ["t2i-adapter","controlnet"]:
            condition_image = self.comfyimage2Image(condition)
            generation = self.styleshot.generate(style_image=style_image, prompt=[[prompt]], image=[condition_image])
        else:
            generation = self.styleshot.generate(style_image=style_image, prompt=[[prompt]],content_image=content_image)
        out_image = torch.from_numpy(np.array(generation[0][0]) / 255.0).unsqueeze(0)
        return (out_image,)


NODE_CLASS_MAPPINGS = {
    "TextNode":TextNode,
    "StyleShotNode": StyleShotNode
}

