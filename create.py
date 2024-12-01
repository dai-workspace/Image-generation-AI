from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import torch

# モデル設定
model_id = "aipicasso/emi-2"

# スケジューラーの設定
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    model_id, subfolder="scheduler"
)

# パイプラインの初期化
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, scheduler=scheduler, torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# プロンプト
# prompt = "1guy, upper body, black perm hair, black eyes, looking at viewer, open your mouth and laugh"
prompt = "Soba with a face, upper body, looking at viewer"

# 画像生成ループ
for idx in range(1):
    # Generatorを毎回新規作成し、シードをランダムに設定
    seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # 画像生成
    images = pipe(
        prompt, num_inference_steps=50, guidance_scale=10, generator=generator
    ).images
    images[0].save(f"test_{idx}.png")
