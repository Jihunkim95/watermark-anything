# test_wam.py 파일 생성
import torch
from PIL import Image
from watermark_anything.inference import load_model_from_checkpoint

# GPU 사용 가능 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 로드
wam = load_model_from_checkpoint(
    "checkpoints/params.json",
    "checkpoints/wam_mit.pth"
).to(device).eval()

print("WAM 모델 로드 성공!")
