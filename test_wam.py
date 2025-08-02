# test_wam.py - 마스크 기반 워터마킹 버전
import torch
import sys
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 필요한 함수들 import
from notebooks.inference_utils import (
    load_model_from_checkpoint, 
    default_transform,
    create_random_mask,
    unnormalize_img,
    plot_outputs,
    msg2str
)
from watermark_anything.data.metrics import msg_predict_inference

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 이미지 로드 함수
def load_img(path):
    img = Image.open(path).convert("RGB")
    img = default_transform(img).unsqueeze(0).to(device)
    return img

# 모델 로드
exp_dir = "checkpoints"
json_path = os.path.join(exp_dir, "params.json")
ckpt_path = os.path.join(exp_dir, 'wam_mit.pth')

if os.path.exists(ckpt_path) and os.path.exists(json_path):
    print("체크포인트와 설정 파일 로드 중...")
    
    # 모델 로드
    wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()
    print("WAM 모델 로드 성공!")
    
    # 테스트 이미지 로드
    test_image_path = "assets/images/alpaca.jpg"
    if os.path.exists(test_image_path):
        # 이미지 로드
        img_pt = load_img(test_image_path)  # [1, 3, H, W]
        
        print(f"\n테스트 이미지 로드: {test_image_path}")
        print(f"텐서 크기: {img_pt.shape}")
        
        # 32비트 랜덤 메시지 생성
        wm_msg = wam.get_random_msg(1)  # [1, 32]
        print(f"원본 워터마크 메시지: {msg2str(wm_msg[0])}")
        
        # 워터마크 임베딩 (전체 이미지)
        outputs = wam.embed(img_pt, wm_msg)
        
        # 랜덤 마스크 생성 (이미지의 50%에 워터마크)
        proportion_masked = 0.5
        mask = create_random_mask(img_pt, num_masks=1, mask_percentage=proportion_masked)  # [1, 1, H, W]
        
        # 마스크를 사용해 부분적으로 워터마크 적용
        img_w = outputs['imgs_w'] * mask + img_pt * (1 - mask)  # [1, 3, H, W]
        
        # 워터마크 검출
        preds = wam.detect(img_w)["preds"]  # [1, 33, 256, 256]
        mask_preds = F.sigmoid(preds[:, 0, :, :])  # [1, 256, 256]
        bit_preds = preds[:, 1:, :, :]  # [1, 32, 256, 256]
        
        # 메시지 예측 및 정확도 계산
        pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()  # [1, 32]
        bit_acc = (pred_message == wm_msg).float().mean().item()
        
        print(f"\n예측된 워터마크 메시지: {msg2str(pred_message[0])}")
        print(f"비트 정확도: {bit_acc:.2%}")
        
        # 워터마킹된 이미지와 마스크 저장
        save_image(unnormalize_img(img_w), "watermarked_image.png")
        
        # 마스크를 원본 크기로 리사이즈
        mask_preds_res = F.interpolate(
            mask_preds.unsqueeze(1), 
            size=(img_pt.shape[-2], img_pt.shape[-1]), 
            mode="bilinear", 
            align_corners=False
        )  # [1, 1, H, W]
        
        save_image(mask_preds_res, "predicted_mask.png")
        save_image(mask, "target_mask.png")
        
        print("\n저장된 파일:")
        print("- watermarked_image.png: 워터마킹된 이미지")
        print("- predicted_mask.png: 예측된 워터마크 위치")
        print("- target_mask.png: 실제 워터마크 위치")
        
        # inference.ipynb의 plot_outputs 함수 사용
        plot_outputs(
            img_pt.detach(), 
            img_w.detach(), 
            mask.detach(), 
            mask_preds_res.detach()
        )
        
        # 추가 시각화 (한글 제목 포함)
        plt.figure(figsize=(16, 8))
        
        # 원본 vs 워터마킹
        plt.subplot(2, 3, 1)
        plt.imshow(unnormalize_img(img_pt).detach().squeeze().permute(1, 2, 0).cpu())
        plt.title('원본 이미지')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(unnormalize_img(img_w).detach().squeeze().permute(1, 2, 0).cpu())
        plt.title('워터마킹된 이미지')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        diff = torch.abs(img_w - img_pt) * mask
        diff_np = unnormalize_img(diff).detach().squeeze().permute(1, 2, 0).cpu().numpy()
        plt.imshow(np.clip(diff_np * 10, 0, 1), cmap='hot')
        plt.title('차이 이미지 (10배 확대)')
        plt.axis('off')
        
        # 마스크 비교
        plt.subplot(2, 3, 4)
        plt.imshow(mask.detach().squeeze().cpu(), cmap='gray')
        plt.title('실제 워터마크 위치')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(mask_preds_res.detach().squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        plt.title('예측된 워터마크 위치')
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        # 마스크 오버레이
        overlay = unnormalize_img(img_w).detach().squeeze().permute(1, 2, 0).cpu().numpy()
        mask_overlay = mask_preds_res.detach().squeeze().cpu().numpy()
        plt.imshow(overlay)
        plt.imshow(mask_overlay, alpha=0.5, cmap='jet')
        plt.title('워터마크 위치 오버레이')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('watermark_analysis.png', dpi=150, bbox_inches='tight')
        print("\n분석 결과 저장: watermark_analysis.png")
        plt.show()
        
    else:
        print("테스트 이미지를 찾을 수 없습니다.")
else:
    print("체크포인트 또는 설정 파일을 찾을 수 없습니다.")