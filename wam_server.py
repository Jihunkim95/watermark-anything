# wam_server.py - 세션별 폴더 구조 지원
import os
import io
import base64
import json
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
from datetime import datetime

# WAM 경로 추가
import sys
sys.path.insert(0, r'C:\Research\watermark-system\watermark-anything')

from notebooks.inference_utils import (
    load_model_from_checkpoint,
    default_transform,
    create_random_mask,
    unnormalize_img,
    msg2str
)
from watermark_anything.data.metrics import msg_predict_inference

app = Flask(__name__)
CORS(app)

# 전역 변수
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# WAM 모델 로드
exp_dir = r"C:\Research\watermark-system\watermark-anything\checkpoints"
json_path = os.path.join(exp_dir, "params.json")
ckpt_path = os.path.join(exp_dir, 'wam_mit.pth')

print("Loading WAM model...")
wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()
print("WAM model loaded successfully!")

# Unity 프로젝트 경로
UNITY_PROJECT_PATH = r"C:\Research\watermark-system\VR_WAM_Watermark\Assets\ProtectedArtworks"

# 홈페이지 라우트 추가
@app.route('/', methods=['GET'])
def home():
    """홈페이지"""
    return '''
    <h1>WAM (Watermark Anything) Server for Unity VR</h1>
    <p>Status: Running</p>
    <p>Endpoints:</p>
    <ul>
        <li>GET /health - 서버 상태 확인</li>
        <li>POST /watermark - 워터마크 적용</li>
        <li>POST /verify - 워터마크 검증</li>
    </ul>
    '''

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'model_loaded': wam is not None,
        'base_path': UNITY_PROJECT_PATH,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/watermark', methods=['POST'])
def watermark_image():
    """Unity에서 받은 이미지에 워터마크 삽입"""
    try:
        data = request.json
        
        # Base64 디코딩
        image_base64 = data['image']
        image_bytes = base64.b64decode(image_base64)
        
        # 메타데이터 추출
        creator_id = data.get('creatorId', 'unknown')
        timestamp = data.get('timestamp', datetime.now().isoformat())
        artwork_id = data.get('artworkId', 'unknown')
        session_id = data.get('sessionId', 'default_session')
        version_number = data.get('versionNumber', 1)
        view_direction = data.get('viewDirection', 'MainView')
        complexity = data.get('complexity', 0.5)
        
        # PIL 이미지로 변환
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = img.size
        
        print(f"이미지 수신: {original_size}, 세션: {session_id}, 버전: v{version_number:03d}")
        
        # 텐서로 변환
        img_pt = default_transform(img).unsqueeze(0).to(device)
        
        # 워터마크 메시지 생성 (creator_id + session_id 기반 시드)
        seed_string = f"{creator_id}_{session_id}"
        torch.manual_seed(hash(seed_string) % 2**32)
        wm_msg = wam.get_random_msg(1)
        
        # 워터마크 임베딩
        with torch.no_grad():
            outputs = wam.embed(img_pt, wm_msg)
            
            # 복잡도에 따른 마스크 크기 조정 (0.3 ~ 0.7)
            mask_percentage = 0.3 + (complexity * 0.4)
            mask = create_random_mask(img_pt, num_masks=1, mask_percentage=mask_percentage)
            
            # 부분 워터마킹
            img_w = outputs['imgs_w'] * mask + img_pt * (1 - mask)
            
            # 워터마크 검출 (검증용)
            preds = wam.detect(img_w)["preds"]
            mask_preds = F.sigmoid(preds[:, 0, :, :])
            bit_preds = preds[:, 1:, :, :]
            
            # 메시지 예측
            pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()
            bit_acc = (pred_message == wm_msg.cpu()).float().mean().item()
        
        # 워터마킹된 이미지를 PIL로 변환
        img_w_np = unnormalize_img(img_w).detach().squeeze().permute(1, 2, 0).cpu().numpy()
        img_w_pil = Image.fromarray((img_w_np * 255).astype(np.uint8))
        
        # 원본 크기로 리사이즈
        img_w_pil = img_w_pil.resize(original_size, Image.LANCZOS)
        
        # 마스크를 원본 크기로 리사이즈
        mask_resized = F.interpolate(
            mask, 
            size=(original_size[1], original_size[0]), 
            mode="nearest"
        )
        mask_np = mask_resized.detach().squeeze().cpu().numpy()
        
        # 예측된 마스크도 원본 크기로 리사이즈
        mask_preds_resized = F.interpolate(
            mask_preds.unsqueeze(1),
            size=(original_size[1], original_size[0]),
            mode="bilinear",
            align_corners=False
        )
        mask_preds_np = mask_preds_resized.detach().squeeze().cpu().numpy()
        
        # 세션별 폴더 생성
        session_folder = os.path.join(UNITY_PROJECT_PATH, session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # 파일명 생성
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 워터마킹된 이미지 저장
        filename = f"{artwork_id}_v{version_number:03d}_{view_direction}_watermarked_{timestamp_str}.png"
        filepath = os.path.join(session_folder, filename)
        img_w_pil.save(filepath)
        print(f"워터마킹된 이미지 저장: {filepath}")
        
        # 2. 원본 이미지 저장
        original_filename = f"{artwork_id}_v{version_number:03d}_{view_direction}_original_{timestamp_str}.png"
        original_filepath = os.path.join(session_folder, original_filename)
        img.save(original_filepath)
        
        # 3. 실제 적용된 마스크 이미지 저장 (Ground Truth)
        mask_filename = f"{artwork_id}_v{version_number:03d}_{view_direction}_mask_applied_{timestamp_str}.png"
        mask_filepath = os.path.join(session_folder, mask_filename)
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_img.save(mask_filepath)
        
        # 4. 예측된 마스크 이미지 저장 (Detection Result)
        mask_pred_filename = f"{artwork_id}_v{version_number:03d}_{view_direction}_mask_detected_{timestamp_str}.png"
        mask_pred_filepath = os.path.join(session_folder, mask_pred_filename)
        mask_pred_img = Image.fromarray((mask_preds_np * 255).astype(np.uint8))
        mask_pred_img.save(mask_pred_filepath)
        
        # 5. 마스크 오버레이 이미지 생성 및 저장
        overlay_filename = f"{artwork_id}_v{version_number:03d}_{view_direction}_overlay_{timestamp_str}.png"
        overlay_filepath = os.path.join(session_folder, overlay_filename)
        
        # 원본 이미지에 마스크 오버레이
        overlay_img = img.copy()
        overlay_np = np.array(overlay_img)
        
        # 마스크 영역을 빨간색으로 표시
        mask_colored = np.zeros_like(overlay_np)
        mask_colored[:, :, 0] = mask_np * 255  # Red channel
        
        # 알파 블렌딩
        alpha = 0.3
        overlay_np = overlay_np.astype(float)
        mask_area = mask_np > 0.5
        overlay_np[mask_area] = overlay_np[mask_area] * (1 - alpha) + mask_colored[mask_area] * alpha
        
        overlay_pil = Image.fromarray(overlay_np.astype(np.uint8))
        overlay_pil.save(overlay_filepath)
        
        print(f"마스크 이미지 저장:")
        print(f"  - 적용 마스크: {mask_filename}")
        print(f"  - 검출 마스크: {mask_pred_filename}")
        print(f"  - 오버레이: {overlay_filename}")
        
        # 메타데이터 저장
        metadata = {
            'creator_id': creator_id,
            'session_id': session_id,
            'artwork_id': artwork_id,
            'version_number': version_number,
            'view_direction': view_direction,
            'timestamp': timestamp,
            'complexity': complexity,
            'watermark_message': msg2str(wm_msg[0]),
            'bit_accuracy': bit_acc,
            'mask_percentage': mask_percentage,
            'original_size': list(original_size),
            'watermarked_file': filename,
            'original_file': original_filename,
            'mask_file': mask_filename,
            'mask_detected_file': mask_pred_filename,
            'overlay_file': overlay_filename
        }
        
        metadata_filename = f"{artwork_id}_v{version_number:03d}_metadata_{timestamp_str}.json"
        metadata_filepath = os.path.join(session_folder, metadata_filename)
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Base64로 인코딩하여 반환
        buffered = io.BytesIO()
        img_w_pil.save(buffered, format="PNG")
        img_w_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # 마스크 이미지도 Base64로 인코딩 (Unity에서 표시용)
        mask_buffered = io.BytesIO()
        overlay_pil.save(mask_buffered, format="PNG")
        mask_overlay_base64 = base64.b64encode(mask_buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'watermarked_image': img_w_base64,
            'mask_overlay_image': mask_overlay_base64,
            'filename': filename,
            'filepath': filepath,
            'session_folder': session_folder,
            'message': msg2str(wm_msg[0]),
            'bit_accuracy': bit_acc,
            'metadata': metadata,
            'files_saved': {
                'watermarked': filename,
                'original': original_filename,
                'mask_applied': mask_filename,
                'mask_detected': mask_pred_filename,
                'overlay': overlay_filename,
                'metadata': metadata_filename
            }
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
@app.route('/verify', methods=['POST'])
def verify_watermark():
    """워터마크 검증"""
    try:
        data = request.json
        image_base64 = data['image']
        image_bytes = base64.b64decode(image_base64)
        
        # PIL 이미지로 변환
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_pt = default_transform(img).unsqueeze(0).to(device)
        
        # 워터마크 검출
        with torch.no_grad():
            preds = wam.detect(img_pt)["preds"]
            mask_preds = F.sigmoid(preds[:, 0, :, :])
            bit_preds = preds[:, 1:, :, :]
            
            # 메시지 예측
            pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()
            
            # 마스크를 원본 크기로 리사이즈
            mask_resized = F.interpolate(
                mask_preds.unsqueeze(1),
                size=(img.size[1], img.size[0]),
                mode="bilinear",
                align_corners=False
            )
            
            # 마스크 이미지 생성
            mask_np = mask_resized.detach().squeeze().cpu().numpy()
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
            
            # Base64 인코딩
            mask_buffered = io.BytesIO()
            mask_img.save(mask_buffered, format="PNG")
            mask_base64 = base64.b64encode(mask_buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'detected': mask_preds.max().item() > 0.5,
            'message': msg2str(pred_message[0]),
            'confidence': mask_preds.max().item(),
            'mask_image': mask_base64
        })
        
    except Exception as e:
        print(f"Verify error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print(f"\n=== WAM Server for Unity VR ===")
    print(f"Unity screenshot path: {UNITY_PROJECT_PATH}")
    print(f"Device: {device}")
    print(f"Server starting on http://localhost:5000")
    print(f"\nEndpoints:")
    print(f"  GET  /         - 홈페이지")
    print(f"  GET  /health   - 서버 상태 확인")
    print(f"  POST /watermark - 워터마크 적용")
    print(f"  POST /verify    - 워터마크 검증")
    print(f"\nPress CTRL+C to quit\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)