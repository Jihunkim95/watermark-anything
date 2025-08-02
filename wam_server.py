# wam_server.py
from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
from PIL import Image
import io
import base64
from watermark_anything.inference import load_model_from_checkpoint, default_transform

app = Flask(__name__)

# 모델 로드 (서버 시작 시 한 번만)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wam = load_model_from_checkpoint(
    "checkpoints/params.json",
    "checkpoints/wam_mit.pth"
).to(device).eval()

@app.route('/watermark', methods=['POST'])
def watermark_image():
    try:
        # Unity에서 받은 이미지 데이터
        data = request.json
        image_base64 = data['image']
        
        # base64 디코딩
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # 워터마킹 처리
        img_pt = default_transform(image).unsqueeze(0).to(device)
        wm_msg = torch.randint(0, 2, (32,)).float().to(device)
        
        outputs = wam.embed(img_pt, wm_msg)
        watermarked_img = outputs['imgs_w']
        
        # 결과 이미지를 base64로 인코딩
        # ... (워터마킹된 이미지 반환 로직)
        
        return jsonify({
            'success': True,
            'watermarked_image': watermarked_base64,
            'message': wm_msg.tolist()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)