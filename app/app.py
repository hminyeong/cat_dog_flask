from flask import Flask, render_template, request, jsonify
import torch
import json
import torchvision
from torchvision import transforms
from PIL import Image

# 'Flask' 클래스의 인스턴스 생성
# 이 인스턴스는 웹 애플리케이션의 기능을 제공하며, 라우팅 및 요청 어리와 같은 기능이 포함됨 
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model load
model = torch.load("../cat_dog_model.pt", map_location=device)

# 모델을 평가 모드로 설정
model.eval()

# json 파일은 클래스 인덱스와 그에 해당하는 라벨을 매핑하는데 사용됨
with open('../class_index.json', 'r') as f:
    class_idx = json.load(f)

# 이미지에 적용할 변환을 순서대로 적용할 수 있는 파이프라인 생성
transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((224,224)), # 이미지 크기
                    torchvision.transforms.ToTensor(),        # 텐서 변환 
                ])

# 이미지 파일의 경로를 입력받아 해당 이미지의 클래스를 예측
def predict_image_class(image_path):
    image = Image.open(image_path)
    with torch.no_grad():
        image = transform(image).unsqueeze(0)
        image = image.to(device)
        pred = model(image)
        pred = pred.cpu().numpy()
        if pred[0][0] >= 0.5:
            predicted_class_idx = 1
        else:
            predicted_class_idx = 0
        confidence = pred[0][0]
        return predicted_class_idx, confidence # 예측 클래스, 신뢰도 반환

# / 경로에 접속하면, index.html 템플릿을 렌더링하여 보여줌 
@app.route('/')
def index():
    return render_template('index.html')

# /predict 경로에 POST 요청을 받으면, 요청으로부터 이미지 파일을 받아서 저장한 후에 predict_image_class() 함수를 사용하여 예측을 수행
# 예측 클래스 결과와 신뢰도를 JSON 형식으로 반환 
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file'] # 파일 불러오기 
        file_path = f"C:/Users/alsdu/Desktop/Git_Repo/cat_dog_flask/app/static/images/{file.filename}"
        file.save(file_path)
        print("file save OK")
        predicted_class_idx, confidence = predict_image_class(file_path)
        if predicted_class_idx == 1:
            result = "dog"
            confidence = confidence*100
        else:
            result = "cat"
            confidence = (1 - confidence)*100
        print(predicted_class_idx, confidence)
        return jsonify({'result': result, 'confidence': str(confidence)})

# 스크립트가 직접 실행되는 경우에만 Flask 애플리케이션을 실행
# flask 외부 접속 허용, 방화벽 설정(인바운드 규칙 5000 추가)
# debug=True로 설정하면 코드 변경이 있을 때마다 서버가 자동으로 재시작
if __name__ == '__main__': 
    app.run(host='192.168.0.129', port = 5000, debug=True)


