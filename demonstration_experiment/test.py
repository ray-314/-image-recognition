import cv2
import datetime
import numpy as np
from flask import Flask, render_template, request
import category_predict
import os

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def hello_world():
    img_dir = "static/imgs/"
    if request.method == 'GET': 
        img_path=None
        result=None
    # if request.method == 'GET': result=None
    elif request.method == 'POST':
        #### POSTにより受け取った画像を読み込む
        food_image = request.files["img"]
        print(food_image)
        stream = request.files['img'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)
        #### 現在時刻を名前として「imgs/」に保存する
        dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        img_path = img_dir + dt_now + ".jpg"
        cv2.imwrite(img_path, img)
    #### 保存した画像ファイルのpathをHTMLに渡す
        food_image = request.files["img"].stream
        img_data = category_predict.img_processing(food_image)
        result = category_predict.model_predict(img_data, model_name="model/food_recognition_ep80.h5", K=5) # pathの位置が違うみたい
        # result = os.getcwd() # 今いるところを把握
    # return result
    return render_template('index.html', img_path=img_path, result=result) 

if __name__ == "__main__":
    app.run(debug=True)