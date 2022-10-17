from flask import Flask
from flask import request, jsonify
import json
import category_predict

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/image", methods=["POST"])
def classify_image():
    food_image = request.files["food_image"].stream
    img_data = category_predict.img_processing(food_image)
    result = category_predict.model_predict(img_data, model_name="../model/food_recognition_ep80.h5", K=1)
    return result


@app.route("/post_food", methods=["POST"])
def post_food():
    json_file = open("../foods.json", "r") # ../foods.jsonは書き込むファイルのPASS
    # 辞書型で読み込む
    json_dict = json.load(json_file)
    # 追加する場所
    food_value = json_dict['foods']

    # 画像ファイルをディレクトリに保存
    image_path = request.files['uploadFile']
    fileName = image_path.filename
    file.save(os.path.join(images, fileName))  # images：保存するディレクトリ名

    # 記入のリクエスト
    comment = request.form["comment"]
    image_path = "/images/" + fileName
    food_category = request.form["food_category"]
    price = int(request.form["price"])
    store_name = request.form["store_name"]

    # 追加するもの
    to_add = {
        'comment': comment,
        'food_category': food_category,
        'image_path': image_path,
        'price': price,
        'store_name': store_name
        }

    # 追加する
    food_value.append(to_add)

    # 書き込んで保存
    with open("../foods.json","w") as f: # ../foods.jsonは書き込むファイルのPASS
        json.dump(json_dict,f,indent=4)

    return "success"


@app.route("/foods", methods=["GET"])

def get_food():
    # ここではjsonファイルを読み込むということを行う
    json_open = open('../foods.json', 'r')
    json_load = json.load(json_open)

    return json_load


if __name__ == "__main__":
    app.run()
