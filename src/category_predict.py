import numpy as np
import pandas as pd
import pickle
import tensorflow
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing import image

def pickle_dump(obj, path: str) -> None: # obj: Any
    """objをpickleファイルとして保存"""
    with open(path, mode="wb") as f:
        pickle.dump(obj,f)

def pickle_load(path: str): # return Any
    """pickleファイルを変数objとして読み込み"""
    with open(path, mode="rb") as f:
        data = pickle.load(f)
        return data

def img_processing(img_data) -> np.ndarray:
    """受け取った画像データの加工(学習時のデータサイズなどに変更)"""
    img = Image.open(img_data).convert(mode="RGB") # カラーの画像を取得
    image_size: int = 224 # 画像サイズ
    img = img.resize((image_size, image_size)) # 指定した画像サイズに変更
    x = image.img_to_array(img) # 画像データからarrayに変換
    x = np.expand_dims(x, axis=0)
    x = x / 255.0 # 色を0~1に変換
    return x

def model_predict(img_data: np.ndarray, model_name: str = "../model/food_recognition_ep80.h5", K:int = 5) -> str:
    """加工したデータをもとに予測カテゴリを返す関数"""
    model: tensorflow.python.keras.engine.sequential.Sequential = load_model(model_name) # モデルのロード
    pred: np.ndarray = model.predict(img_data)[0] # カテゴリの予測
    # 予測カテゴリのうち上位K個を取得
    unsorted_max_indices: np.ndarray = np.argpartition(-pred, K)[:K]
    y: np.ndarray = pred[unsorted_max_indices]
    indices: np.ndarray = np.argsort(-y)
    topK_indices: np.ndarray = unsorted_max_indices[indices]
    # 予測カテゴリのidからカテゴリ名を取得
    result: list = []
    for index in topK_indices:
        ctg_name: str = get_ctgname_by_index(index)
        result.append([ctg_name, str(pred[index])])
        # print(ctg_name + ":" + str(pred[index]))
    return result[0][0]


def get_ctgname_by_index(index: int) -> str:
    """予測したindexからそのカテゴリ名を返す関数"""
    df_category: pd.DataFrame = pd.read_pickle("../data/dataset/category/category_df.pkl") # 元のデータラベルデータフレームを取得
    ctg_dic: dict = pickle_load("../data/dataset/category/train_generator_class_indices.pkl")
    key: list[int] = [k for k, v in ctg_dic.items() if v == index] # 訓練時のラベルと予測ラベルが同じ時、元データのラベルを返す
    id: int = int(key[0]) # 元のデータセットのラベル
    ctg_name_index: int = 1 # カテゴリ名の列を取得するためのカラムインデックス
    ctg_name: str = df_category.iloc[int(id)-1, ctg_name_index] # 引数indexに対するカテゴリ名
    return ctg_name
