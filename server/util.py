import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d


__model = None
__class_name_to_number=None
__class_Number_to_name=None



def get_cv2_img_from_base64(b64str):
    encoded_data= b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data),np.uint8)
    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    return img

def map_classnum_to_name (class_num):
    return __class_Number_to_name[class_num]



def classify_image (_base64_img , image_path= None):
    imgs = get_cropped_image_if_2_eyes(image_path,_base64_img)
    results=[]
    if imgs is not None :
        print("img recieved")
        #for img in imgs:
        resized_sample = cv2.resize(imgs, (32, 32))
        transformed_img = w2d(imgs,"db1",5)
        transformed_img_rescaled = cv2.resize(transformed_img, (32, 32))
        combined_img = np.vstack((resized_sample.reshape(32 * 32 * 3, 1), transformed_img_rescaled.reshape(32 * 32, 1)))
        len_img = 32*32*3+32*32
        final = combined_img.reshape(1,len_img).astype(float)
        results.append({
                       'class' :map_classnum_to_name(__model.predict(final)[0]),
                        'class_prob' : np.round(__model.predict_proba(final)*100,2).tolist(),
                        'class_name' : __class_name_to_number
        })
    return results



def load_artifacts():
    print("Loading model artifacts")
    global __class_name_to_number
    global __class_Number_to_name
    with open (r"C:\Users\Mohamed\Downloads\Face_classifier\model\class_dictionary.json","r") as f:
        __class_name_to_number = json.load(f)
        __class_Number_to_name = {v:k for k,v in __class_name_to_number.items()}
    global __model
    if __model == None:
        with open (r"C:\Users\Mohamed\Downloads\Face_classifier\model\saved_model.pkl","rb") as f:
            __model = joblib.load(f)
    print("Saving Model Artifacts")

def read_test():
    with open (r"C:\Users\Mohamed\Downloads\Face_classifier\model\test\7.txt") as f :
        return f.read()


def get_cropped_image_if_2_eyes(image_path,b64str):
    #img = io.imread(image_path)
    eye_cascade = cv2.CascadeClassifier(r"C:\Users\Mohamed\Downloads\Face_classifier\model\haar\haarcascade_eye.xml")
    face_cascade =  cv2.CascadeClassifier(r"C:\Users\Mohamed\Downloads\Face_classifier\model\haar\haarcascade_frontalface_default.xml")
    if image_path :
        img = cv2.imread(image_path)
    else :
        img = get_cv2_img_from_base64(b64str)
    if len(img.shape)==3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                return roi_color



if __name__ == "__main__":
    load_artifacts()
    print(classify_image(read_test(),None))
