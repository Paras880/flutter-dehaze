#app.py
from flask import Flask, json, request, jsonify,url_for
import os
import cv2
import numpy as np
import pandas as pd
import math
import urllib
import urllib.request
from werkzeug.utils import secure_filename
from PIL import Image as im
 
app = Flask(__name__)
 
# app.secret_key = "caircocoders-ednalan"
 
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/')
def main():
    return 'Homepage'
 
@app.route('/upload', methods=['GET'])
def upload_file():
    query = dict(request.form)["url"]
    print(query)
    return jsonify({"response":"hello"})
    # check if the post request has the file part
    # if 'files' not in request.files:
    #     resp = jsonify({'message' : 'No file part in the request'})
    #     resp.status_code = 400
    #     return resp
 
    # files = request.files.getlist('files')
     
    # errors = {}
    # success = False
     
    # for file in files:      
    #     if file and allowed_file(file.filename):
    #         filename = secure_filename(file.filename)
    #         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #         # url = (url_for('static',filename = 'uploads/' + filename))
    #         # url_response = urllib.request.urlopen(url)
    #         # img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    #         # result = dehaze(img_array)
    #         # data = im.fromarray(result)
    #         # data.save(os.path.join(app.config['RESULT_FOLDER'], filename))
    #         success = True
    #     else:
    #         errors[file.filename] = 'File type is not allowed'
    
 
    # if success and errors:
    #     errors['message'] = 'File(s) successfully uploaded'
    #     resp = jsonify(errors)
    #     resp.status_code = 500
    #     return resp
    # if success:
    #     resp = jsonify({'message' : 'Files successfully uploaded'})
    #     resp.status_code = 201
    #     return resp
    # else:
    #     resp = jsonify(errors)
    #     resp.status_code = 500
    #     return resp

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    #print(b , g , r)
    dc = cv2.min(cv2.min(r,g),b);
    #print(dc)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def dehaze(img):
    
    
    I = img.astype('float64')/255;

    dark = DarkChannel(I,15);
    A = AtmLight(I,dark);
    te = TransmissionEstimate(I,A,15);
    t = TransmissionRefine(img,te);


    #dcp
    J = Recover(I,te,A,0.1);
    #guided filter
    J = Recover(I,t,A,0.1);

    return J*255
    # cv2.imwrite("dehazed",J*255);

 
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT',8080)),debug=True)
