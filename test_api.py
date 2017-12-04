

import requests
from PIL import Image
import io
import json
import numpy as np

elephant = 'elephant.jpeg'
img_path = '/Users/mwoods/Developer/Repo/trxsfr-learning-web-app/images/ILSVRC/Data/DET/test/ILSVRC2017_test_00000074.JPEG'
with open(img_path, 'rb') as f:
    image_data = f.read()
    image = Image.open(io.BytesIO(image_data))
    print image.size
    resp = requests.post('http://localhost:5000/api/classes', image_data)
    #resp = requests.get('http://localhost:5000/api/test/classes')
    print resp.status_code
    print resp.json()
    #data = np.array(resp.json()['data'][0])
    #print data.shape, data.max()
    #print np.array(resp.json()['data'][0]).argsort()[-5:]

    resp = requests.post('http://localhost:5000/api/features', image_data)
    print resp.json().keys()

# print '########################\n'
# print 'http://localhost:5000/api/test/classes'
# resp = requests.get('http://localhost:5000/api/test/classes')