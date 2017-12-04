

import requests
from PIL import Image
import io
import json
import numpy as np
import time

elephant = 'elephant.jpeg'
img_path = '/Users/mwoods/Developer/Repo/trxsfr-learning-web-app/images/ILSVRC/Data/DET/test/ILSVRC2017_test_00000074.JPEG'
with open(img_path, 'rb') as f:
    image_data = f.read()
    image = Image.open(io.BytesIO(image_data))

    for i in range(5000):

        print i, image.size
        s = time.time()
        resp = requests.post('http://35.197.64.160/api/classes', image_data)
        print 'time:', time.time() - s
        #resp = requests.get('http://localhost:5000/api/test/classes')
        print resp.status_code
        print resp.json()['data'][0]
        #data = np.array(resp.json()['data'][0])
        #print data.shape, data.max()
        #print np.array(resp.json()['data'][0]).argsort()[-5:]

        # resp = requests.post('http://35.197.64.160/api/features', image_data)
        # print len( resp.json()['data'] )
        time.sleep(0.1)

# print '########################\n'
# print 'http://localhost:5000/api/test/classes'
# resp = requests.get('http://localhost:5000/api/test/classes')