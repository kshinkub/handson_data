import tensorflow as tf
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

from hdbcli import dbapi
import glob
import os
from PIL import Image
import numpy as np

conn = dbapi.connect(address='localhost',databasename='HXE',port=39013,user='EMLUSER',password='...')
cursor = conn.cursor()

cursor.execute('TRUNCATE TABLE "MNIST"')

img_files = glob.glob("/tmp/sample/test/tf/*")
for img_file in img_files:
    print('inserting '+img_file)
    label = int(os.path.basename(img_file).split("_")[1])

    features = Image.open(img_file).convert('L')
    features.thumbnail((28, 28))
    features = np.array(features, dtype=np.float32)
    features = np.array(features / 255)
    features = features.reshape(1, 784)

    example = tf.train.Example(
        features = tf.train.Features(
            feature = {'x': _float_feature(features[0]),
    }))

    sql = 'INSERT INTO "MNIST" ("Label","Image") VALUES (?,?)'
    cursor.execute(sql, (label, example.SerializeToString()))

cursor.close()
conn.close()
