# encoding: utf-8
import chardet

with open('E:/fyp/meteornet-master/scene_flow_kitti/viz1.py','r') as f:
    data = f.read()
type = chardet.detect(data)
print(type)
print (data.decode(type['encoding']))