## YOLOv5 NCNN (coding now ...)
NCNN Deployment code for YOLOv5-v6.0 

Based on https://github.com/ultralytics/yolov5

1. prepare your yolov5.pt file based one MS-COCO or your own dataset.  
2. convert .pt file to .onnx file using export.py in yolov5 git repo.  
3. convert .onnx file to ncnn's .param and .bin files using ncnn converter tools.  
4. change the ncnn and opencv library paths in CMakeLists.txt.  
5. build this project:  
``` cmd
   mkdir build && cd build
   cmake ..
   make
```
6. run.

