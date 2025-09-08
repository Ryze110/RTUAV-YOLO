import warnings
import os
from pathlib import Path
from ultralytics import RTDETR,YOLO
import torch

warnings.filterwarnings('ignore')


def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    current_dir = Path(__file__).parent
    yaml_path = 'E:/PycharmProjects/LBUAV-YOLO/ultralytics-main/data.yaml' # Visdrone data.yaml
    # yaml_path = '/root/autodl-tmp/UAVDT_YOLO/data.yaml' # UAVDT
    check_path(yaml_path)
    model = YOLO('E:/PycharmProjects/LBUAV-YOLO/ultralytics-main/ultralytics/cfg/models/11/yolo11s_RTUAVYOLO.yaml')  


    model.train(
        data=str(yaml_path),
        
        # 基础训练参数
        epochs=300,              
        batch=1,                
        imgsz=640,               
        cache=False,              
        device='0',
        workers=8,
        patience = 50,
        mosaic=1,
        optimizer='SGD',
        amp=False,                
        project='runs/trainyolo',
        name='new',
    )