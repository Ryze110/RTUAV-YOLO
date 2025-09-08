from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(
    "/root/autodl-tmp/best_x copy.pt")  # build a new model from YAML
    # Validate the model
    model.val(val=True, data='/root/autodl-tmp/data.yaml', split='val', batch=1, save_json=True)