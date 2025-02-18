from ultralytics import YOLO

def detect_in_video(model_path, video_path, conf_thres=0.2):
    model = YOLO(model_path)
    results = model.predict(
        source=video_path,
        conf=conf_thres,
        save=True,            # <--- mantiene True
        project="inference_results",
        name="video_out",
        exist_ok=True
    )
    print("✅ Inferencia completada.")
    print("Resultados guardados en:", results[0].path)

if __name__ == "__main__":
    model_path = "runs/detect/train10/weights/best.pt"
    video_path = "/Users/macuser/mis_proyectos/proyecto_yolo/videos/test.mp4"

    detect_in_video(model_path, video_path)
