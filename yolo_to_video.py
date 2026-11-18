from ultralytics import YOLO

caminho_video = '/home/leo/deep_learning/yolo-in-video/MVI_8191.MOV'

# carregar o modelo já pré-treinado
modelo = YOLO('yolo11s.pt')

# fazer a aplicação do YOLO
video_com_modelo = modelo.predict(
    source=caminho_video,

    # Resolução
    imgsz=1280,

    # Detecção
    conf=0.30,
    iou=0.45,
    max_det=300,

    # Visualização
    show_labels=True,
    show_conf=True,
    show_boxes=True,
    line_width=2,

    # Processamento
    vid_stride=1,

    # Exportação
    save=True,
    save_txt=True,
    save_conf=True
)