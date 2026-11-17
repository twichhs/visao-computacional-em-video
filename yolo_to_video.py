from ultralytics import YOLO

caminho_video = 'caminho_do_seu_video/video.mp4'

# carregar o modelo já pré-treinado
modelo = YOLO('yolo11n.pt')

# fazer a aplicação do YOLO
video_com_modelo = modelo.predict(
    source= caminho_video,
    conf=0.40, # conficança minima dos objetos,
    save = True
)