import cv2

caminho_video = '/home/leo/deep_learning/yolo-in-video/MVI_8191.MOV'
video = cv2.VideoCapture(caminho_video)

fps_original = video.get(cv2.CAP_PROP_FPS)
largura = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

saida = cv2.VideoWriter(
    'video_60fps.mp4',
    fourcc,
    60,
    (largura, altura)
)

while True:

    ret, frame = video.read()

    if not ret:
        break

    # Escreve o frame duas vezes
    saida.write(frame)
    saida.write(frame)

video.release()
saida.release()