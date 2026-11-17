# YOLO com Python

## 1. Introdução

YOLO (You Only Look Once) é uma arquitetura de visão computacional utilizada para detectar e localizar objetos em imagens e vídeos.

Com a biblioteca **Ultralytics**, é possível trabalhar com:

- Detecção de objetos (Object Detection).
- Segmentação de objetos (Instance Segmentation).
- Classificação de imagens (Classification).
- Estimativa de pose (Pose Estimation).
- Detecção de caixas orientadas (Oriented Bounding Boxes).
- Rastreamento de objetos (Object Tracking).

Instalação:

```bash
pip install ultralytics opencv-python
```

Importação:

```python
from ultralytics import YOLO
```

---

# 2. Carregamento do modelo

O primeiro passo é carregar os pesos do modelo.

```python
model = YOLO('yolo11n.pt')
```

### Modelos disponíveis

| Modelo       | Descrição                                       |
| ------------ | ----------------------------------------------- |
| `yolo11n.pt` | Nano — menor e mais rápido                      |
| `yolo11s.pt` | Small — leve, com maior capacidade              |
| `yolo11m.pt` | Medium — equilíbrio entre desempenho e precisão |
| `yolo11l.pt` | Large — maior capacidade                        |
| `yolo11x.pt` | Extra Large — maior demanda computacional       |

modelos maiores geralmente exigem mais memória e processamento. O desempenho deve ser avaliado conforme a aplicação.

---

# 3. Método predict()

O método `predict()` executa a inferência sobre imagens, vídeos, webcams ou streams.

Exemplo básico:

```python
results = model.predict(
    source='video.mp4',
    conf=0.40,
    save=True
)
```

## 3.1. Entrada e processamento

| Argumento    | Tipo      | Descrição                              | Exemplo        |
| ------------ | --------- | -------------------------------------- | -------------- |
| `source`     | str/int   | Origem do conteúdo                     | `'video.mp4'`  |
| `imgsz`      | int/tuple | Tamanho da imagem na inferência        | `640`          |
| `device`     | str       | Hardware utilizado                     | `'cpu'`, `'0'` |
| `batch`      | int       | Tamanho do lote de imagens             | `16`           |
| `stream`     | bool      | Retorna resultados progressivamente    | `True`         |
| `vid_stride` | int       | Intervalo de frames processados        | `2`            |
| `half`       | bool      | Utiliza precisão FP16 quando suportada | `True`         |
| `verbose`    | bool      | Exibe informações do processamento     | `True`         |

### Detalhamento

**source**

Define a origem dos dados.

```python
source='video.mp4'
source='imagem.jpg'
source=0  # Webcam
source='https://exemplo.com/video.mp4'
```

**imgsz**

Define a resolução utilizada pelo modelo durante a inferência.

```python
imgsz=640
```

Também é possível especificar dimensões:

```python
imgsz=(640, 640)
```

Resoluções maiores podem ajudar na detecção de objetos pequenos, mas aumentam o custo computacional.

**device**

Define onde o modelo será executado.

```python
device='cpu'
device='0'       # Primeira GPU NVIDIA
device='0,1'     # Duas GPUs
device='mps'     # Apple Silicon, quando suportado
```

**stream**

Controla a forma como os resultados são disponibilizados.

```python
stream=True
```

Com `True`, o método retorna um gerador de resultados, evitando armazenar todas as detecções simultaneamente na memória.

**vid_stride**

Define o intervalo entre frames processados.

```python
vid_stride=1  # Processa todos os frames
vid_stride=2  # Processa um a cada dois frames
vid_stride=5  # Processa um a cada cinco frames
```

Útil para reduzir o processamento em vídeos longos.

---

# 4. Controle das detecções

Esses parâmetros controlam quais objetos serão considerados válidos pelo modelo.

| Argumento      | Tipo  | Descrição                                 | Exemplo        |
| -------------- | ----- | ----------------------------------------- | -------------- |
| `conf`         | float | Confiança mínima da detecção              | `0.40`         |
| `iou`          | float | Threshold de IoU utilizado no NMS         | `0.45`         |
| `classes`      | list  | Classes que serão consideradas            | `[2, 3, 5, 7]` |
| `max_det`      | int   | Máximo de detecções por imagem            | `100`          |
| `agnostic_nms` | bool  | NMS independente da classe                | `True`         |
| `augment`      | bool  | Ativa inferência com aumentação de imagem | `True`         |

## 4.1. Confidence Threshold (conf)

Define a confiança mínima necessária para aceitar uma detecção.

```python
conf=0.25
```

Exemplos:

| Threshold | Comportamento     |
| --------- | ----------------- |
| `0.10`    | Mais permissivo   |
| `0.25`    | Permissivo        |
| `0.50`    | Mais conservador  |
| `0.75`    | Muito conservador |

Quanto maior o threshold, maior a exigência de confiança para aceitar uma detecção.

Aumentar o threshold pode reduzir falsos positivos, mas também pode eliminar objetos corretamente identificados.

## 4.2. Intersection over Union (iou)

O IoU é utilizado no processo de Non-Maximum Suppression (NMS).

O NMS elimina bounding boxes redundantes quando o modelo identifica o mesmo objeto mais de uma vez.

```python
iou=0.45
```

Em geral:

- IoU menor: elimina mais caixas sobrepostas.
- IoU maior: permite manter mais caixas sobrepostas.

## 4.3. Classes (classes)

Permite filtrar as categorias que serão consideradas.

No modelo pré-treinado no COCO:

| ID | Classe      |
| -- | ----------- |
| 0  | Pessoa      |
| 1  | Bicicleta   |
| 2  | Carro       |
| 3  | Motocicleta |
| 5  | Ônibus      |
| 7  | Caminhão    |

Exemplo:

```python
classes=[2, 3, 5, 7]
```

Nesse caso, serão consideradas apenas as categorias de veículos selecionadas.

## 4.4. Maximum Detections (max_det)

Define o número máximo de objetos detectados em cada imagem.

```python
max_det=100
```

Útil para limitar o volume de detecções em cenas muito movimentadas.

## 4.5. Agnostic NMS

Permite realizar a supressão de caixas sobrepostas independentemente da classe.

```python
agnostic_nms=True
```

Pode ser útil quando existem detecções sobrepostas entre categorias diferentes.

---

# 5. Personalização da visualização

Os parâmetros abaixo controlam como as detecções serão apresentadas.

| Argumento     | Tipo | Descrição                                 | Exemplo |
| ------------- | ---- | ----------------------------------------- | ------- |
| `show`        | bool | Exibe o resultado durante o processamento | `True`  |
| `save`        | bool | Salva o resultado processado              | `True`  |
| `show_labels` | bool | Exibe o nome das classes                  | `True`  |
| `show_conf`   | bool | Exibe a confiança                         | `True`  |
| `show_boxes`  | bool | Desenha bounding boxes                    | `True`  |
| `line_width`  | int  | Define a espessura das caixas             | `2`     |

### Exemplo

```python
results = model.predict(
    source='video.mp4',
    save=True,
    show=True,
    show_labels=True,
    show_conf=True,
    show_boxes=True,
    line_width=2
)
```

### Removendo as informações visuais

```python
results = model.predict(
    source='video.mp4',
    save=True,
    show_boxes=False,
    show_labels=False,
    show_conf=False
)
```

---

# 6. Exportação dos resultados

Além de salvar o vídeo processado, é possível exportar informações sobre as detecções.

| Argumento     | Tipo | Descrição                             | Exemplo      |
| ------------- | ---- | ------------------------------------- | ------------ |
| `save`        | bool | Salva o resultado processado          | `True`       |
| `save_txt`    | bool | Exporta as detecções em TXT           | `True`       |
| `save_conf`   | bool | Inclui confiança nos arquivos TXT     | `True`       |
| `save_crop`   | bool | Salva recortes dos objetos detectados | `True`       |
| `save_frames` | bool | Salva frames individuais do vídeo     | `True`       |
| `project`     | str  | Diretório principal de saída          | `'results'`  |
| `name`        | str  | Nome da pasta do experimento          | `'veiculos'` |
| `exist_ok`    | bool | Permite reutilizar a pasta existente  | `True`       |

### Exemplo

```python
results = model.predict(
    source='video.mp4',

    save=True,
    save_txt=True,
    save_conf=True,
    save_crop=True,

    project='resultados',
    name='deteccao_veicular',
    exist_ok=True
)
```

### save_txt

Exporta as coordenadas das detecções em arquivos TXT, normalmente utilizando coordenadas normalizadas.

### save_conf

Inclui a confiança associada a cada detecção no arquivo TXT.

### save_crop

Salva imagens recortadas dos objetos identificados.

Exemplo de estrutura de saída:

```text
resultados/
    deteccao_veicular/
        video.mp4
        labels/
            frame_001.txt
            frame_002.txt
        crops/
            car/
            motorcycle/
            truck/
            bus/
```

---

# 7. Processamento com stream=True

Para vídeos grandes, recomenda-se utilizar o processamento em fluxo.

```python
results = model.predict(
    source='video.mp4',
    stream=True
)

for result in results:

    print(result.boxes)
```

A vantagem é que os resultados são disponibilizados progressivamente, sem manter todas as detecções na memória.

## Extraindo informações das detecções

```python
for result in results:

    for box in result.boxes:

        classe = int(box.cls[0])
        confianca = float(box.conf[0])

        coordenadas = box.xyxy[0].tolist()

        print({
            'classe': classe,
            'confianca': confianca,
            'coordenadas': coordenadas
        })
```

As coordenadas `xyxy` representam:

- `x1`: coordenada horizontal inicial.
- `y1`: coordenada vertical inicial.
- `x2`: coordenada horizontal final.
- `y2`: coordenada vertical final.

---

# 8. Object Tracking

O tracking permite acompanhar objetos ao longo dos frames.

Diferentemente da detecção tradicional, o rastreamento atribui identificadores aos objetos acompanhados.

A Ultralytics disponibiliza algoritmos como:

- ByteTrack.
- BoT-SORT.

## 8.1. Método track()

```python
results = model.track(
    source='video.mp4',
    tracker='bytetrack.yaml',
    conf=0.40,
    save=True,
    show=True
)
```

## 8.2. Principais argumentos

| Argumento | Tipo  | Descrição                                 | Exemplo            |
| --------- | ----- | ----------------------------------------- | ------------------ |
| `source`  | str   | Origem do vídeo                           | `'video.mp4'`      |
| `tracker` | str   | Algoritmo de rastreamento                 | `'bytetrack.yaml'` |
| `conf`    | float | Confiança mínima                          | `0.40`             |
| `iou`     | float | Threshold de IoU                          | `0.45`             |
| `classes` | list  | Classes rastreadas                        | `[2, 3, 5, 7]`     |
| `persist` | bool  | Mantém o estado do tracker entre chamadas | `True`             |
| `stream`  | bool  | Processamento em fluxo                    | `True`             |
| `save`    | bool  | Salva o vídeo processado                  | `True`             |

**Importante:** `persist=True` é especialmente relevante quando os frames são enviados individualmente ao método `track()`. Ele mantém o estado do rastreador entre chamadas consecutivas.

## 8.3. Extraindo IDs dos objetos

```python
for result in results:

    boxes = result.boxes

    if boxes is not None and boxes.id is not None:

        ids = boxes.id.int().cpu().tolist()

        for box, tracking_id in zip(boxes, ids):

            classe = int(box.cls[0])
            confianca = float(box.conf[0])

            print({
                'tracking_id': tracking_id,
                'classe': classe,
                'confianca': confianca
            })
```

**Atenção:** o tracking ID não representa a identidade real do veículo. Ele serve para acompanhar o objeto durante a sequência de vídeo.

---

# 9. Tipos de modelos e tarefas

| Tarefa         | Descrição                                    |
| -------------- | -------------------------------------------- |
| Detection      | Identifica objetos utilizando bounding boxes |
| Segmentation   | Identifica a região exata dos objetos        |
| Classification | Classifica imagens em categorias             |
| Pose           | Identifica pontos-chave do corpo             |
| OBB            | Detecta objetos com caixas orientadas        |

Exemplos:

```python
# Detecção
model = YOLO('yolo11n.pt')

# Segmentação
model = YOLO('yolo11n-seg.pt')

# Classificação
model = YOLO('yolo11n-cls.pt')

# Estimativa de pose
model = YOLO('yolo11n-pose.pt')

# Oriented Bounding Boxes
model = YOLO('yolo11n-obb.pt')
```

---

# 10. Exemplo completo: detecção veicular

Exemplo de configuração para começar um projeto de análise de vídeos de monitoramento veicular.

```python
from ultralytics import YOLO

# Carregar modelo
model = YOLO('yolo11n.pt')

# Processar vídeo
results = model.predict(

    # Entrada
    source='video.mp4',

    # Processamento
    imgsz=640,
    device='cpu',
    vid_stride=1,
    stream=True,

    # Detecção
    conf=0.35,
    iou=0.45,
    classes=[2, 3, 5, 7],
    max_det=100,

    # Visualização
    save=True,
    show=False,
    show_labels=True,
    show_conf=True,
    show_boxes=True,
    line_width=2,

    # Exportação
    save_txt=True,
    save_conf=True,
    save_crop=True,

    # Organização
    project='resultados',
    name='deteccao_veicular',
    exist_ok=True
)

# Extrair informações
for result in results:

    print(f"Objetos detectados: {len(result.boxes)}")

    for box in result.boxes:

        classe = int(box.cls[0])
        confianca = float(box.conf[0])
        coordenadas = box.xyxy[0].tolist()

        print({
            'classe': classe,
            'confianca': confianca,
            'coordenadas': coordenadas
        })
```

---

# 11. Possibilidade de integração com OCR e LPR

Uma aplicação interessante é combinar YOLO com OCR (Optical Character Recognition).

Pipeline sugerido:

1. YOLO identifica os veículos.
2. Tracking acompanha cada veículo.
3. Um segundo modelo detecta a região da placa.
4. OCR realiza a leitura dos caracteres.
5. Python organiza os resultados em uma base estruturada.

Exemplo de estrutura de dados:

| timestamp | tracking_id | classe   | placa   | confiança |
| --------- | ----------: | -------- | ------- | --------: |
| 00:00:01  |           1 | Carro    | ABC1D23 |      0.94 |
| 00:00:03  |           2 | Moto     | DEF4G56 |      0.89 |
| 00:00:05  |           3 | Caminhão | GHI7J89 |      0.96 |

A leitura de placas exige um modelo específico de detecção de placas e um OCR adequado. O YOLO genérico pré-treinado no COCO não realiza essa leitura diretamente.

---

# 12. Documentação oficial

- [Ultralytics — Predict](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics — Track](https://docs.ultralytics.com/modes/track/)
- [Ultralytics — Configuration](https://docs.ultralytics.com/usage/cfg/)
- [Ultralytics — Models](https://docs.ultralytics.com/models/)

**Nota técnica:** os argumentos disponíveis podem variar conforme a versão da Ultralytics e o tipo de modelo. Consulte a documentação correspondente à versão instalada antes de utilizar parâmetros em produção.
