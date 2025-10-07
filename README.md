# Visão Computacional Headless com Raspberry Pi 4 + RaspiCam + YOLOv8 + FIWARE (Ultralight/MQTT)

Projeto pessoal de contagem de pessoas em tempo real usando **Raspberry Pi 4** com **câmera CSI (RaspiCam)**, sistema **Ubuntu Server 24.04 LTS** e processamento **headless**. A aplicação captura frames via **Picamera2/libcamera**, faz pré-processamento leve e executa **YOLOv8** para detecção. O resultado (estado/contagem) é publicado por **MQTT** no formato **Ultralight 2.0** para integração com **FIWARE** (IoT Agent → Orion).

> Nota de transparência: Inteligência Artificial foi utilizada como apoio no desenvolvimento, revisão de arquitetura, correções de cores da câmera (RGB/BGR e ColourGains) e escrita desta documentação.

---

# Integração rápida com o FIWARE Descomplicado

Este projeto foi pensado para funcionar de ponta a ponta com a stack do FIWARE Descomplicado, que oferece orquestração simplificada (Orion, IoT Agent Ultralight, etc.) para testes, PoCs e ensino.

Repositório: FIWARE Descomplicado — (https://github.com/fabiocabrini/fiware)

Público-alvo: estudantes, pesquisadores e desenvolvedores que precisam subir o FIWARE de forma prática para integrar sensores/câmeras e construir provas de conceito com Ultralight/MQTT.

Passos recomendados

Suba o FIWARE Descomplicado conforme o README do repositório acima (IoT Agent + Orion).

Ajuste as variáveis do seu app para apontar ao ambiente FIWARE:

FIWARE_SERVICE, FIWARE_SERVICE_PATH

IOTA_HOST, IOTA_PORT, IOTA_UL_HTTP_PORT

ORION_HOST, ORION_PORT

APIKEY, DEVICE_ID

Provisione a câmera no IoT Agent (Ultralight) e consuma a entidade Cam no Orion.
Uma Collection Postman de exemplo acompanha este projeto.

>Crédito e agradecimento: este projeto utiliza referências e boas práticas do FIWARE Descomplicado, iniciativa do Prof. Dr. Fábio Henrique Cabrini. Caso este material lhe ajude, considere citar e bonificar o trabalho do professor (estrela no repositório, menção em publicações, citações acadêmicas e links), fortalecendo o ecossistema educacional e de pesquisa em FIWARE no Brasil.

## Sumário

- [Arquitetura](#arquitetura)
- [Principais recursos](#principais-recursos)
- [Requisitos](#requisitos)
- [Instalação passo a passo](#instalação-passo-a-passo)
- [Configuração](#configuração)
- [Como executar](#como-executar)
- [Integração com FIWARE (Ultralight/MQTT)](#integração-com-fiware-ultralightmqtt)
- [Coleção Postman](#coleção-postman)
- [Detalhes de processamento e filtros](#detalhes-de-processamento-e-filtros)
- [Snapshots e Debug](#snapshots-e-debug)
- [Correção de cores (libcamera/Picamera2)](#correção-de-cores-libcamerapicamera2)
- [Desempenho e escolha de modelos](#desempenho-e-escolha-de-modelos)
- [Serviço systemd (modo headless)](#serviço-systemd-modo-headless)
- [Boas práticas e segurança](#boas-práticas-e-segurança)
- [Estrutura de diretórios](#estrutura-de-diretórios)
- [Solução de problemas (FAQ)](#solução-de-problemas-faq)
- [Licença MIT](#licença-mit)

---

## Arquitetura

```
RaspiCam (CSI) ──> Picamera2/libcamera (RGB888)
                 └─> Conversão p/ BGR (OpenCV)
                        └─> Pré-proc. leve (WB grayworld, gamma, CLAHE)
                              └─> YOLOv8 (ONNX na CPU; fallback p/ .pt)
                                   └─> Pós-filtros geométricos/ROI
                                        └─> Publicação MQTT (Ultralight 2.0)
                                             └─> IoT Agent (FIWARE) → Orion
```

- Headless: sem preview gráfico (DRM/pykms “stubado”).
- Baixa latência: leitura rate-limited, sem fila.
- Resiliência: fallbacks (MQTT print-only, ONNX → .pt, snapshots).

---

## Principais recursos

- Captura estável com **Picamera2** configurada em **RGB888** e conversão consistente para **BGR** (OpenCV/YOLO).
- Ajuste de cor no sensor:
  - **`auto_grayworld_once`**: mede ganhos por algumas amostras e “fixa” no sensor (mais estável que AWB contínuo).
  - Alternativas: **AWB** por modo de cena ou **gains manuais**.
- Pré-processamento **apenas** para IA (não altera snapshots):
  - **WB gray-world**, **gamma**, **CLAHE (Y)**.
- Detecção **YOLOv8** (CPU):
  - Modelo **ONNX** por padrão; **fallback** automático para `.pt` caso “silêncio” prolongado.
- **Publicação MQTT** no formato **Ultralight 2.0**.
- **Snapshot periódico** único a cada **5 s** (sobrescreve o anterior) para depuração.
- Logs de telemetria do frame (dimensão, média, desvio-padrão) 1x/s.

---

## Requisitos

### Hardware
- Raspberry Pi 4 (2GB+ recomendado)
- Câmera CSI compatível (RaspiCam)
- Armazenamento microSD (Classe A1/A2 recomendado)
- Rede com acesso ao broker MQTT / IoT Agent FIWARE

### Software
- Ubuntu Server **24.04 LTS** para Raspberry Pi
- Pacotes de câmera e Python:
  - `libcamera0`, `libcamera-apps`, `python3-picamera2`
  - `python3-opencv`, `python3-venv` (ou via `pip`)
  - `ultralytics`, `onnxruntime` (se usar onnx), `paho-mqtt`, `numpy`

---

## Instalação passo a passo

### 1) Sistema e câmera

```bash
sudo apt update && sudo apt upgrade -y

# Stack de câmera (dependendo da imagem 24.04, python3-picamera2 pode estar em repo específico)
sudo apt install -y libcamera0 libcamera-apps v4l2loopback-utils

# Opcional (se disponível no repo):
sudo apt install -y python3-picamera2
```

Teste rápido da câmera (checar cores, exposição):

```bash
libcamera-still -o /tmp/test.jpg --awb auto --timeout 200
```

### 2) Ambiente Python

```bash
sudo apt install -y python3-venv python3-pip python3-opencv

cd ~/OPT_DRIVEN
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel
pip install ultralytics onnxruntime paho-mqtt numpy opencv-python
# Se estiver usando OpenCV do apt, pode dispensar opencv-python.
```

> Em dispositivos ARM, a instalação de `onnxruntime` precompilado pode variar. Caso haja dificuldades, considere usar apenas `.pt` inicialmente (o código faz fallback).

---

## Configuração

O arquivo `app.py` contém o dicionário `CONFIG`. Principais chaves:

```python
CONFIG = {
  "capture_width": 1296,
  "capture_height": 972,
  "target_process_fps": 10.0,

  "picam_control_mode": "indoor",   # "auto" | "indoor" | "manual"

  "color": {
    "mode": "auto_grayworld_once",   # "awb" | "manual" | "auto_grayworld_once"
    "awb_mode": "Auto",
    "colour_gains": [3.1, 3.15],     # (R, B) se "manual"
    "ev": 0.8, "saturation": 1.05, "contrast": 1.0, "sharpness": 1.0
  },

  "ai_preproc": {
    "wb": "grayworld", "gamma": 1.15, "clahe": True, "clahe_clip": 2.0, "clahe_grid": 8
  },

  "yolo_model_path": "yolov8n.onnx",
  "imgsz": 256,
  "onnx_input_size": 640,            # imgsz será forçado para 640 ao usar ONNX
  "conf": 0.60, "iou": 0.30, "max_det": 50, "only_person": True,

  "min_person_area_ratio": 0.012,    # filtros geométricos
  "min_person_height_ratio": 0.10,
  "aspect_ratio_range": (0.25, 0.95),
  "roi_norm": None,                  # ex: (x0,y0,x1,y1) normalizado

  "yolo_torch_fallback": "yolov8n.pt",
  "fallback_after_seconds": 8,
  "fallback_pt_imgsz": 320,

  "periodic_snapshot_every_s": 5,
  "periodic_snapshot_path": "/tmp/cam_view.jpg",

  "mqtt": {
    "enabled": True,
    "host": "BROKER_IP",
    "port": 1883,
    "fiware_service": "smart",
    "fiware_service_path": "/",
    "apikey": "TEF",
    "device_id": "cam001",
    "qos": 0, "retain": False
  }
}
```

Ajustes recomendados:

- Em ambientes internos, `gamma` entre 1.10 e 1.25 pode realçar pessoas.
- Se perder pessoas pequenas, reduza `min_person_area_ratio` para `0.008`.

---

## Como executar

```bash
cd ~/OPT_DRIVEN
source .venv/bin/activate
python app.py
```

Logs típicos por segundo:

```
[frame] 1296x972 mean=128.4 std=55.1
[debug] mode=picamera2 count(person)=2 conf_max=0.93 imgsz=640 conf=0.6 iou=0.3 classes={0: 2} | state=detected
```

Snapshots:

- Último frame salvo: `/tmp/cam_last.jpg` (1x/s)
- Snapshot periódico de depuração: `/tmp/cam_view.jpg` (a cada 5 s, sobrescrito)

---

## Integração com FIWARE (Ultralight/MQTT)

Formato **Ultralight 2.0**:

- **Tópico**: `/<apikey>/<device_id>/attrs`
- **Payload**: `s|<state>|c|<count>`
  - `state ∈ { detected, idle }`
  - `count`: inteiros (apenas `detected` envia contagem > 0)

Exemplo:

```
/TEF/cam001/attrs   s|detected|c|3
```

Fluxo recomendado:

1. **IoT Agent (Ultralight)** recebe MQTT.
2. IoT Agent **atualiza entidade** no Orion (NGSIv2/NGSI-LD).
3. **Subscrições** no Orion reagem a mudanças (dashboards, regras).

Provisionamento exemplo (conceitual):

- `device_id`: `cam001`
- `entity_type`: `Cam`
- Atributos:
  - `state` (Text)
  - `count` (Number)

---

## Coleção Postman

- A coleção Postman (incluída no repositório) cobre:
  - Provisionamento de **Device** no IoT Agent Ultralight (via REST).
  - Consulta e atualização da entidade no **Orion**.
  - Exemplos de **subscriptions** para notificar mudanças de `count/state`.

Ajuste host/ports, service/servicePath e apikey conforme seu ambiente.

---

## Detalhes de processamento e filtros

### Pré-processamento (apenas para IA)

- **WB gray-world**: equaliza médias B,G,R.
- **Gamma**: realce de tons médios/pele em ambientes internos.
- **CLAHE (Y)**: melhora contraste preservando crominância.

### Pós-filtros geométricos

- **Área mínima relativa** (`min_person_area_ratio`)
- **Altura mínima relativa** (`min_person_height_ratio`)
- **Faixa de razão de aspecto** (`aspect_ratio_range`)
- **ROI normalizado** (`roi_norm`), se desejar limitar a área útil.

### Classes

- `only_person=True`: filtra apenas classe 0 (person) do COCO.

---

## Snapshots e Debug

- `/tmp/cam_view.jpg`: snapshot periódico único, sobrescrito a cada 5 s (o que a IA “vê” antes do pré-proc).
- `/tmp/cam_last.jpg`: último frame salvo a cada ~1 s.
- Loga `mean/std` para detectar problemas de exposição/escuro.

---

## Correção de cores (libcamera/Picamera2)

Problemas comuns:

1. **Ciano/Azul forte**: ordem de canais. A Picamera2 entrega **RGB**, e o OpenCV usa **BGR**.
   - Solução adotada: configurar `RGB888` na Picamera2 e converter para **BGR** imediatamente após capturar:
     ```python
     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
     ```
2. **Ganho no sensor**:
   - `ColourGains` do libcamera é **(R, B)**. Se trocar para (R, G) ou (B, G), as cores estouram.
   - No modo `auto_grayworld_once`, os ganhos são estimados e fixados para estabilidade.

Teste de referência:

```bash
libcamera-still -o /tmp/test.jpg --awb auto --timeout 200
```

Se o `test.jpg` estiver com cores corretas, a pipeline do app também deverá estar.

---

## Desempenho e escolha de modelos

- **ONNX (CPU)**: boa compatibilidade; `imgsz` fixo (tipicamente **640**). O app força `imgsz` ao detectar ONNX.
- **Fallback `.pt`**: se o ONNX não detectar nada por `fallback_after_seconds`, troca automaticamente para `.pt` com `imgsz` menor (ex.: **320**) para reduzir custo.
- **Taxa de processamento**: `target_process_fps=10.0` costuma ser suficiente para contagem; ajuste conforme CPU livre.
- **Filtros**: reduza `conf` para **0.55–0.60** em ambientes internos; ajuste `min_person_area_ratio` se pessoas estiverem distantes.

---

## Serviço systemd (modo headless)

Arquivo: `/etc/systemd/system/opt-driven.service`

```ini
[Unit]
Description=OPT-Driven Headless Vision
After=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/OPT_DRIVEN
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/ubuntu/OPT_DRIVEN/.venv/bin/python /home/ubuntu/OPT_DRIVEN/app.py
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Ativar:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now opt-driven.service
journalctl -u opt-driven.service -f
```

---

## Boas práticas e segurança

- Isolar o Python em **venv**.
- MQTT: se possível, usar **TLS** e autenticação por usuário/senha; restringir tópicos por **ACL**.
- FIWARE: isolar Orion/IoT Agent em rede segura; aplicar políticas de `service`/`servicePath`.
- Atualizações: travar versões mínimas de libs e testar ao atualizar `ultralytics`/`onnxruntime`.

---

## Estrutura de diretórios

```
OPT_DRIVEN/
├── app.py                      # aplicação principal (headless)
├── yolov8n.onnx               # modelo ONNX (exemplo)
├── yolov8n.pt                 # fallback torchscript/pt (opcional)
├── README.md                  # este documento
└── postman_collection.json    # coleção Postman (provisionamento/consulta)
```

---

## Solução de problemas (FAQ)

**Imagem azul/ciano**  
- Garanta `RGB888` na Picamera2 e conversão `cv2.COLOR_RGB2BGR`.  
- Se usar `manual` em `colour_gains`, lembre: `(R, B)`.

**“No module named picamera2”**  
- Instale `python3-picamera2` (repo da RPi para Ubuntu) ou utilize pacote equivalente. Em último caso, versões `pip` não-oficiais podem não dar suporte completo.

**ONNX erro de dimensão**  
- O modelo ONNX geralmente requer `imgsz=640`. O app corrige isso automaticamente, mas verifique `CONFIG["onnx_input_size"]`.

**CPU alta/baixa taxa**  
- Reduza `target_process_fps`, use `.pt` com `imgsz=320`, ajuste filtros.

**MQTT não conecta**  
- O app continua em “print-only”. Verifique firewall/porta, credenciais e compatibilidade com o IoT Agent.

**Detecção instável**  
- Ajuste `conf` (0.55–0.65), `min_person_area_ratio` (0.008–0.015), ou melhore iluminação.

---

## Licença MIT

Copyright (c) 2025 Fábio Cabrini

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
