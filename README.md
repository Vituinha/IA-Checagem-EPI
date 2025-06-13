# IA Checagem EPI - Carlão
Sistema Integrado de Reconhecimento Facial e Monitoramento de EPIs

Este projeto combina técnicas avançadas de visão computacional para criar um sistema de monitoramento em tempo real que:

Identifica funcionários através de reconhecimento facial

Detecta Equipamentos de Proteção Individual (EPIs) usando YOLOv8

Gera alertas visuais e registros quando:

Funcionários são reconhecidos

EPIs obrigatórios são detectados

Todos os EPIs necessários estão sendo utilizados

O sistema é ideal para ambientes industriais, canteiros de obras e locais onde o cumprimento de normas de segurança precisa ser monitorado automaticamente.

Recursos Principais
🎭 Reconhecimento facial com DeepFace (modelo Facenet)

🛡️ Detecção de EPIs com YOLOv8 customizado

📊 Logs detalhados com informações técnicas

🖥️ Interface visual com marcações em tempo real

⚙️ Configuração flexível para diferentes EPIs

📈 Monitoramento contínuo via webcam

Documentação Técnica
Pré-requisitos
Python 3.10+

Bibliotecas: opencv-python, numpy, deepface, ultralytics

Webcam funcional

Modelo YOLO treinado (best.pt)

Estrutura de Arquivos
text
sistema-epi-reconhecimento/
├── main.py                  # Script principal
├── best.pt                  # Modelo YOLO treinado
├── objetos_referencia/
│   └── Pessoas/             # Fotos de referência para reconhecimento
│       ├── funcionario1.jpg
│       ├── funcionario2.png
│       └── ...
└── requirements.txt         # Dependências do projeto
Configuração
python
# === Configurações Modificáveis ===
CAMINHO_PESSOAS = 'objetos_referencia/Pessoas'  # Pasta com fotos de referência
EPI_LABELS = ['Capacete de seguranca', 'Oculos de protecao']  # EPIs a monitorar
RESOLUCAO = (640, 480)       # Resolução da webcam
MODELO_RECONHECIMENTO = 'Facenet'  # Modelo de reconhecimento facial
LIMIAR = 15                   # Limiar de similaridade facial (0-100)
CONF = 0.3                    # Limiar de confiança para detecção de EPIs (0-1)
Funcionamento do Sistema
Inicialização:

Carrega embeddings faciais das fotos de referência

Inicializa modelo YOLO para detecção de EPIs

Inicia captura de vídeo da webcam

Reconhecimento Facial (Thread separada):

python
def reconhecer_faces(frame_rgb, frame_shape):
    # 1. Detecta rostos no frame
    # 2. Extrai embeddings faciais
    # 3. Compara com referências
    # 4. Calcula distâncias de similaridade
    # 5. Identifica funcionários acima do limiar
Detecção de EPIs:

python
resultados_yolo = modelo_yolo.predict(source=frame, conf=CONF)
# Processa detecções e:
# - Marca EPIs no frame
# - Atualiza conjunto de EPIs detectados
# - Gera alertas para novos EPIs
Saída Visual:

Retângulo verde: Funcionário reconhecido

Retângulo vermelho: EPI detectado

Textos identificadores sobre as marcações

Logs do Sistema
Tipo de Log	Cor	Descrição
[PESSOA] Identificada	Verde	Novo funcionário reconhecido
[EPI] Detectado	Amarelo	Novo EPI detectado
[EPI] TODOS CONFEREM	Azul	Todos EPIs obrigatórios detectados
[DEEPFACE]	Branco	Detalhes técnicos de reconhecimento facial
[YOLO]	Branco	Detalhes técnicos de detecção de objetos
Execução
bash
python main.py
Teclas:

q: Encerra o sistema

Console mostra eventos em tempo real com cores

Saída de Exemplo
text
=== INICIANDO SISTEMA DE MONITORAMENTO ===
----------------------------------------
[DEEPFACE] Rosto detectado em (x=120, y=80, w=150, h=150)
[DEEPFACE] Rosto reconhecido como: Joao Silva (distância: 9.82)
[PESSOA] Identificada: Joao Silva
[YOLO] Detectado: Capacete de seguranca (conf: 0.87, pos: (100, 50, 200, 200))
[EPI] Detectado: Capacete de seguranca
[YOLO] Detectado: Oculos de protecao (conf: 0.92, pos: (110, 60, 190, 180))
[EPI] Detectado: Oculos de protecao
[EPI] TODOS EPIS CONFEREM!
Personalização
Adicione fotos de funcionários em objetos_referencia/Pessoas

Atualize EPI_LABELS com os equipamentos relevantes

Ajuste CONF e LIMIAR para maior/menor sensibilidade

Modifique RESOLUCAO conforme sua webcam

Notas
O modelo YOLO (best.pt) deve ser treinado para os EPIs específicos

Fotos de referência devem ser frontais e bem iluminadas

Sistema otimizado para funcionar em tempo real com hardware moderado

Este sistema proporciona uma solução integrada para gestão de segurança e identificação de pessoal, reduzindo a necessidade de supervisão humana constante em ambientes de risco.
