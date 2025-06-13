import cv2
import os
import numpy as np
import threading
from deepface import DeepFace
from ultralytics import YOLO
import time

# === Configurações ===
CAMINHO_PESSOAS = 'objetos_referencia/Pessoas'

# Lista de EPIs para monitorar
EPI_LABELS = ['Capacete de seguranca', 'Oculos de protecao']
RESOLUCAO = (640, 480)
BACKEND_DETECTOR = 'opencv'
MODELO_RECONHECIMENTO = 'Facenet'
LIMIAR = 15 
CONF = 0.3

# === Variáveis globais para controle ===
epis_detectados = set()
pessoas_identificadas = set()
todos_epis_detectados = False
ultimo_log_epi = time.time()
todas_classes_detectadas = set()  # Armazena todas as classes detectadas pelo YOLO

# === Carrega embeddings das pessoas de referência ===
print("[INFO] Carregando embeddings das pessoas de referência...")
embeddings_pessoas = []
for arquivo in os.listdir(CAMINHO_PESSOAS):
    if arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
        caminho = os.path.join(CAMINHO_PESSOAS, arquivo)
        try:
            emb = DeepFace.represent(
                img_path=caminho,
                model_name=MODELO_RECONHECIMENTO,
                enforce_detection=False,
                detector_backend=BACKEND_DETECTOR
            )
            embedding = emb[0]['embedding'] if isinstance(emb, list) else emb['embedding']
            embeddings_pessoas.append((os.path.splitext(arquivo)[0], embedding))
            print(f"[DEEPFACE] Embedding carregado para: {os.path.splitext(arquivo)[0]}")
        except Exception as e:
            print(f"[ERRO] Erro ao processar {arquivo}: {e}")

print(f"[INFO] {len(embeddings_pessoas)} pessoas carregadas.")

# === Inicializa modelo YOLO para EPIs ===
modelo_yolo = YOLO("best.pt")
print("\n[YOLO] Classes disponíveis no modelo:")
for class_id, class_name in modelo_yolo.names.items():
    print(f"  → Classe {class_id}: {class_name}")
print("")

# === Inicializa webcam ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUCAO[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUCAO[1])
print("[INFO] Pressione 'q' para sair.")

# Variável para resultados
resultados_faces = []
lock = threading.Lock()  # Lock para acesso seguro às variáveis globais

def reconhecer_faces(frame_rgb, frame_shape):
    global resultados_faces
    altura, largura = frame_shape[:2]
    try:
        rostos = DeepFace.extract_faces(
            img_path=frame_rgb,
            enforce_detection=False,
            detector_backend=BACKEND_DETECTOR
        )
        processadas = []
        for face in rostos:
            area = face['facial_area']
            x, y, w, h = area['x'], area['y'], area['w'], area['h']
            x, y = max(0, x), max(0, y)
            w, h = min(w, largura - x), min(h, altura - y)
            if w <= 0 or h <= 0: 
                continue
            
            print(f"[DEEPFACE] Rosto detectado em (x={x}, y={y}, w={w}, h={h})")
            
            face_crop = frame_rgb[y:y + h, x:x + w]
            try:
                emb = DeepFace.represent(
                    img_path=face_crop,
                    model_name=MODELO_RECONHECIMENTO,
                    enforce_detection=False,
                    detector_backend=BACKEND_DETECTOR
                )
                emb_face = emb[0]['embedding'] if isinstance(emb, list) else emb['embedding']
            except Exception as e:
                print(f"[DEEPFACE] Erro ao extrair embedding: {str(e)}")
                continue
                
            nome, menor_dist = "Analisando...", float('inf')
            distancias = []
            
            for nome_ref, emb_ref in embeddings_pessoas:
                dist = np.linalg.norm(np.array(emb_face) - np.array(emb_ref))
                distancias.append(f"{nome_ref}: {dist:.2f}")
                if dist < menor_dist:
                    menor_dist = dist
                    nome = nome_ref
                    
            if menor_dist > LIMIAR:
                nome = "Analisando..."
                print(f"[DEEPFACE] Rosto não reconhecido (distância mínima: {menor_dist:.2f})")
                print(f"[DEEPFACE] Distâncias: {', '.join(distancias)}")
            else:
                print(f"[DEEPFACE] Rosto reconhecido como: {nome} (distância: {menor_dist:.2f})")
                print(f"[DEEPFACE] Distâncias comparativas: {', '.join(distancias)}")
                
            processadas.append((x, y, w, h, nome))
        resultados_faces = processadas
    except Exception as e:
        print(f"[ERRO Reconhecimento] {str(e)}")
        resultados_faces = []

frame_count = 0
print("\n=== INICIANDO SISTEMA DE MONITORAMENTO ===")
print("----------------------------------------")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Falha ao capturar frame da câmera")
        break
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_count % 10 == 0:
        threading.Thread(target=reconhecer_faces, args=(frame_rgb.copy(), frame.shape)).start()

    # Verifica novos reconhecimentos de pessoas
    for (x, y, w, h, nome) in resultados_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, nome, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if nome != "Analisando...":
            with lock:
                if nome not in pessoas_identificadas:
                    pessoas_identificadas.add(nome)
                    print(f"\033[92m[PESSOA] Identificada: {nome}\033[0m")  # Texto verde

    # Verifica detecção de EPIs
    epi_frame = set()
    resultados_yolo = modelo_yolo.predict(source=frame, conf=CONF, save=False, verbose=False)
    
    # Log de todas as detecções
    deteccoes_por_frame = []
    for i, resultado in enumerate(resultados_yolo):
        for j, box in enumerate(resultado.boxes):
            cls = int(box.cls[0])
            label = resultado.names[cls]
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Armazena todas as detecções para log
            deteccoes_por_frame.append({
                "frame": frame_count,
                "detection_id": f"{i}-{j}",
                "label": label,
                "confidence": conf,
                "position": (x1, y1, x2, y2)
            })
            
            # Registra a classe detectada
            todas_classes_detectadas.add(label)
            
            # Filtra apenas EPIs relevantes
            if label in EPI_LABELS and conf > CONF:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"EPI: {label} {conf:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                epi_frame.add(label)

    # Log detalhado a cada 5 segundos
    if time.time() - ultimo_log_epi > 5.0:
        print("\n[YOLO] Resumo de detecções:")
        
        # Imprime todas as classes já detectadas
        print("  Classes detectadas pelo YOLO durante a execução:")
        for classe in sorted(todas_classes_detectadas):
            print(f"    → {classe}")
        
        # Imprime detecções do frame atual
        print("\n  Detecções no frame atual:")
        if deteccoes_por_frame:
            for det in deteccoes_por_frame:
                print(f"    → {det['label']} (conf: {det['confidence']:.2f}, pos: {det['position']})")
        else:
            print("    Nenhuma detecção neste frame")
        
        ultimo_log_epi = time.time()

    # Atualiza lista global de EPIs detectados
    with lock:
        if epi_frame:
            for epi in epi_frame:
                if epi not in epis_detectados:
                    epis_detectados.add(epi)
                    print(f"\033[93m[EPI] Detectado: {epi}\033[0m")  # Texto amarelo
        
        # Verifica se todos os EPIs foram detectados
        if not todos_epis_detectados and epis_detectados.issuperset(set(EPI_LABELS)):
            print("\033[94m[EPI] TODOS EPIS CONFEREM!\033[0m")  # Texto azul
            todos_epis_detectados = True

    try:
        cv2.imshow("Reconhecimento Facial + EPI", frame)
    except cv2.error as e:
        print(f"[ERRO] Falha ao exibir frame: {str(e)}")
        break
        
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Antes de sair, mostrar todas as classes detectadas
        print("\n[YOLO] Resumo final de todas as classes detectadas:")
        for classe in sorted(todas_classes_detectadas):
            print(f"  → {classe}")
        break

cap.release()
cv2.destroyAllWindows()