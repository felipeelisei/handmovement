import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
INDEX_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP

landmarks_referencia = None


def alinhar_mao(keypoints):
    """Corrige a rotação dos keypoints usando PCA e tenta diferentes ângulos."""
    centroide = np.mean(keypoints, axis=0)
    keypoints_centralizados = keypoints - centroide

    # Aplica PCA para encontrar a orientação principal
    cov_matrix = np.cov(keypoints_centralizados.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Direção principal
    idx_maior = np.argmax(eigenvalues)
    direcao_principal = eigenvectors[:, idx_maior]

    # Garante que a direção principal sempre esteja apontando para cima
    if direcao_principal[1] < 0:
        direcao_principal = -direcao_principal

    # Calcula ângulo
    angulo = np.arctan2(direcao_principal[1], direcao_principal[0])

    # Matriz de rotação
    rot_matrix = np.array([
        [np.cos(-angulo), -np.sin(-angulo)],
        [np.sin(-angulo), np.cos(-angulo)]
    ])

    keypoints_rotacionados = np.dot(keypoints_centralizados, rot_matrix)

    return keypoints_rotacionados


def gerar_variacoes_rotacao(keypoints):
    """Gera versões rotacionadas dos keypoints para tolerar inclinações maiores."""
    variacoes = []
    for angulo in [-90, -45, 0, 45, 90, 135, 180]:
        rad = np.radians(angulo)
        rot_matrix = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)]
        ])
        keypoints_rotacionados = np.dot(keypoints, rot_matrix)
        variacoes.append(keypoints_rotacionados)
    return variacoes


def espelhar_mao(keypoints):
    """Espelha os keypoints horizontalmente (para reconhecer mão invertida)."""
    keypoints_espelhados = keypoints.copy()
    keypoints_espelhados[:, 0] *= -1  # Inverte eixo X (espelhamento)
    return keypoints_espelhados


def get_keypoints(hand_landmarks):
    """Obtém keypoints normalizados e alinhados."""
    keypoints = np.array([(hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y) for i in range(21)])

    # Normaliza pela posição do pulso
    wrist = keypoints[0]
    keypoints -= wrist  

    # Normaliza pelo tamanho da mão
    max_distance = np.max(np.linalg.norm(keypoints, axis=1))
    keypoints /= max_distance

    # Corrige a rotação inicial
    keypoints_alinhados = alinhar_mao(keypoints)

    # Gera variações de rotação para comparação
    variacoes_rotacao = gerar_variacoes_rotacao(keypoints_alinhados)

    # Também gera versão espelhada para cada rotação
    variacoes_espelhadas = [espelhar_mao(var) for var in variacoes_rotacao]

    return variacoes_rotacao, variacoes_espelhadas


def calcular_distancia_dedos(hand_landmarks):
    """Calcula a distância entre polegar e indicador."""
    thumb = np.array([hand_landmarks.landmark[THUMB_TIP].x, hand_landmarks.landmark[THUMB_TIP].y])
    index = np.array([hand_landmarks.landmark[INDEX_TIP].x, hand_landmarks.landmark[INDEX_TIP].y])

    return np.linalg.norm(thumb - index)


def processar_imagem_referencia(image_path):
    """Processa a imagem de referência e extrai os keypoints."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Imagem {image_path} não encontrada!")
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        result = hands.process(image_rgb)
        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            keypoints_ref_variacoes, keypoints_ref_espelhadas = get_keypoints(landmarks)
            distancia_ref = calcular_distancia_dedos(landmarks)
            return keypoints_ref_variacoes, keypoints_ref_espelhadas, distancia_ref
    
    return None, None, None


def comparar_keypoints(keypoints_video_variacoes, keypoints_ref_variacoes, keypoints_ref_espelhadas, distancia_video, distancia_ref, tolerancia=0.12):
    """Compara keypoints permitindo rotações e inversão."""
    if not keypoints_video_variacoes:
        return False

    melhor_erro = float("inf")

    for keypoints_video in keypoints_video_variacoes:
        for keypoints_ref in keypoints_ref_variacoes + keypoints_ref_espelhadas:
            diff = np.linalg.norm(keypoints_video - keypoints_ref, axis=1)
            media_diff = np.mean(diff)
            melhor_erro = min(melhor_erro, media_diff)

    limite_max_distancia = distancia_ref * 1.08  # Permite 8% de variação

    return melhor_erro < tolerancia and distancia_video < limite_max_distancia


# Carrega a imagem de referência
landmarks_referencia_variacoes, landmarks_referencia_espelhadas, distancia_referencia = processar_imagem_referencia("ref.png")

cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                keypoints_video_variacoes, _ = get_keypoints(hand_landmarks)
                distancia_video = calcular_distancia_dedos(hand_landmarks)

                if comparar_keypoints(keypoints_video_variacoes, landmarks_referencia_variacoes, landmarks_referencia_espelhadas, distancia_video, distancia_referencia):
                    cv2.putText(frame, "Gesto Correto ✅", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Gesto Errado ❌", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Comparação de Gestos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
