import cv2  # Permissões da Webcam
import mediapipe as mp
import random
import pygame

# Inicializa o som
pygame.mixer.init()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
)


# Função pra checar a posição das mãos
def is_hand_open(hand_landmarks):
    try:
        landmarks = hand_landmarks.landmark

        is_index_open = (
            landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        )
        is_middle_open = (
            landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        )
        is_ring_open = (
            landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
            < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
        )
        is_pinky_open = (
            landmarks[mp_hands.HandLandmark.PINKY_TIP].y
            < landmarks[mp_hands.HandLandmark.PINKY_PIP].y
        )

        return is_index_open and is_middle_open and is_ring_open and is_pinky_open
    except Exception as e:
        return False


# Upload das imagens
ASSET_MAP = [
    {"image": "numetal.jpg", "sound": "assets/NOOKIE.mp3"},
    {"image": "avril.jpg", "sound": None},
    {"image": "serj.jpg", "sound": "assets/i-e-a-i.mp3"},
    {"image": "davi.jpg", "sound": "assets/calma-calabreso.mp3"},
    {"image": "calma.jpg", "sound": None},
]


# Definindo o tamanho das imagens
MEME_WIDTH = 200
MEME_HEIGHT = 150

# Carregando as imagens e os áudios
loaded_assets = []

print("Carregando imagens...")
for asset_info in ASSET_MAP:
    try:
        img_path = asset_info["image"]
        sound_path = asset_info["sound"]

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Não foi possível carregar: {img_path}")

        # Redimensiona a imagem
        img_resized = cv2.resize(
            img, (MEME_WIDTH, MEME_HEIGHT), interpolation=cv2.INTER_AREA
        )

        processed_image_tuple = None

        # Processa a transparência
        if img_resized.shape[2] == 4:
            b, g, r, alpha = cv2.split(img_resized)
            img_bgr = cv2.merge((b, g, r))
            alpha_mask = alpha / 255.0
            processed_image_tuple = (img_bgr, alpha_mask)
        else:  # Se não tem imagem
            processed_image_tuple = (img_resized, None)

        loaded_sound_obj = None
        if sound_path:
            loaded_sound_obj = pygame.mixer.Sound(sound_path)

        loaded_assets.append(
            {"image_data": processed_image_tuple, "sound_obj": loaded_sound_obj}
        )

        print(f"Sucesso ao carregar e processar: {img_path}")

    except Exception as e:
        print(f"ERRO ao carregar asset '{img_path}': {e}")
        print("Verifique o nome e se o arquivo existe. Pulando este arquivo.")

if not loaded_assets:  # Verifica se a lista ta vazia
    print("NENHUMA IMAGEM FOI CARREGADA. Verifique os nomes dos arquivos.")
    exit()

print(f"--- {len(loaded_assets)} imagens carregadas. Iniciando webcam. ---")


# Inicializa a Webcam
cap = cv2.VideoCapture(0)

# Controle de estado
pose_detected_previously = False
current_asset_to_display = None  # Guarda a imagem aleatória escolhida

print("Mostre as duas mãos abertas para a 'Nu Metal Pose'! Pressione 'q' para sair.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Inverte para modo selfie
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    open_hands_count = 0
    pose_found_this_frame = False  # Para checar a pose (no frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                bgr_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            if is_hand_open(hand_landmarks):
                open_hands_count += 1

        # Checa se a pose foi detectada (no frame congelado)
        if len(results.multi_hand_landmarks) == 2 and open_hands_count == 2:
            pose_found_this_frame = True

    # Configurando o modo aleatório

    if pose_found_this_frame:
        # Se a pose foi encontrada AGORA e NÃO estava sendo mostrada antes
        if not pose_detected_previously:
            pose_detected_previously = True  # Ativa a trava
            # ESCOLHE UMA NOVA IMAGEM ALEATÓRIA da lista
            current_asset_to_display = random.choice(loaded_assets)

            if current_asset_to_display["sound_obj"]:
                current_asset_to_display["sound_obj"].play()

        if current_asset_to_display:
            # Pega a imagem e a máscara
            img_bgr, img_alpha = current_asset_to_display["image_data"]

            # Posição da imagem
            x_offset = int((bgr_frame.shape[1] - MEME_WIDTH) / 2)
            y_offset = 20

            if (
                y_offset + MEME_HEIGHT < bgr_frame.shape[0]
                and x_offset + MEME_WIDTH < bgr_frame.shape[1]
            ):
                roi = bgr_frame[
                    y_offset : y_offset + MEME_HEIGHT, x_offset : x_offset + MEME_WIDTH
                ]

                if img_alpha is not None:  # Se é png
                    for c in range(0, 3):
                        bgr_frame[
                            y_offset : y_offset + MEME_HEIGHT,
                            x_offset : x_offset + MEME_WIDTH,
                            c,
                        ] = (
                            roi[:, :, c] * (1 - img_alpha)
                            + img_bgr[:, :, c] * img_alpha
                        )
                else:  # Se é jpg
                    bgr_frame[
                        y_offset : y_offset + MEME_HEIGHT,
                        x_offset : x_offset + MEME_WIDTH,
                    ] = img_bgr

    else:  # Se a pose não foi encontrada neste frame
        if pose_detected_previously:
            pygame.mixer.stop()

        pose_detected_previously = False
        current_asset_to_display = None  # Limpa o asset

    cv2.imshow("Nu Metal Detector (Com Imagem!)", bgr_frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

print("Fechando...")
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
hands.close()
