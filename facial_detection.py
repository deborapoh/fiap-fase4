import cv2

# Acessar a webcam

def capture_video():
    cap = cv2.VideoCapture(0)

    # Verificar se a webcam existe e foi aberta com sucesso
    if not cap.isOpened():
        print("Erro ao acessar a webcam")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    try:
        # Pra fazer a detecção facial, precisamos pegar frame a frame do video
        while True:
            # Buscar o frame do video
            ret, frame = cap.read()

            if not ret:
                print("Erro ao capturar o frame")
                break

            # O haarcascade entende melhor imagens em escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar faces no frame
            faces = face_cascade.detectMultiScale(
                gray,
                # Escala do modelo (1.1 é o padrão, geralmente funciona bem)
                scaleFactor=1.1,
                # Quantos vizinhos para considerar um rosto (5 é o padrão)
                minNeighbors=5,
                # Tamanho mínimo do rosto (30x30 é o padrão)
                minSize=(30, 30)
            )

            # Desenhar retângulos nas faces detectadas
            for (x, y, w, h) in faces:
                # Desenhar retângulo na face detectada
                # (x, y) é o canto superior esquerdo do retângulo
                # (x + w, y + h) é o canto inferior direito do retângulo
                # (0, 255, 0) é a cor do retângulo (verde)
                # 2 é a espessura do retângulo
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Mostrar o frame com as faces detectadas
            cv2.imshow('Face Detection', frame)

            # Sair do loop se a tecla 'q' for pressionada
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrompido pelo usuário")

    except Exception as e:
        print(f"Erro ao capturar o frame: {e}")

    # Liberar a webcam
    cap.release()
    # Fechar todas as janelas
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_video()
