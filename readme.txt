MONITOR DE SOMNOLENCIA - README
  Plataforma: Raspberry Pi 3B + RaspiCam (modulo CSI)
  Sistema operativo: Debian 13 (Trixie) / Raspberry Pi OS basado en Debian 13
  Python: 3.11+
================================================================================
 
 
--------------------------------------------------------------------------------
DESCRIPCION DEL PROYECTO
--------------------------------------------------------------------------------
 
Sistema de deteccion de somnolencia en tiempo real usando vision por computadora.
Captura video desde la camara CSI (RaspiCam), detecta el rostro del conductor o
usuario con clasificadores Haar, mide el Eye Aspect Ratio (EAR) con landmarks
faciales LBF de 68 puntos, y activa una alerta visual cuando detecta ojos cerrados
durante mas de ~2 segundos consecutivos.
 
Resolucion de captura : 720 x 480
FPS objetivo          : 30 (configurable hasta 45)
Interfaz grafica      : PyQt5
Deteccion de rostro   : OpenCV Haar Cascade (frontal + perfil)
Landmarks / EAR       : OpenCV Face LBF (68 puntos)
 
 
--------------------------------------------------------------------------------
ARCHIVOS NECESARIOS EN EL DIRECTORIO DEL PROYECTO
--------------------------------------------------------------------------------
 
Los siguientes archivos deben estar en la misma carpeta que monitor_somnolencia.py:
 
  monitor_somnolencia.py    <- Script principal (este proyecto)
 
  face.xml                  <- Clasificador Haar para rostro frontal
                               Nombre oficial: haarcascade_frontalface_default.xml
                               Ubicacion tipica en Debian:
                               /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml
                               Copiar con:
                               cp /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml ./face.xml
 
  profile.xml               <- Clasificador Haar para rostro de perfil
                               Nombre oficial: haarcascade_profileface.xml
                               Ubicacion tipica en Debian:
                               /usr/share/opencv4/haarcascades/haarcascade_profileface.xml
                               Copiar con:
                               cp /usr/share/opencv4/haarcascades/haarcascade_profileface.xml ./profile.xml
 
  lbfmodel.yaml             <- Modelo de landmarks faciales LBF (68 puntos)
                               NO viene incluido con OpenCV estandar.
                               Descargar desde:
                               https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml
                               Comando de descarga:
                               wget https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml

 

 
 
--------------------------------------------------------------------------------
INSTALACION DE DEPENDENCIAS DEL SISTEMA
--------------------------------------------------------------------------------
 
Actualizar repositorios primero:
 
  sudo apt update && sudo apt upgrade -y
 
Instalar dependencias del sistema operativo:
 
  sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-pyqt5 \
    libopencv-dev \
    python3-opencv \
    libcamera-dev \
    libcamera-apps \
    python3-picamera2 \
    libatlas-base-dev \
    libjasper-dev \
    libhdf5-dev \
    libqt5gui5 \
    libqt5webkit5 \
    libqt5test5 \
    fonts-liberation \
    wget
 
NOTA IMPORTANTE sobre python3-opencv en Debian 13:
El paquete del repositorio puede no incluir el modulo cv2.face (contrib).
Verificar con:
 
  python3 -c "import cv2; print(hasattr(cv2, 'face'))"
 
Si imprime False, instalar la version contrib desde pip (ver seccion siguiente).
 
 
--------------------------------------------------------------------------------
INSTALACION DE DEPENDENCIAS PYTHON
--------------------------------------------------------------------------------
 
Se recomienda usar un entorno virtual para no afectar el sistema:
 
  python3 -m venv venv --system-site-packages
  source venv/bin/activate
 
Instalar dentro del entorno virtual:
 
  pip install --upgrade pip
 
  pip install numpy
 
  pip install opencv-contrib-python-headless==4.8.1.78
 
IMPORTANTE: Usar opencv-contrib-python-headless y NO opencv-python, porque:
  - "contrib" incluye el modulo cv2.face necesario para los landmarks LBF
  - "headless" evita conflictos con las librerias Qt del sistema (PyQt5 ya instalado)
 
Si la version 4.8.1.78 no esta disponible para ARM en pip, instalar la ultima
disponible para armv7l:
 
  pip install opencv-contrib-python-headless
 
Verificar instalacion completa:
 
  python3 -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'face'))"
  python3 -c "import picamera2; print('picamera2 OK')"
  python3 -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK')"
  python3 -c "import numpy; print('numpy', numpy.__version__)"
 
 
--------------------------------------------------------------------------------
ESTRUCTURA DE DIRECTORIOS RECOMENDADA
--------------------------------------------------------------------------------
 
  /home/pi/monitor_somnolencia/
  |
  |-- monitor_somnolencia.py    <- Script principal
  |-- face.xml                  <- Haar frontal (copiado desde haarcascades)
  |-- profile.xml               <- Haar perfil (copiado desde haarcascades)
  |-- lbfmodel.yaml             <- Modelo LBF descargado
  |-- venv/                     <- Entorno virtual Python (opcional)
  |-- README.txt                <- Este archivo
 
 
--------------------------------------------------------------------------------
PASOS DE INSTALACION COMPLETA (RESUMEN RAPIDO)
--------------------------------------------------------------------------------
 
  1. Habilitar camara en /boot/firmware/config.txt y reiniciar
 
  2. sudo apt update && sudo apt upgrade -y
 
  3. sudo apt install -y python3 python3-pip python3-venv python3-pyqt5
       python3-picamera2 python3-opencv libcamera-apps
 
  4. mkdir -p /home/pi/monitor_somnolencia
     cd /home/pi/monitor_somnolencia
 
  5. python3 -m venv venv --system-site-packages
     source venv/bin/activate
 
  6. pip install numpy opencv-contrib-python-headless
 
  7. cp /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml ./face.xml
     cp /usr/share/opencv4/haarcascades/haarcascade_profileface.xml ./profile.xml
 
  8. wget https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml
 
  9. Copiar monitor_somnolencia.py en este directorio
 
  10. python3 monitor_somnolencia.py
 
 
--------------------------------------------------------------------------------
EJECUCION
--------------------------------------------------------------------------------
 
Con entorno virtual activo:
 
  source venv/bin/activate
  python3 monitor_somnolencia.py
 
Sin entorno virtual (si todo se instalo a nivel sistema):
 
  python3 monitor_somnolencia.py
 
Para ejecutar al iniciar sesion automaticamente, agregar al archivo
~/.bashrc o crear un servicio systemd (ver seccion avanzada abajo).
 
 
--------------------------------------------------------------------------------
PARAMETROS CONFIGURABLES EN EL CODIGO
--------------------------------------------------------------------------------
 
Los siguientes valores pueden ajustarse en la seccion superior de
monitor_somnolencia.py segun las condiciones de uso:
 
  CAPTURE_W, CAPTURE_H  Resolucion de captura. Default: 720, 480
  TARGET_FPS            FPS deseados. Default: 30. Maximo estable en RPi 3B: 45
  PROC_SCALE            Factor de reduccion para Haar. Default: 0.5 (360x240)
  DETECT_EVERY_N        Cada cuantos frames detectar rostro. Default: 4
  LANDMARK_EVERY_N      Cada cuantas detecciones calcular landmarks. Default: 2
  EAR_THRESH            Umbral EAR para considerar ojo cerrado. Default: 0.19
                        (valores normales con ojo abierto: 0.28 a 0.35)
  FRAMES_ALERTA         Ticks seguidos de ojos cerrados antes de alertar. Default: 8
                        (~2.1 segundos reales con la config por defecto)
  FRAMES_RESET          Ticks de ojos abiertos para cancelar la alerta. Default: 3
  KPI_WINDOW            Tamano de la ventana de promedio del KPI. Default: 20
  KPI_ALERTA            Porcentaje de KPI que activa el color rojo. Default: 50
 
 
--------------------------------------------------------------------------------
PROBLEMAS FRECUENTES
--------------------------------------------------------------------------------
 
  Camara no detectada:
    - Verificar que el cable CSI este bien conectado (con la RPi apagada)
    - Verificar /boot/firmware/config.txt tenga camera_auto_detect=1
    - Ejecutar: rpicamera-hello para confirmar que rpicamera ve la camara
 
  Error "cv2.face no disponible":
    - Significa que se instalo opencv-python en vez de opencv-contrib-python-headless
    - Desinstalar: pip uninstall opencv-python
    - Reinstalar: pip install opencv-contrib-python-headless
 
  Error al cargar lbfmodel.yaml:
    - El archivo no existe o la ruta es incorrecta
    - Verificar que lbfmodel.yaml este en el mismo directorio que el script
    - Sin este modelo el sistema funciona pero no puede medir EAR (sin landmarks)
 
  face.xml o profile.xml no encontrados:
    - Buscar la ruta correcta en el sistema: find / -name "haarcascade_frontalface*" 2>/dev/null
    - Copiar los archivos al directorio del proyecto con los nombres face.xml y profile.xml
 
  La aplicacion abre pero la imagen esta negra:
    - La camara puede estar siendo usada por otro proceso
    - Verificar con: sudo fuser /dev/video0
    - Matar el proceso y reiniciar la aplicacion
 
  Muchos falsos positivos de somnolencia con luz baja:
    - La iluminacion insuficiente hace fallar los landmarks con frecuencia
    - Agregar iluminacion al entorno o bajar EAR_THRESH a 0.17
    - La caida del EAR por sombras en los ojos es el caso mas comun
 
  Aplicacion muy lenta o cae por debajo de 20 FPS:
    - Subir DETECT_EVERY_N a 5 o 6
    - Bajar CAPTURE_W, CAPTURE_H a 640, 480
    - Verificar temperatura de la RPi: vcgencmd measure_temp
      Si supera 80C hay throttling termico, agregar disipador o ventilador
 
 
--------------------------------------------------------------------------------
DEPENDENCIAS - RESUMEN
--------------------------------------------------------------------------------
 
  Sistema (apt):
    python3                      >= 3.11
    python3-pip
    python3-venv
    python3-pyqt5
    python3-picamera2
    python3-opencv               (base, puede no incluir contrib)
    libcamera-apps
    libatlas-base-dev            (necesario para numpy en ARM)
 
  Python (pip, dentro del venv):
    numpy                        >= 1.24
    opencv-contrib-python-headless  >= 4.8   (incluye cv2.face con LBF)
 
  Archivos de modelo (descargar/copiar manualmente):
    face.xml                     (haarcascade_frontalface_default.xml)
    profile.xml                  (haarcascade_profileface.xml)
    lbfmodel.yaml                (modelo LBF 68 puntos, ~54 MB)
 
 
--------------------------------------------------------------------------------