import sys, cv2, numpy as np
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from core.analysis import draw_box, resize_high_quality
from ultralytics import YOLO

class AnalysisState:
    def __init__(self, model=None, min_conf=0.40):
        self.model = model
        self.min_conf = min_conf

class AnalysisCard(QFrame):
    """alt tarafta roi kartları"""
    def __init__(self, data):
        super().__init__()
        self.setFixedSize(200, 240)
        self.setStyleSheet("""
            QFrame { 
                background-color: #2d2d2d; border-radius: 15px; 
                border: 2px solid #3d3d3d; margin: 5px;
            }
            QLabel { border: none; color: white; }
        """)
        layout = QVBoxLayout(self)
        
        title = QLabel(data['id'])
        title.setStyleSheet("color: #8bc34a; font-weight: bold; font-size: 12px;")
        title.setAlignment(Qt.AlignCenter)
        
        img_label = QLabel()
        h, w = data['image'].shape[:2]
        # BGR den RGB ye çevir
        rgb_img = cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_img.data, w, h, w*3, QImage.Format_RGB888)
        img_label.setPixmap(QPixmap.fromImage(qimg).scaled(160, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setStyleSheet("background: #000; border-radius: 5px;")
        
        val_label = QLabel(f"D: {data['fractal']:.4f}")
        val_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff;")
        val_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(title)
        layout.addWidget(img_label)
        layout.addWidget(val_label)

class AnalysisThread(QThread):
    result_ready = Signal(object, list)
    def __init__(self, path, model):
        super().__init__()
        self.path, self.model = path, model

    def run(self):
        frame = cv2.imread(self.path)
        if frame is None: return
        frame = resize_high_quality(frame)
        st = AnalysisState(model=self.model, min_conf=0.40)
        processed_img, data_list = draw_box(st, frame)
        self.result_ready.emit(processed_img, data_list)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fractal Vision Pro - Dental Analysis Station")
        self.setMinimumSize(1300, 900)
        self.setStyleSheet("background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', sans-serif;")
        
        self.model = YOLO("my_model.pt")
        self.image_path = None

        #sol panel kotrol sağ panel analizler
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_h_layout = QHBoxLayout(central_widget)

        #sol panel
        self.side_panel = QVBoxLayout()
        self.side_panel.setContentsMargins(20, 30, 20, 30)
        self.side_panel.setSpacing(15)

        logo_label = QLabel("FRACTAL\nVISION")
        logo_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #8bc34a; line-height: 0.8;")
        self.side_panel.addWidget(logo_label)
        
        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setStyleSheet("color: #333;"); self.side_panel.addWidget(line)

        self.btn_load = QPushButton("📂 RÖNTGEN YÜKLE")
        self.btn_run = QPushButton("⚡ ANALİZİ BAŞLAT")
        self.btn_run.setEnabled(False)
        
        for btn in [self.btn_load, self.btn_run]:
            btn.setFixedHeight(55)
            btn.setCursor(Qt.PointingHandCursor)
            self.side_panel.addWidget(btn)

        self.btn_load.setStyleSheet("QPushButton { background-color: #333; border-radius: 8px; font-weight: bold; } QPushButton:hover { background-color: #444; }")
        self.btn_run.setStyleSheet("QPushButton { background-color: #2e7d32; border-radius: 8px; font-weight: bold; } QPushButton:hover { background-color: #388e3c; } QPushButton:disabled { background-color: #1b331d; color: #555; }")

        self.side_panel.addStretch()
        
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setFixedHeight(200)
        self.status_log.setStyleSheet("background: #1a1a1a; border: 1px solid #333; border-radius: 8px; font-size: 11px; color: #888;")
        self.side_panel.addWidget(QLabel("Sistem Günlüğü:"))
        self.side_panel.addWidget(self.status_log)

        #sağ panel
        self.right_container = QVBoxLayout()
        
        # Üst: Resim Alanı
        self.img_display = QLabel("Analiz için röntgen yükleyiniz...")
        self.img_display.setAlignment(Qt.AlignCenter)
        self.img_display.setStyleSheet("background: #000; border: 2px dashed #333; border-radius: 15px; margin: 10px;")
        self.right_container.addWidget(self.img_display, 10) # Başta tüm alanı kaplar

        # Alt: Analiz Kartları
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setFixedHeight(280)
        self.results_scroll.setHidden(True) # analiz bitince gözükecek
        self.results_scroll.setStyleSheet("background: transparent; border: none; margin-top: 10px;")
        
        self.card_widget = QWidget()
        self.card_layout = QHBoxLayout(self.card_widget)
        self.card_layout.setAlignment(Qt.AlignLeft)
        self.results_scroll.setWidget(self.card_widget)
        
        self.right_container.addWidget(self.results_scroll)

        # Düzenleri birleştir
        self.main_h_layout.addLayout(self.side_panel, 1)
        self.main_h_layout.addLayout(self.right_container, 4)

        # Slot bağlantıları
        self.btn_load.clicked.connect(self.load_image)
        self.btn_run.clicked.connect(self.start_analysis)

    def load_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Röntgen Seç", "", "Images (*.jpg *.png *.jpeg)")
        if file:
            self.image_path = file
            self.results_scroll.setHidden(True) # Yeni resimde kartları gizle
            pixmap = QPixmap(file)
            self.img_display.setPixmap(pixmap.scaled(self.img_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.img_display.setStyleSheet("background: #000; border: 2px solid #333; border-radius: 15px;")
            self.btn_run.setEnabled(True)
            self.status_log.append(f"✅ Görüntü yüklendi: {file.split('/')[-1]}")

    def start_analysis(self):
        self.btn_run.setEnabled(False)
        self.status_log.append("⏳ Analiz süreci başlatıldı, lütfen bekleyiniz...")
        
        # Önceki kartları temizle
        for i in reversed(range(self.card_layout.count())): 
            self.card_layout.itemAt(i).widget().setParent(None)

        self.thread = AnalysisThread(self.image_path, self.model)
        self.thread.result_ready.connect(self.on_analysis_finished)
        self.thread.start()

    def on_analysis_finished(self, processed_img, data_list):
        # 1. İşlenmiş resmi güncelle
        h, w, c = processed_img.shape
        # OpenCV BGR -> RGB çevrimi GUI için şarttır
        rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb_img.data, w, h, w*3, QImage.Format_RGB888)
        self.img_display.setPixmap(QPixmap.fromImage(qi).scaled(self.img_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # 2. Alt paneli göster (RESMİ YUKARI İTER)
        if data_list:
            self.results_scroll.setHidden(False)
            for data in data_list:
                self.card_layout.addWidget(AnalysisCard(data))
            self.status_log.append(f"✅ Analiz tamamlandı. {len(data_list)} ROI işlendi.")
        else:
            self.status_log.append("⚠️ Hiç implant tespit edilemedi.")

        self.btn_run.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())