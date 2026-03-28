import os
import sys
import argparse
import glob

from skimage.morphology import skeletonize
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

BOX_SIZES = [2, 3, 4, 6, 8, 12, 16, 32, 64]

class State:
    def __init__(self):
        self.model = None
        self.labels = None
        self.image_list = []
        self.image_count = 0
        self.source_type = None
        self.min_conf = 0.4
        
        self.box_colors = [
            (164,120,87), (68,148,228), (93,97,2), (178,182,133), (88,159,106),
            (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)
        ]


def score_threshold(binary_img):
    #threshold kalitesi için kontür toplamını ölçer
    cnts, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    score = 0
    for c in cnts:
        score += cv2.arcLength(c, True)
    return score


def auto_best_threshold(gray):
    thresh_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(thresh_val)

def box_cnt(img, box_size):
    if len(img.shape) == 3:
        img = img[:, :, 0]
    h, w = img.shape
    counts = {}
    for size in box_size:
        h_trim = (h // size) * size
        w_trim = (w // size) * size
        trimmed = img[:h_trim, :w_trim]
        boxes = trimmed.reshape(h_trim // size, size, w_trim // size, size)
        non_empty = np.any(boxes != 0, axis=(1, 3))
        counts[size] = int(np.sum(non_empty))
    return counts

def fractal_analysis(sklt, box_counts, plot=False):

    # 0 olanları kaldır
    filtered = {s: c for s, c in box_counts.items() if c > 0}

    # log alınacak en az 2 nokta olmalı
    if len(filtered) < 2:
        print("Fractal analiz için yeterli veri yok (box_counts çok az)")
        return float('nan')

    box_sizes = np.array(list(filtered.keys()), dtype=np.float32)
    counts = np.array(list(filtered.values()), dtype=np.float32)

    log_sizes = np.log(1.0 / box_sizes)
    log_counts = np.log(counts)

    slope, intercept = np.polyfit(log_sizes, log_counts, 1)
    fractal_dimension = slope

    if plot:
        plt.figure(figsize=(11, 5))

        # sol skeleton 
        plt.subplot(1, 2, 1)
        plt.imshow(sklt, cmap='gray')
        plt.title("Skeleton Image")
        plt.axis("off")

        # sağ fractal 
        plt.subplot(1, 2, 2)
        plt.plot(log_sizes, log_counts, 'o', label='Data')
        plt.plot(log_sizes, slope * log_sizes + intercept,
                label=f"D = {fractal_dimension:.2f}")
        plt.xlabel("log(1 / box size)")
        plt.ylabel("log(count)")
        plt.title("Fractal Analysis")
        plt.legend()

        plt.suptitle(f"Fractal Analysis Result (D={fractal_dimension:.2f})", fontsize=14)
        plt.tight_layout()
        plt.show()
    return fractal_dimension

def resize_high_quality(img, max_dim=1000):
    # resimler ekrana sığmıyo
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_args_and_load(state: State):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="YOLO model path (.pt)")
    parser.add_argument('--image', required=True, help="Image file or folder")
    args = parser.parse_args()

    model_path = args.model
    image_source = args.image

    if not os.path.exists(model_path):
        print("HATA: Model bulunamadı.")
        sys.exit(1)

    state.model = YOLO(model_path)
    state.labels = state.model.names

    valid_ext = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    if os.path.isdir(image_source):
        state.source_type = "folder"
        for file in glob.glob(image_source + "/*"):
            ext = os.path.splitext(file)[1]
            if ext in valid_ext:
                state.image_list.append(file)

        if not state.image_list:
            print("HATA: Klasörde uygun resim yok.")
            sys.exit(1)

    elif os.path.isfile(image_source):
        ext = os.path.splitext(image_source)[1]
        if ext not in valid_ext:
            print("HATA: Dosya uzantısı uygun değil.")
            sys.exit(1)

        state.source_type = "image"
        state.image_list.append(image_source)

    else:
        print("HATA: Path geçersiz.")
        sys.exit(1)


def image_process(crop_img):
    # belgede yazan tüm görüntü işlemleri yapıldı
    resized_crop = cv2.resize(crop_img, (crop_img.shape[1], crop_img.shape[0])) #resim küçük, büyüttük
    resized_gauss = cv2.GaussianBlur(resized_crop, (35, 35), 0)
    duplicate = resized_crop.copy()
    subtraction = cv2.subtract(resized_gauss, duplicate)
    add = cv2.add(subtraction, 128)
    _, thresh128 = cv2.threshold(add, 128, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh128, kernel, 1)
    dilated = cv2.dilate(eroded, kernel, 1)
    inverted = cv2.bitwise_not(dilated)

    bw = inverted > 0
    skeleton = skeletonize(bw)
    skeleton = (skeleton.astype(np.uint8) * 255)

    #skeleton yeşil siyah döndürüyor, fractal çin isyah beyaz yaptık
    if len(skeleton.shape) == 3:
        skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)

    skeleton_bgr = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    box_counts = box_cnt(skeleton_bgr, BOX_SIZES)
    fractal=fractal_analysis(skeleton_bgr,box_counts, True)

    print("\n--- FRACTAL ANALYSIS RESULT ---")
    print("Box sizes:", BOX_SIZES)
    print("Box counts:", box_counts)
    print(f"Fractal Dimension: {fractal:.4f}")
    print("--------------------------------\n")

    return skeleton_bgr


def rotated_box(frame, xmin, ymin, xmax, ymax, mask_frame=None, return_rect=False):
   
    obj = frame[ymin:ymax, xmin:xmax]

    gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #grey yap kontür bul
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return (frame, None) if return_rect else frame

    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(int)

    # box ve rect'i global frame koordinatına taşı
    box[:, 0] += xmin
    box[:, 1] += ymin
    (cx, cy), (w, h), angle = rect
    rect = ((cx + xmin, cy + ymin), (w, h), angle)

    cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)
    if mask_frame is not None:
        cv2.drawContours(mask_frame, [box], 0, (0, 255, 255), 2)

    if return_rect:
        return frame, rect
    else:
        return frame

def draw_parallel_rects(mask_frame, rect, rect_size=(20, 10), offset=3, color=(0, 255, 0), search_range=5, return_coords=False):
    """yolodan gelen dikdörtgenler bizim istedğimiz gibi tam çevrelemiyor bjeyi, bu yğüzden ona uygun bir çevreleme yapmak için
    detect edilen objenin kontürüne paralel olacak roiler çizmek daha mantıklı çinkü roileri kestikten sonra döndürme işlemi
    daha zor oldu"""
    (cx, cy), (w, h), angle = rect
    rect_w, rect_h = rect_size
    if w >= h: #objenin uzun ve kısa kenarına göre roileri nereye yerleştireceğimi ölçüyor
        long_len, short_len = w, h
        angle_long = angle
    else:
        long_len, short_len = h, w
        angle_long = angle + 90

    half_short = short_len/2 + rect_w/2 + offset
    quarter_long = long_len/4

    #hesaplanan uzaklıklara göre kareleri yerlerşitr. x uzun y kısa kenar
    centers_local = [
        (quarter_long, -half_short),
        (-quarter_long, -half_short),
        (quarter_long, half_short),
        (-quarter_long, half_short),
    ]

    #açıyı radyana çevir
    theta = np.radians(angle_long)
    gray_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    
    all_boxes = []  # koordinatları kaydetmek için

    for dx, dy in centers_local:
        #dikdörtgenin açısına göre döndür- rotasyon matrisi
        new_x = dx * np.cos(theta) - dy * np.sin(theta)
        new_y = dx * np.sin(theta) + dy * np.cos(theta)
        init_center = (int(cx + new_x), int(cy + new_y))

        best_center = init_center
        max_mean = -1
        for sx in range(init_center[0]-search_range, init_center[0]+search_range+1):
            #her noktadan roi seçerek karşılaştırıp en parlak olanı belirliyor
            for sy in range(init_center[1]-search_range, init_center[1]+search_range+1):
                x1 = np.clip(sx - rect_w//2, 0, mask_frame.shape[1]-1)
                y1 = np.clip(sy - rect_h//2, 0, mask_frame.shape[0]-1)
                x2 = np.clip(sx + rect_w//2, 0, mask_frame.shape[1]-1)
                y2 = np.clip(sy + rect_h//2, 0, mask_frame.shape[0]-1)
                roi = gray_frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                mean_val = np.mean(roi)
                if mean_val > max_mean:
                    max_mean = mean_val
                    best_center = (sx, sy)
        #best entera roi çiziyor
        new_rect = (best_center, (rect_w, rect_h), angle_long)
        box = cv2.boxPoints(new_rect).astype(int)
        #roinin taşian kısımlarını kırp
        box[:,0] = np.clip(box[:,0], 0, mask_frame.shape[1]-1)
        box[:,1] = np.clip(box[:,1], 0, mask_frame.shape[0]-1)

        cv2.drawContours(mask_frame, [box], 0, color, 2)
        all_boxes.append(box)

    return all_boxes

def draw_box(state: State, frame):
    mask_full = find_thresh(frame)

    # crop için orijinal frame kopyasını al
    orig_frame = frame.copy()

    results = state.model(frame, verbose=False)
    detections = results[0].boxes

    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        conf = det.conf.item()
        if conf < state.min_conf:
            continue

        # kontürü çiz ve rect bilgisini al
        frame, rect = rotated_box(frame, xmin, ymin, xmax, ymax, mask_frame=mask_full, return_rect=True)

        if rect is not None:
            # mask üzerinde dikdörtgenleri çiz ve koordinatları al
            boxes = draw_parallel_rects(mask_full, rect, return_coords=True)

            # orijinal resimde dikdörtgenleri çiz ve crop al
            for i, box in enumerate(boxes):
                # yeşil kutular orilinale çiz
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                # orijinal resimden perspektife göre crop
                pts1 = np.float32(box)
                w = int(np.linalg.norm(box[0] - box[1]))
                h = int(np.linalg.norm(box[1] - box[2]))
                if w == 0 or h == 0:
                    continue
                pts2 = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                crop_img = cv2.warpPerspective(orig_frame, M, (w, h))

                # crop'u büyüt
                scale = 3
                crop_img_resized = cv2.resize(crop_img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

                # crop'u image_process fonksiyonuna gönder
                processed_crop = image_process(crop_img_resized)

                # işlenmiş crop'u ekranda göster
                window_name = f"Processed Rect {i+1}"
                cv2.imshow(window_name, processed_crop)

    cv2.imshow("Mask with contours & parallel regions", mask_full)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return frame


def find_thresh(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobelx, sobely)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges_med = cv2.medianBlur(edges, 3)

    best_t = auto_best_threshold(edges_med)
    print("Seçilen en iyi threshold:", best_t)
    _, mask = cv2.threshold(edges_med, best_t, 255, cv2.THRESH_BINARY)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_bgr[:, :, 1] = 0
    mask_bgr[:, :, 0] = 0
    return mask_bgr


def main():
    state = State()
    get_args_and_load(state)

    while state.image_count < len(state.image_list):
        path = state.image_list[state.image_count]
        frame = cv2.imread(path)
        frame = resize_high_quality(frame)
        state.image_count += 1

        processed = draw_box(state, frame)

        cv2.imshow("YOLO Detection", processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Tüm resimler işlendi.")


if __name__ == "__main__":
    main()
