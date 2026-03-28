import cv2
import numpy as np
from skimage.morphology import skeletonize

BOX_SIZES = [2, 3, 4, 6, 8, 12, 16, 32, 64]

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

def fractal_analysis(sklt, box_counts):
    filtered = {s: c for s, c in box_counts.items() if c > 0}
    if len(filtered) < 2:
        return float('nan')
    box_sizes = np.array(list(filtered.keys()), dtype=np.float32)
    counts = np.array(list(filtered.values()), dtype=np.float32)
    log_sizes = np.log(1.0 / box_sizes)
    log_counts = np.log(counts)
    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return slope

def resize_high_quality(img, max_dim=1000):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def image_process(crop_img):
    resized_gauss = cv2.GaussianBlur(crop_img, (35, 35), 0)
    subtraction = cv2.subtract(resized_gauss, crop_img)
    add = cv2.add(subtraction, 128)
    _, thresh128 = cv2.threshold(add, 128, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh128, kernel, 1)
    dilated = cv2.dilate(eroded, kernel, 1)
    inverted = cv2.bitwise_not(dilated)
    bw = inverted > 0
    skeleton = skeletonize(bw)
    skeleton = (skeleton.astype(np.uint8) * 255)
    if len(skeleton.shape) == 3:
        skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
    skeleton_bgr = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    box_counts = box_cnt(skeleton_bgr, BOX_SIZES)
    fractal = fractal_analysis(skeleton_bgr, box_counts)
    return fractal

def rotated_box(frame, xmin, ymin, xmax, ymax, mask_frame=None, return_rect=False):
    obj = frame[ymin:ymax, xmin:xmax]
    if obj.size == 0: return (frame, None) if return_rect else frame
    gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return (frame, None) if return_rect else frame
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(int)
    box[:, 0] += xmin
    box[:, 1] += ymin
    (cx, cy), (w, h), angle = rect
    rect = ((cx + xmin, cy + ymin), (w, h), angle)
    cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)
    if mask_frame is not None:
        cv2.drawContours(mask_frame, [box], 0, (0, 255, 255), 2)
    return (frame, rect) if return_rect else frame

def draw_parallel_rects(mask_frame, rect, rect_size=(20, 10), offset=1, color=(0, 255, 0), search_range=2):
    (cx, cy), (w, h), angle = rect
    rect_w, rect_h = rect_size
    
    # İmplantın dikey mi yatay mı olduğunu belirle (Genelde h > w olur)
    angle_long = angle if w >= h else angle + 90
    long_dim = w if w >= h else h
    short_dim = h if w >= h else w
    
    # 1. DEĞİŞİKLİK: Offset'i küçük tutarak implanta yakınlaştırdık (Sizin istediğiniz gibi)
    # 2. DEĞİŞİKLİK: quarter_long değerini long_dim/8 yaparak ROI'leri implantın 
    # tam orta gövdesine çektik. Böylece diş kuronundan ve kök ucundan uzaklaştık.
    half_short = short_dim/2 + rect_w/2 + offset
    middle_stay = long_dim/8 
    
    centers_local = [(middle_stay, -half_short), (-middle_stay, -half_short),
                     (middle_stay, half_short), (-middle_stay, half_short)]
    
    theta = np.radians(angle_long)
    gray_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    all_boxes = []
    
    for dx, dy in centers_local:
        new_x = dx * np.cos(theta) - dy * np.sin(theta)
        new_y = dx * np.sin(theta) + dy * np.cos(theta)
        init_center = (int(cx + new_x), int(cy + new_y))
        
        best_center = init_center
        # 3. DEĞİŞİKLİK: Arama kriterini 'en parlak' (diş) yerine 
        # 'kemik yoğunluğu' (orta parlaklık) arayacak şekilde güncelledik.
        # Röntgenlerde 200-255 arası genelde diş/metaldir. 100-180 arası kemiktir.
        min_diff_to_bone = float('inf')
        target_bone_intensity = 140 # Kemik dokusu için ideal gri değer hedefi
        
        # Arama mesafesini kısıtladık (search_range=2) ki çok uzaklaşmasın
        for sx in range(init_center[0]-search_range, init_center[0]+search_range+1):
            for sy in range(init_center[1]-search_range, init_center[1]+search_range+1):
                x1 = np.clip(sx - rect_w//2, 0, mask_frame.shape[1]-1)
                y1 = np.clip(sy - rect_h//2, 0, mask_frame.shape[0]-1)
                x2 = np.clip(sx + rect_w//2, 0, mask_frame.shape[1]-1)
                y2 = np.clip(sy + rect_h//2, 0, mask_frame.shape[0]-1)
                
                roi = gray_frame[y1:y2, x1:x2]
                if roi.size > 0:
                    m = np.mean(roi)
                    # Dişten kaçınmak için: Eğer bölge çok parlaksa (diş/metal) uzak dur
                    if m > 210: 
                        diff = 1000 # Ceza puanı
                    else:
                        diff = abs(m - target_bone_intensity)
                    
                    if diff < min_diff_to_bone:
                        min_diff_to_bone = diff
                        best_center = (sx, sy)
                        
        new_rect = (best_center, (rect_w, rect_h), angle_long)
        box = cv2.boxPoints(new_rect).astype(int)
        box[:,0] = np.clip(box[:,0], 0, mask_frame.shape[1]-1)
        box[:,1] = np.clip(box[:,1], 0, mask_frame.shape[0]-1)
        all_boxes.append(box)
        
    return all_boxes

def find_thresh(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobelx, sobely)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges_med = cv2.medianBlur(edges, 3)
    best_t = auto_best_threshold(edges_med)
    _, mask = cv2.threshold(edges_med, best_t, 255, cv2.THRESH_BINARY)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_bgr[:, :, 0] = 0
    mask_bgr[:, :, 1] = 0
    return mask_bgr

def draw_box(state, frame):
    mask_full = find_thresh(frame)
    orig_frame = frame.copy()
    img_height = frame.shape[0] #röntgen toplam yükseklik
    # IoU eşiğini sıkılaştırarak (örn. 0.35) çakışan kutuları eleyin.
    # Conf eşiğini (örn. 0.45) düşürerek kaçırılan implantı yakalamaya çalışın.
    results = state.model(frame, iou=0.35, conf=0.45, verbose=False)
    detections = results[0].boxes

    analysis_results = [] #kartlardakki veri listesi

    for idx, det in enumerate(detections):
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        if det.conf.item() < state.min_conf: continue

        # İmplant isimlendirme
        implant_name = f"Implant {idx + 1}"

        frame, rect = rotated_box(frame, xmin, ymin, xmax, ymax, mask_frame=mask_full, return_rect=True)
        
        if rect is not None:
            #alt çenedeyse ismi kutunun altına yazdır
            if ymin > img_height / 2:
                text_y = ymax + 25 
            else:
                #üst çenedeyse ismi kutunun üstüne yazdır
                text_y = ymin - 15

            #yazıya gölge ekle
            cv2.putText(frame, implant_name, (xmin, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4) #gölge
            cv2.putText(frame, implant_name, (xmin, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) #yazı

            boxes = draw_parallel_rects(mask_full, rect)
            for i, box in enumerate(boxes):
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                
                pts1 = np.float32(box)
                w = int(np.linalg.norm(box[0] - box[1]))
                h = int(np.linalg.norm(box[1] - box[2]))
                if w <= 0 or h <= 0: continue
                pts2 = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                crop_img = cv2.warpPerspective(orig_frame, M, (w, h))

                crop_img_resized = cv2.resize(crop_img, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
                fractal_val = image_process(crop_img_resized)

                #roilerin fractal değerlerini yaz
                cv2.putText(frame, f"{fractal_val:.2f}", (box[0][0], box[0][1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                #analiz kartlarına verileri kaydet
                analysis_results.append({
                    "id": f"{implant_name} - ROI {i+1}",
                    "fractal": fractal_val,
                    "image": crop_img_resized
                })

    return frame, analysis_results