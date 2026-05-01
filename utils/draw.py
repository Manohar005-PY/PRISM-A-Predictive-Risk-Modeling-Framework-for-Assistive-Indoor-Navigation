import cv2

def draw_box(frame, bbox, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

def draw_text(frame, text, pos, color=(255,255,255)):
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)