import argparse
import threading
import time
from pathlib import Path
import requests
import json
import socket
import time

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import numpy as np
from scipy.stats import norm
from filterpy.monte_carlo import systematic_resample

ip_server_url = 'http://124.71.164.229:5000'
pi_ip = None

shared_position = [0, 0]
data_lock = threading.Lock()
stop_flag = threading.Event()

class ParticleFilter:
    def __init__(self, num_particles, x_range, y_range):
        self.num_particles = num_particles
        self.particles = np.empty((num_particles, 2))
        self.particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=num_particles)
        self.particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=num_particles)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, std):
        self.particles[:, 0] += np.random.normal(0, std, size=self.num_particles)
        self.particles[:, 1] += np.random.normal(0, std, size=self.num_particles)

    def update(self, measurement, std):
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        self.weights *= norm(0, std).pdf(distances)
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        indexes = systematic_resample(self.weights)
        self.particles[:] = self.particles[indexes]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)

    def run(self, detections, std, resample_threshold=0.5):
        for detection in detections:
            self.predict(std)
            self.update(detection, std)
            if self.neff() < resample_threshold * self.num_particles:
                self.resample()
            yield self.estimate()

    def neff(self):
        return 1. / np.sum(np.square(self.weights))

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception as e:
        print(f"An error occurred: {e}")
        ip = "Unable to get IP"
    finally:
        s.close()
    return ip

def convert_to_tensor(data):
    tensor_data = torch.tensor(data, device='cuda')
    return tensor_data.cpu().numpy()

def send_position():
    global shared_position
    data = [0, 0]
    last_send_time = time.time()

    while not stop_flag.is_set():
        with data_lock:
            if shared_position != [0, 0]:

                data[0] = shared_position[0]
                data[1] = shared_position[1]

                shared_position[0] = 0
                shared_position[1] = 0

        if data != [0, 0]:
            try:
                last_send_time = time.time()

                # waste too much time on sending data

                response = requests.post(flask_server_url, json={'position_x': str(data[0]), 'position_y': str(data[1])})
                if response.status_code == 200:
                    current_time = time.time()
                    time_interval = current_time - last_send_time
                    last_send_time = current_time
                    print('位置数据:' + 'position_x = ' + str(data[0]) + ' ' + 'position_y = '+ str(data[1]) + ' ' + '已发送到' + flask_server_url)
                    print(f'发送数据用时：: {time_interval:.2f} 秒')
                else:
                    print('无法发送位置数据。状态码:', response.status_code)
                    print('响应内容:', response.content)
            except requests.exceptions.RequestException as e:
                print('发送请求时发生错误:', e)
        
        time.sleep(0.1)
        data = [0, 0]

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://')) or (source == 'pi')

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    global shared_position

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        if source == 'pi':
            dataset = LoadStreams(webcam_url, img_size=imgsz, stride=stride)
        else:
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    pf = ParticleFilter(num_particles=10000, x_range=(0, imgsz), y_range=(0, imgsz))
    target_detected = False
    target_lost = False
    lost_counter = 0
    max_lost_frames = 30  # 允许目标丢失的最大帧数
    initial_target_position = None  # 初始目标位置

    for path, img, im0s, vid_cap in dataset:
        if stop_flag.is_set():
            break

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        detected_in_frame = False
        closest_detection = None
        closest_distance = float('inf')
        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if len(det):
                if not target_detected:
                    # 记录第一个目标
                    target_detected = True
                    initial_target_position = det[0][:4].cpu().numpy()  # 获取初始目标位置
                    closest_detection = det[0]
                    detected_in_frame = True
                    print("First target detected")
                else:
                    # 只跟踪与初始目标最接近的目标
                    for *xyxy, conf, cls in det:
                        x_center = (xyxy[0] + xyxy[2]) / 2
                        y_center = (xyxy[1] + xyxy[3]) / 2
                        initial_x_center = (initial_target_position[0] + initial_target_position[2]) / 2
                        initial_y_center = (initial_target_position[1] + initial_target_position[3]) / 2
                        distance = np.sqrt(convert_to_tensor((x_center - initial_x_center) ** 2 + (y_center - initial_y_center) ** 2))
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_detection = torch.tensor([*xyxy, conf, cls])
                            detected_in_frame = True

            if closest_detection is not None:
                closest_detection = closest_detection.unsqueeze(0)
                closest_detection[:, :4] = scale_coords(img.shape[2:], closest_detection[:, :4], im0.shape).round()
                
                for c in closest_detection[:, -1].unique():
                    n = (closest_detection[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(closest_detection):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    x_center = (xyxy[0] + xyxy[2]) / 2
                    y_center = (xyxy[1] + xyxy[3]) / 2

                    x_center_normalized = (x_center / im0.shape[1]) * 2 - 1
                    y_center_normalized = (y_center / im0.shape[0]) * 2 - 1

                    print(f"Center X normalized: {x_center_normalized:.3f}")
                    print(f"Center Y normalized: {y_center_normalized:.3f}")

                    pf.update(np.array([convert_to_tensor(x_center_normalized), 0]), std=0.05)
                    pf.update(np.array([convert_to_tensor(y_center_normalized), 0]), std=0.05)

                    if pi_ip is not None:
                        with data_lock:
                            # what is estimate()?
                            # shared_position = float(pf.estimate()[0])

                            shared_position[0] = float(x_center_normalized)
                            shared_position[1] = float(y_center_normalized)

            if not detected_in_frame and target_detected:
                lost_counter += 1
                if lost_counter > max_lost_frames:
                    target_lost = True
                    print("Target lost")

            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

def detect_and_optimize():
    for weights in ['yolov7.pt']:
        if stop_flag.is_set():
            break
        detect()
        strip_optimizer(weights)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source, pi for picamera')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.65, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    opt = parser.parse_args()
    print(opt)

    detect_thread = None
    thread_send = None

    if opt.source == 'pi':
        # register the backend IP address
        ip_register_url = ip_server_url + '/register'
        local_ip_data = {'name': 'backend', 'ip': get_local_ip()}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(ip_register_url, json=local_ip_data, headers=headers)
        print(response.json())

        ip_fetch_url = ip_server_url + '/get_ips'

        while True:
            try:
                response = requests.get(ip_fetch_url)
                data = response.json()
                if response.status_code == 200:
                    data = response.json()
                    pi_ip = data.get("pi")
                    if pi_ip:
                        print(f"IP address of pi: {pi_ip}")
                        break
                    else:
                        print("Device 'pi' not found")
                else:
                    print(f"Failed to fetch IPs, status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Error occurred: {e}")

            time.sleep(1)

        flask_server_url = f"http://{pi_ip}:5000/position"
        webcam_url = f"http://{pi_ip}:9000/stream.mjpg"

    with torch.no_grad():
        if opt.update:
            thread_detect_and_optimize = threading.Thread(target=detect_and_optimize)
            thread_detect_and_optimize.start()
            detect_thread = thread_detect_and_optimize
        else:
            thread_detect = threading.Thread(target=detect)
            thread_detect.start()
            detect_thread = thread_detect
            
        time.sleep(10)    
        thread_send = threading.Thread(target=send_position)
        thread_send.start()

    # stop threads on CTRL+C
    try:
        while True:
            if detect_thread is not None and detect_thread.is_alive():
                detect_thread.join(1)
            if thread_send is not None and thread_send.is_alive():
                thread_send.join(1)
    except KeyboardInterrupt:
        print("CTRL+C pressed, stopping threads...")
        stop_flag.set()

        if detect_thread is not None and detect_thread.is_alive():
            detect_thread.join()
        if thread_send is not None and thread_send.is_alive():

            # reset gimbal position
            try:
                response = requests.post(flask_server_url, json={'position_x': str(0), 'position_y': str(0)})
                if response.status_code == 200:
                    print('Reset position has sent to ' + flask_server_url)
                else:
                    print('Unable to reset. Status code: ', response.status_code)
                    print('Response: ', response.content)
            except requests.exceptions.RequestException as e:
                print('An error occurs during reset', e)
            thread_send.join()

    print('All threads stopped.')