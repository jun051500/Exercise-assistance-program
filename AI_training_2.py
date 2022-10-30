import cv2
import time
import mediapipe as mp
import numpy as np
import pygame
import streamlit as st
import time

from numpy import linalg as LA

movie_path = 'movie/'
img_path = 'image/'

pygame.mixer.init()
seton_sound = pygame.mixer.Sound('seton.mp3')
count_sound = pygame.mixer.Sound('counter.mp3')
gjob_sound = pygame.mixer.Sound('good_job.mp3')
cheer_sound = pygame.mixer.Sound('cheer2.mp3')

def sidebar_parm():
    col1, col2 = st.sidebar.columns(2)
    button_run = col1.button('START')
    button_stop = col2.button('STOP')
    training = st.sidebar.selectbox('トレニンーグ種類',['sit-up Training','push-up Training'])
    mode = st.sidebar.selectbox('モードの選択', ['Use movie file', 'Use WebCam'])
    aim_count = st.sidebar.slider('目標回数', 1, 20, 5)

    uploaded_mv_file = None
    if mode == 'Use movie file':
        uploaded_mv_file = st.sidebar.file_uploader("動画ファイルアップロード", type='mp4')
        if uploaded_mv_file is not None:
            st.sidebar.video(uploaded_mv_file)

    return button_run, button_stop, training, mode, uploaded_mv_file, aim_count


def read_movie_camera(movie_path, uploaded_mv_file):
    mv_file_path = None
    cap_file = None

    if mode == 'Use movie file':
        mv_file_path = movie_path + uploaded_mv_file.name
        cap_file = cv2.VideoCapture(mv_file_path)
    else:
        cap_file = cv2.VideoCapture(0)
    
    return cap_file


def img_point5(point33, height, width) :
    """　この関数はMediapipeから得られた33個の情報から姿勢情報判断に必要な5つの点のコードをリストとして保存する。
    Input（ poseオブジェクト、画像の縦サイズ、画像の横サイズ　）
    5つポイント　：　耳、肩、ヒップ、膝、かかと
    0から1までのサイズを元のサイズにあわせて座標変換する。
    """
    # 右耳の座標
    right_ear_x = int(point33.pose_landmarks.landmark[8].x * width)
    right_ear_y = int(point33.pose_landmarks.landmark[8].y * height)
    right_ear_code = [right_ear_x, right_ear_y]

    #　右肩の座標
    right_shoulder_x = int(point33.pose_landmarks.landmark[12].x * width)
    right_shoulder_y = int(point33.pose_landmarks.landmark[12].y * height)
    right_shoulder_code = [right_shoulder_x, right_shoulder_y]

    #　右ヒップの座標
    right_hip_x = int(point33.pose_landmarks.landmark[24].x * width)
    right_hip_y = int(point33.pose_landmarks.landmark[24].y * height)
    right_hip_code = [right_hip_x, right_hip_y]

    #　右膝の座標
    right_knee_x = int(point33.pose_landmarks.landmark[26].x * width)
    right_knee_y = int(point33.pose_landmarks.landmark[26].y * height)
    right_knee_code = [right_knee_x, right_knee_y]

    #　右かかとの座標
    right_heel_x = int(point33.pose_landmarks.landmark[30].x * width)
    right_heel_y = int(point33.pose_landmarks.landmark[30].y * height)
    right_heel_code = [right_heel_x, right_heel_y]

    return right_ear_code, right_shoulder_code, right_hip_code, right_knee_code, right_heel_code


def get_point3(point33, height, width):
    left_shoulder_x = int(point33.pose_landmarks.landmark[11].x * width)
    left_shoulder_y = int(point33.pose_landmarks.landmark[11].y * height)
    left_shoulder_xy = [left_shoulder_x, left_shoulder_y]

    left_elbow_x = int(point33.pose_landmarks.landmark[13].x * width)
    left_elbow_y = int(point33.pose_landmarks.landmark[13].y * height)
    left_elbow_xy = [left_elbow_x, left_elbow_y]

    left_wrist_x = int(point33.pose_landmarks.landmark[15].x * width)
    left_wrist_y = int(point33.pose_landmarks.landmark[15].y * height)
    left_wrist_xy = [left_wrist_x, left_wrist_y]

    return left_shoulder_xy, left_elbow_xy, left_wrist_xy


def draw_line5(img, RADIUS, RED, GREEN, THICKNESS, P1, P2, P3, P4, P5 ) :
    """
    5つのポイントを線で綱く、各ポイントは〇を入れる。
    入力　：　入力画像、円の半径、円の色、線の色、線の太さ、５つの座標
    出力　：　画像に円と線が追加
    """
    # P1 ear, P2 shoulder, P3 hip, P4 knee, P5 heel
    cv2.circle(img, (P1[0], P1[1]), RADIUS, RED, THICKNESS)
    cv2.circle(img, (P2[0], P2[1]), RADIUS, RED, THICKNESS)
    cv2.circle(img, (P3[0], P3[1]), RADIUS, RED, THICKNESS)
    cv2.circle(img, (P4[0], P4[1]), RADIUS, RED, THICKNESS)
    cv2.circle(img, (P5[0], P5[1]), RADIUS, RED, THICKNESS)

    # 5つの関節部部を線でつなぐ
    cv2.line(img, (P1[0], P1[1]), (P2[0], P2[1]), GREEN, THICKNESS, lineType=cv2.LINE_8, shift=0)
    cv2.line(img, (P2[0], P2[1]), (P3[0], P3[1]), GREEN, THICKNESS, lineType=cv2.LINE_8, shift=0)
    cv2.line(img, (P3[0], P3[1]), (P4[0], P4[1]), GREEN, THICKNESS, lineType=cv2.LINE_8, shift=0)
    cv2.line(img, (P4[0], P4[1]), (P5[0], P5[1]), GREEN, THICKNESS, lineType=cv2.LINE_8, shift=0)

    return img

def draw_line3(image, RADIUS, RED, GREEN, THICKNESS, P1, P2, P3):
    """
    input : shoulder, elbow, wrist　順
　　3つのポイントの印とラインを描画
    """
    cv2.circle(image, (P1[0], P1[1]), RADIUS, RED, THICKNESS)
    cv2.circle(image, (P2[0], P2[1]), RADIUS, RED, THICKNESS)
    cv2.circle(image, (P3[0], P3[1]), RADIUS, RED, THICKNESS)

    cv2.line(image, (P1[0], P1[1]), (P2[0], P2[1]), GREEN, THICKNESS, lineType=cv2.LINE_8)
    cv2.line(image, (P2[0], P2[1]), (P3[0], P3[1]), GREEN, THICKNESS, lineType=cv2.LINE_8)

    return image

def angle(p1, p2, p3) :
    """
    input : numpy.ndarray
    -----------------
    膝とヒップと肩がなす角度を計算する
    肩とヒップのベクトルとヒップと膝のベクトルを利用して内角を計算する
    """
    n_p1 = p1-p2  # ヒップと肩のベクトル
    n_p3 = p3-p2  # ヒップと膝のベクトル

    inn = np.inner(n_p1, n_p3)
    n = LA.norm(n_p1) * LA.norm(n_p3)
    c = inn/n
    a = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))  # 両ベクトルの内角

    return a


def counter(ang, set, count, aim) :
    """
    上半身が下にあるときをFalseにし、上半身が上にあるときをTrueにする
    FalseからTrueになるとカウンターが上がる。
    """
    global count_sound, seton_sound, cheer_sound
    #　上半身が正しく上がったか確認
    if set == False and ang < 45 :
        count += 1
        set = True              # 肩が最高地点に到達した
        if count < aim*0.8 :
            count_sound.play()
        elif count >= aim :
            gjob_sound.play()   # 目標に到達したら’お疲れ様’を言う
        else :
            cheer_sound.play()  # 目標の8割の回数で’頑張れ’で応援する

    # 上半身が正しく下がったか確認
    if set == True and ang > 125 :
        set = False           #　肩が最低地点に到達した
        seton_sound.play()
    
    return count,set

def counter_pushup(ang, set, count, aim) :
    """
    上半身が下にあるときをFalseにし、上半身が上にあるときをTrueにする
    FalseからTrueになるとカウンターが上がる。
    """
    global count_sound, seton_sound, cheer_sound
    #　上半身が正しく下がったか確認　：　腕の角度70度
    if set == False and ang < 70 :
        count += 1
        set = True              # 肩が最高地点に到達した
        if count < aim*0.8 :
            count_sound.play()
        elif count >= aim :
            gjob_sound.play()   # 目標に到達したら’お疲れ様’を言う
        else :
            cheer_sound.play()  # 目標の8割の回数で’頑張れ’で応援する

    # 上半身が正しく上がった確認　：　腕の角度　145度
    if set == True and ang > 145 :
        set = False           #　肩が最低地点に到達した
        seton_sound.play()
    
    return count,set



def sit_train(button_stop, cap_file, mp_pose, mode,set_on, count, aim_count):
    image_container = st.empty()
    height = int(cap_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap_file.get(cv2.CAP_PROP_FRAME_WIDTH))

    with mp_pose.Pose(min_detection_confidence=0.5, static_image_mode=True) as pose_detection :
        while cap_file.isOpened:
            success, image = cap_file.read()

            if not success:
                st.text(' 動画読み込み終了　')
                break

            if button_stop == True:
                break

            # 動画ファイル処理
            if mode == 'Use movie file':
                image = cv2.resize(image , dsize=None, fx=1.0, fy=1.0)
            else:
                image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 人の骨格検出データを読み込む
            pose_data = pose_detection.process(rgb_image)

            if not pose_data.pose_landmarks:
                print('not results')
            else:
                # 骨格情報から耳、肩、ヒップ、膝、かかとの情報を取り出す
                r_ear, r_shoulder, r_hip, r_knee, r_heel = img_point5(pose_data, height, width)

                # 肩とヒップと膝の角度を求める
                ang = angle(np.array(r_shoulder), np.array(r_hip), np.array(r_knee))

                # 角度を用いてカウンター状況を判断する
                count,set_on = counter(ang, set_on, count, aim_count)

                # カウンターを画面に表示する
                cv2.putText(rgb_image, str(int(count)), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(rgb_image, '/', (80,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(rgb_image, str(int(aim_count)), (130,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)

                # 画像データに耳、肩、ヒップ、膝、かかとの場所とラインを追加する
                pose_image = draw_line5(rgb_image, RADIUS, RED, GREEN, THICKNESS, r_ear, r_shoulder, r_hip, r_knee, r_heel)

            # 出力
            # time.sleep(1/fps_val)
            image_container.image(pose_image)

            if count > aim_count :      #　目標回数を過ぎたら終了
                break

    cap_file.release()
    return 0

def pushup_train(button_stop, cap_file, mp_pose, mode, set_on, count, aim_count):
    image_container = st.empty()
    height = int(cap_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    count = 0
    flg_low = False
    
    with mp_pose.Pose(min_detection_confidence=0.5, static_image_mode=True) as pose_detection :
        while cap_file.isOpened:
            success, image = cap_file.read()

            if not success:
                st.text(' 動画読み込み終了　')
                break

            if button_stop == True:
                break

            # 動画ファイル処理
            if mode == 'Use movie file':
                image = cv2.resize(image , dsize=None, fx=1.0, fy=1.0)
            else:
                image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 人の骨格検出データを読み込む
            pose_data = pose_detection.process(rgb_image)

            if not pose_data.pose_landmarks:
                print('not results')
            else:
                left_shoulder_xy, left_elbow_xy, left_wrist_xy = get_point3(pose_data, height, width)

                ang = angle(np.array(left_shoulder_xy), np.array(left_elbow_xy), np.array(left_wrist_xy))

                # 角度を用いてカウンター状況を判断する
                count,set_on = counter_pushup(ang, set_on, count, aim_count)
                
                # カウンターを画面に表示する
                cv2.putText(rgb_image, str(int(count)), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(rgb_image, '/', (80,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(rgb_image, str(int(aim_count)), (130,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)

                # 画像データに肩、肘、手首の場所とラインを追加する
                pose_image = draw_line3(rgb_image, RADIUS, RED, GREEN, THICKNESS, left_shoulder_xy, left_elbow_xy, left_wrist_xy)

            # 出力
            image_container.image(pose_image)

            if count > aim_count :      #　目標回数を過ぎたら終了
                break

    cap_file.release()
    return 0

if __name__ == "__main__":

    # 描画する際の色とマーカの大きさ設定
    RADIUS = 5
    THICKNESS = 2
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

    count = 0
    set_on = False
    mp_pose = mp.solutions.pose

    st.sidebar.title('各種設定')
    button_run, button_stop, training, mode, uploaded_mv_file, aim_count = sidebar_parm()

    st.title('筋トレアシスト')
    st.subheader('頭が画像の左側になるようにして準備してください。')
    if button_run == True:
        if mode == 'Use movie file' and uploaded_mv_file is None:
            st.text('動画ファイルをアップロードしてください')
        else:
            mp_pose = mp.solutions.pose
            # mp_selfie_segmentation = mp.solutions.selfie_segmentation
            cap_file = read_movie_camera(movie_path, uploaded_mv_file)
            
            if training == 'sit-up Training' :
                st.subheader('腹　筋　運　動')
                sit_train(button_stop, cap_file, mp_pose, mode,set_on, count, aim_count)

            if training =='push-up Training' :
                st.subheader('腕　立　て')
                pushup_train(button_stop, cap_file, mp_pose, mode,set_on, count, aim_count)
