import cv2
import mediapipe as mp
import numpy as np
import pygame

from numpy import linalg as LA

pygame.mixer.init()
seton_sound = pygame.mixer.Sound('seton.mp3')
count_sound = pygame.mixer.Sound('counter.mp3')

def img_point5(img, heght, width) :
    """　この関数はMediapipeから得られた33個の情報から姿勢情報判断に必要な5つの点のコードをリストとして保存する。
    Input（ poseオブジェクト、画像の縦サイズ、画像の横サイズ　）
    5つポイント　：　耳、肩、ヒップ、膝、かかと
    0から1までのサイズを元のサイズにあわせて座標変換する。
    """
    # 右耳の座標
    right_ear_x = int(img.pose_landmarks.landmark[8].x * width)
    right_ear_y = int(img.pose_landmarks.landmark[8].y * height)
    right_ear_code = [right_ear_x, right_ear_y]

    #　右肩の座標
    right_shoulder_x = int(img.pose_landmarks.landmark[12].x * width)
    right_shoulder_y = int(img.pose_landmarks.landmark[12].y * height)
    right_shoulder_code = [right_shoulder_x, right_shoulder_y]

    #　右ヒップの座標
    right_hip_x = int(img.pose_landmarks.landmark[24].x * width)
    right_hip_y = int(img.pose_landmarks.landmark[24].y * height)
    right_hip_code = [right_hip_x, right_hip_y]

    #　右膝の座標
    right_knee_x = int(img.pose_landmarks.landmark[26].x * width)
    right_knee_y = int(img.pose_landmarks.landmark[26].y * height)
    right_knee_code = [right_knee_x, right_knee_y]

    #　右かかとの座標
    right_heel_x = int(img.pose_landmarks.landmark[30].x * width)
    right_heel_y = int(img.pose_landmarks.landmark[30].y * height)
    right_heel_code = [right_heel_x, right_heel_y]

    return right_ear_code, right_shoulder_code, right_hip_code, right_knee_code, right_heel_code

def draw_line(img, RADIUS, RED, GREEN, THICKNESS, P1, P2, P3, P4, P5 ) :
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


def angle(p1, p2, p3) :
    """
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

def counter(ang, set, count) :
    """
    上半身が下にあるときをFalseにし、上半身が上にあるときをTrueにする
    FalseからTrueになるとカウンターが上がる。
    """
    global count_sound, seton_sound
    #　上半身が正しく上がったか確認
    if set == False and ang < 45 :
        count += 1
        set = True
        count_sound.play()

    # 上半身が正しく下がったか確認
    if set == True and ang > 125 :
        set = False
        seton_sound.play()
    
    return count,set

if __name__ == "__main__":
    # 描画する際の色とマーカの大きさ設定
    RADIUS = 5
    THICKNESS = 2
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

    count = 0
    set_on = False
    mp_pose = mp.solutions.pose

    # 動画を読み込む
    video_file = cv2.VideoCapture('training.mp4')
    # 画像サイズ取得
    height = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Videoファイルとして結果を出力
    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    video  = cv2.VideoWriter('result_video.mp4', fourcc, 20.0, (width, height))

    with mp_pose.Pose(min_detection_confidence=0.5, static_image_mode=True) as pose_detection :
        while video_file.isOpened :
            # 動画から画像を読み込む
            success, image = video_file.read()

            # 入力動画が存在しない場合終了する
            if not success :
                print(" ERROR Video Data")
                break

            # BGRをRGBに変換
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
                count,set_on = counter(ang, set_on, count)

                # カウンターを画面に表示する
                # cv2.putText(image, str(int(ang)), (30,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(image, str(int(count)), (30,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)

                # 画像データに耳、肩、ヒップ、膝、かかとの場所とラインを追加する
                pose_image = draw_line(image, RADIUS, RED, GREEN, THICKNESS, r_ear, r_shoulder, r_hip, r_knee, r_heel)

            # 画像を描画する
            cv2.imshow('Training Helper', pose_image)
            # 動画出力
            video.write(pose_image)

            # ESCキーを押すと終了する
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        video.release()

