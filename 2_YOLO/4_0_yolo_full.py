
'''
딥러닝 중 yolov8방식으로 object detection 하는 방법에 대한 풀 패키지가 들어있습니다.
순서는 
[1] 사진찍기 -> 
[2] roboflow에 사진 올려 데이터라벨링하기 -> 
[3] PC로 다운로드 하기 -> 
[4] 모델 학습시키기 -> 
[5] 학습한 모델 다시 roboflow에 올리기 -> 
[6] 학습한 모델 테스트 하기 입니다.

순서대로 함수화를 해놓아 원하는 함수를 불러서 사용하면 됩니다.

먼저
pip install ultralytics와
pip install roboflow를 해야 합니다. (버전은 상관없습니다.)
그리고 
'''
## [0] 모듈/라이브러리 불러오기 (import)
from ultralytics import YOLO # yolo는 epoch할때 씁니다.
from roboflow import Roboflow # roboflow의 기능을 사용할 때는 roboflow를 씁니다.
import os # (기본lib)만약 프로젝트에서 폴더가 없다면 폴더를 만들고 저장해주는걸 자동으로 해줍니다.
import shutil # (기본lib)os라이브러리의 약점으로 폴더를 폴더에 옮길 때 이미 폴더가 있다면 오류를 뱉습니다. 덮어쓰지 않도록 도와줍니다. 

## ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
## ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡvvvvvvvvvvvv설정 값 입력하기vvvvvvvvvvvvvvvvvㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
## ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
roboflow_api_key = "Id9ji66mJaSZdL0qXsE1"
roboflow_workspace = "240422d0449mycobot320"
roboflow_project_name = "seven-books"
roboflow_proj_version = 6  # 이걸 제작한 버전에 맞추어야 합니다.
## ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
## ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

## [0-1] roboflow API key 와 워크스페이스,프로젝트,버전을 지정해줍니다. (기본이라 주석할 필요는 없습니다.)
rf = Roboflow(api_key = roboflow_api_key)
project = rf.workspace(roboflow_workspace).project(roboflow_project_name)
version = project.version(roboflow_proj_version)

## ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

# ## [3] PC로 다운로드 하기 (워크스페이스 최 상단에 다운하고 원하는 디렉토리에 집어넣는 기능입니다)
# dataset_folder = "datasets"

# dataset = version.download("yolov8") ##사실 이 한줄이 메인입니다. 나머지 지워도 돌아는 갑니다. 단지 나머지는 다운 받고 나서 특정 디렉토리로 이송하는 코드입니다.

# ## 다운로드한 데이터셋 경로 (예시로 "yolov8" 형식의 데이터셋이 다운로드됩니다)
# downloaded_path = dataset.location  ## Roboflow에서 다운로드한 경로

# ## 원하는 저장 경로 설정
# desired_path = os.path.join(dataset_folder,f"{roboflow_project_name}-{roboflow_proj_version}")

# ## 다운로드한 파일을 원하는 경로로 이동시키는 함수
# def move_dataset(source_path, destination_path):
#     ## 대상 경로에 동일한 이름의 디렉토리가 이미 존재하는지 확인
#     if not os.path.exists(destination_path):
#         shutil.move(source_path, destination_path)
#         print(f"Dataset moved to {destination_path}")
#     else:
#         print(f"The directory {destination_path} already exists. No files were moved.")

# ## 데이터셋 폴더가 없으면 생성
# if not os.path.exists(dataset_folder):
#     os.makedirs(dataset_folder)

# ## 데이터셋 이동
# move_dataset(downloaded_path, desired_path)

# ##이러면 워크스페이스 최상단에 되므로 조금 조정할 필요는 있어보입니다.

## ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

## [4] 모델 학습시키기

import os

def train_yolo():
    dataset_location = 'datasets/seven-books-6'  ## 데이터셋 경로 설정 (상대 경로 사용)

    data_yaml = os.path.join(dataset_location, 'data.yaml')

    model = YOLO('1_Common/Camera/2_YOLO/src/src_model/yolov8n.pt') ## 모델 로드 (YOLOv8)
    
    model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        lr0=0.001,
        conf=0.25,
        plots=True
    ) # 모델 학습

##함수로 만들어 놨기 떄문에 아래의 줄을 주석을 풀어서 사용하면 됩니다.
if __name__ == '__main__':
    train_yolo()


## deploy는 만들었던 모델을 다시 roboflow에 업로드 해서 테스트 하기 위함 입니다.
## roboflow에서 Version의 create now version을 해서 모델이 들어가있지 않는 버전인지 확인해야 합니다.

## ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

## [5] 학습한 모델 다시 roboflow에 올리기

#version.deploy("yolov8","C:\\Users\\oli\\Music\\DPOD\\0_KG\\code\\3_vscode\\KG_MIG\\runs\\detect\\train31\\weights","best.pt")


## 인자 1은 모델 종류 (여기서는 yolov8버전을 의미), 인자 2는 주소 (모델 전까지 폴더 주소), 인자 3은 모델 이름 (안넣으면 best.pt로 기본적으로 인식합니다.)
##VScode 에서 Copy Relative Path 할때 항상 \는 python에서 이스케이프 문자열이라 문제가 됩니다.. /로 바꾸거나 \\해줍시다.
#추가, roboflow에서는 자동으로 \\를 넣기 때문에 따라야 합니다..

## ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
