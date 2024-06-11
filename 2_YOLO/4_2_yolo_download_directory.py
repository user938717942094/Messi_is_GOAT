from ultralytics import YOLO  # yolo는 epoch할 때 씁니다.
from roboflow import Roboflow  # roboflow의 기능을 사용할 때는 roboflow를 씁니다.
import os # (기본lib)만약 프로젝트에서 폴더가 없다면 폴더를 만들고 저장해주는걸 자동으로 해줍니다.
import shutil # (기본lib)os라이브러리의 약점으로 폴더를 폴더에 옮길 때 이미 폴더가 있다면 오류를 뱉습니다. 덮어쓰지 않도록 도와줍니다. 

## ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
## ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡvvvvvvvvvvvv설정 값 입력하기vvvvvvvvvvvvvvvvvㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
## ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
roboflow_api_key = "Id9ji66mJaSZdL0qXsE1"
roboflow_workspace = "240422d0449mycobot320"
roboflow_project_name = "seven-books"
roboflow_proj_version = 6  # 이걸 제작한 버전에 맞추어야 합니다.

rf = Roboflow(api_key = roboflow_api_key)
project = rf.workspace(roboflow_workspace).project(roboflow_project_name)
version = project.version(roboflow_proj_version)
dataset = version.download("yolov8")

# 다운로드한 데이터셋 경로 (예시로 "yolov8" 형식의 데이터셋이 다운로드됩니다)
downloaded_path = dataset.location  # Roboflow에서 다운로드한 경로

# 원하는 저장 경로 설정
dataset_folder = "datasets"
desired_path = os.path.join(dataset_folder,f"{roboflow_project_name}-{roboflow_proj_version}")

# 다운로드한 파일을 원하는 경로로 이동시키는 함수
def move_dataset(source_path, destination_path):
    # 대상 경로에 동일한 이름의 디렉토리가 이미 존재하는지 확인
    if not os.path.exists(destination_path):
        shutil.move(source_path, destination_path)
        print(f"Dataset moved to {destination_path}")
    else:
        print(f"The directory {destination_path} already exists. No files were moved.")

# 데이터셋 폴더가 없으면 생성
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# 데이터셋 이동
move_dataset(downloaded_path, desired_path)