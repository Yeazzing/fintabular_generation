# fintabular_generation

고려대학교 금융공학 전공 이예진의 석사 학위 논문 코드입니다.

## 1. 데이터 및 모델 설명

본 프로젝트에서 사용하는 데이터와 학습된 모델 가중치는 아래 링크에서 다운로드할 수 있습니다.

- [**데이터 링크**](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM015&aihubDataSe=data&dataSetSn=71792)

- [**모델 weight 및 부가 파일 링크**](https://drive.google.com/drive/folders/1OPuGtA2fVh_eQlhx-maF3DthOiYfyaLq?usp=drive_link)


위 파일들을 다운로드한 뒤, 각 실행 예시에서 요구하는 경로(`--data_path`, `--metadata`, `--weight_path`, `--condition_csv`)로 지정하여 사용하면 됩니다.
### 모델 출처 
본 레포지토리에서 사용한 생성 모델들은 아래 공개 구현을 기반으로 합니다.
- **FinDiff**  
  - Official repository: https://github.com/sattarov/FinDiff  
  - 금융 테이블 데이터 생성을 위한 diffusion 기반 모델로, 본 프로젝트 목적에 맞게 재구성하여 사용하였습니다.

- **CTGAN / TVAE**  
  - Official repository: https://github.com/sdv-dev/CTGAN  
  - SDV(Synthetic Data Vault) 프로젝트에서 제공하는 GAN/VAE 기반 테이블 데이터 생성 모델로,  
    본 프로젝트에서는 학습 및 추론 스크립트를 일부 수정하여 사용하였습니다.

## 2. FinDiff 모델 실행 방법
- `findiff/` 디렉토리 내의 `main.py`를 통해 실행합니다.
- `--mode` 옵션으로 **학습(train)** 또는 **추론(infer)** 을 선택합니다.
### 학습 (Train)
```bash
python main.py \
  --mode train \
  --data_path /path/to/data.csv
```
### 추론 (Infer)
```bash
python main.py \
  --mode infer \
  --data_path /path/to/data.csv \
  --weight_path /path/to/model.pt
```

---

## 3. CTGAN 모델 실행 방법
- `ctgan_runner/` 디렉토리 내의 `main_ctgan.py`를 통해 실행합니다.
- `--mode` 옵션으로 **학습(train)** 또는 **추론(infer)** 을 선택합니다.
- 학습에 사용하는 `metadata.json`과 추론에 사용하는 `label.csv`는 [모델 weight 링크](https://drive.google.com/drive/folders/1OPuGtA2fVh_eQlhx-maF3DthOiYfyaLq?usp=drive_link)에 함께 포함되어 있습니다.  
  - `metadata.json` : 컬럼 타입/범주형 or 연속형 컬럼 정보를 담은 메타데이터  
  - `label.csv` : eval 데이터의 조건 컬럼 분포(예: LifeStage 분포)에 맞춰 샘플링하기 위한 라벨 파일
### 학습 (Train)
```bash
python main_ctgan.py \
  --mode train \
  --data_path /path/to/data.csv \
  --metadata /path/to/metadata.json
  ```
  ### 추론 (Infer)
  ```bash
python main_ctgan.py \
  --mode infer \
  --weight_path /path/to/model.pt \
  --condition_csv /path/to/label.csv 
```
---

## 4. TVAE 모델 실행 방법
- `ctgan_runner/` 디렉토리 내의 `main_tvae.py`를 통해 실행합니다.
- `--mode` 옵션으로 **학습(train)** 또는 **추론(infer)** 을 선택합니다.
- 학습에 사용하는 `metadata.json` 파일은 [모델 weight 링크](https://drive.google.com/drive/folders/1OPuGtA2fVh_eQlhx-maF3DthOiYfyaLq?usp=drive_link)에 함께 포함되어 있습니다.  
  - `metadata.json` : 컬럼 타입/범주형 or 연속형 컬럼 정보를 담은 메타데이터  
  ### 학습 (Train)
```bash
python main_tvae.py \
  --mode train \
  --data_path /path/to/data.csv \
  --metadata /path/to/metadata.json
  ```
  ### 추론 (Infer)
  ```bash
python main_tvae.py \
  --mode infer \
  --weight_path /path/to/model.pt \
  ```