# fintabular_generation

고려대학교 금융공학 전공 이예진의 석사 학위 논문 코드입니다.

## 1. Findiff 모델 실행 방법
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
* 추론 결과는 기본적으로 findiff/pred_result/result1 경로에 저장됩니다.
* 저장 경로는 --output_dir 옵션으로 변경할 수 있습니다.