# Esports Analysis - 게이머 키 입력 데이터 분석 프로젝트

## 프로젝트 개요
이 프로젝트는 **Esports(리그 오브 레전드 등)에서 게이머의 키보드 입력 데이터를 수집하고 분석하는 시스템**입니다.
`pynput` 라이브러리를 활용하여 키 입력 빈도를 기록하고, 게임 플레이 스타일을 분석할 수 있습니다.

## 주요 기능
- **키보드 입력 데이터 수집** (`pynput` 활용)
- **특정 키(`q`, `w`, `e`, `r`, `d`, `f`, `a`, `1`~`7`, `s`)의 입력 횟수 추적**
- **엔터 키를 활용한 데이터 수집 활성화/비활성화 기능**
- **데이터를 `pandas`를 사용하여 분석 가능**

## 프로젝트 구조
```
EsportsAnalysis/
│── test.py  # 키 입력 데이터 수집 코드
│── hello.py  # 테스트용 코드
│── test3.py  # 출력문 포함 (분석 기능 없음)
│── tset2.py  # 출력문 포함 (분석 기능 없음)
│── README.md  # 프로젝트 설명 파일
```

## 설치 및 실행 방법
### 1. 필수 라이브러리 설치
```bash
pip install pynput numpy pandas matplotlib
```

### 2. 키 입력 데이터 수집 실행
```bash
python test.py
```

### 3. 수집된 데이터 활용 예시
```python
import pandas as pd

# 예제 데이터
key_data = {
    "q": 94,
    "w": 59,
    "e": 79,
    "r": 33,
    "d": 7,
    "f": 10,
}

# 데이터프레임 생성
df = pd.DataFrame.from_dict(key_data, orient='index', columns=['Frequency'])
print(df)
```

## 필요 라이브러리
- `pynput`
- `numpy`
- `pandas`
- `matplotlib`

## 기여 방법
1. 본 레포지토리를 포크합니다.
2. 새로운 브랜치를 생성합니다.
3. 변경 사항을 커밋하고 푸시합니다.
4. Pull Request를 생성하여 기여합니다.

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

