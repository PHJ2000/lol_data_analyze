# League of Legends Data Analysis - 2023 시즌 메타 분석

## 프로젝트 개요
이 프로젝트는 **2023년 리그 오브 레전드(LoL) 경기 데이터를 활용하여 시즌 트렌드를 분석하는 프로젝트**입니다.
LCK, LCS, LEC, LPL 리그 데이터를 기반으로 **유리한 포지션 및 챔피언을 식별**하고, 승률이 특정 챔피언에 의해 좌우되는지를 분석합니다.

## 주요 목표
- **어떤 포지션이 전체적으로 유리했는가?**
- **특정 챔피언이 승률에 영향을 미치는가?**
- **팀 경기력과 관련된 주요 지표 탐색**

## 프로젝트 구조
```
LOL_Data_Analysis/
│── 2023_LoL_match_data.csv  # 원본 경기 데이터
│── 2023_LoL_match_data_m.csv  # 가공된 경기 데이터
│── C.csv, C12_13.csv, p_l.csv, pre.csv  # 추가 데이터셋
│── LeagueStats_1.ipynb  # Jupyter Notebook을 활용한 분석
│── hitmap/  # 데이터 시각화 결과
│── README.md  # 프로젝트 설명 파일
```

## 데이터 설명
본 프로젝트에서는 2023년 1월~6월까지의 **9개월치 경기 데이터를 활용**하여 분석을 수행합니다.

### **사용된 데이터 컬럼**
- `gameid`: 경기 ID
- `league`: 리그 (LCK, LCS, LEC, LPL)
- `year`, `split`, `date`: 경기 연도 및 일정 정보
- `teamname`, `playername`: 팀 및 플레이어 정보
- `position`: 포지션 (탑, 정글, 미드, 원딜, 서폿)
- `champion`: 사용한 챔피언
- `result`: 경기 결과 (승/패)
- `kills`, `deaths`, `assists`: KDA 지표
- `dragons`, `barons`, `towers`: 주요 오브젝트 컨트롤
- 기타 지표 포함

## 설치 및 실행 방법
### 1. 필수 라이브러리 설치
```bash
pip install pandas matplotlib seaborn jupyter
```

### 2. Jupyter Notebook 실행
```bash
jupyter notebook
```

Notebook에서 `LeagueStats_1.ipynb` 파일을 열어 분석을 진행합니다.

## 데이터 분석 예제
```python
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv("2023_LoL_match_data.csv")

# 챔피언별 승률 분석
df_winrate = df.groupby("champion")["result"].mean().sort_values(ascending=False)

# 시각화
df_winrate.head(10).plot(kind='bar', figsize=(10,5), title="Top 10 Champion Win Rate")
plt.show()
```

## 필요 라이브러리
- `pandas`
- `matplotlib`
- `seaborn`
- `jupyter`

## 기여 방법
1. 본 레포지토리를 포크합니다.
2. 새로운 브랜치를 생성합니다.
3. 변경 사항을 커밋하고 푸시합니다.
4. Pull Request를 생성하여 기여합니다.

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

