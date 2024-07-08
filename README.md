[<img src="https://img.shields.io/badge/notion-00AA55?style=for-the-badge&logo=notion&logoColor=white"/>](https://www.notion.so/af9b9d42913f4875b7ec9190ec0f1585)

## Objective
E-commerce Customer Churn Dataset 분석을 통한 고객이탈 예측

## Project Description
1. 주어진 데이터의 열과 고객 이탈 유무간의 상관계수를 도출
2. 필요한 데이터 열을 가공하여 이탈유무와 상관계수가 높은 열을 만들어 낸다.
3. 머신러닝 분류 모델 중 정확도와 AUC가 높은 분류 모델을 찾아낸다.
4. 선정한 모델에 대해 GridSearchCV를 이용해 최적의 하이퍼파라미터를 찾아낸다.
5. 분석한 내용을 직관적인 시각화 자료로 표현한다.
6. 최종적으로 주어진 신규 데이터에 대하여 해당 고객의 이탈 확률을 출력해주는 기능을 구현한다.

## Team 3
김진유(팀장) 박경희 정우영 정인교

## Stack
##### 개발환경
<img src="https://img.shields.io/badge/visualstudiocode-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"><img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white"/><img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white">

##### 데이터 수집 및 처리
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"/><img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">

##### 웹페이지로 시각화
<img src="https://img.shields.io/badge/Streamlit-43B02A?style=for-the-badge&logo=Selenium&logoColor=white">

## Description of Used Models
- LogisticRegression
  
  이진 종속변수와 하나 이상의 독립변수 간의 관계를 분석하고 모델링하는데 사용되는 통계적 방법이다. 독립변수의 선형조합을 로지스틱 함수를 사용하여 종속변수에 대한 확률 점수로 변환하는 모델이다. 로지스틱 회귀는 이진 및 다중 클래스 분류 문제뿐만 아니라 시그모이드 패턴을 따르는 연속 결과 모델링에도 사용되어 의학, 경제, 사회과학 등 다양한 분야에서 널리 사용된다.
  
- DecisionTree
  
  의사 결정 규칙과 그 결과물들을 트리 구조로 도식화한 것이다. 직관적이고 해석하기 쉽기 때문에 데이터 분석과 다른 머신 러닝 모델의 기본 요소로 많이 사용되는 분류 및 예측 모델 중 하나이다. 의사결정나무는 그래프 형태로 구성되고, 각 분기점에서 가능한 선택 사항을 고려하여 데이터를 분류하거나 예측한다. 루트 노드(root node)에서 시작하여 각 분기점마다 하위 분기점(자식 노드)으로 나누어지며, 각 분기점에서는 하나의 특성(feature)을 선택하여 이를 기준으로 데이터를 분류한다. 분류된 데이터는 다시 하위 분기점으로 이동하면서 분류 과정을 지고하여 최종적으로 하나의 결과(결정, 예측)를 도출한다.
  
- RandomForest
  
  분류와 회귀에 사용되는 지도학습 알고리즘으로 여러 개의 의사결정나무 모델을 훈련시킨 결과를 종합해 예측하는 앙상블 알고리즘이다. 훈련 과정에서 구성한 다수의 결정 트리로부터 부류(분류) 또는 평균 예측치(회귀 분석)를 출력함으로써 동작한다. 랜덤 포레스트는 검출, 분류, 그리고 회귀 등 다양한 문제에 활용된다.
  
- XGBoost(Extream Gradient Boosting)
  
  효율적이고 확장 가능한 그라디언트 부스팅 알고리즘의 한 형태이다. 분류 및 회귀에 사용되며 성능과 자원 효율이 좋아 자주 사용되는 알고리즘이다.

## Model Training Results
|모델명|혼동행렬|ROC Curve 그래프|
|---|---|---|
|로지스틱 회귀|<img src = "https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-3Team/blob/main/img/logi_output.png">|<img src = "https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-3Team/blob/main/img/logi_roccurve.png" width = "300" height = "250">|
|의사 결정 나무|<img src = "https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-3Team/blob/main/img/tree_output.png">|<img src = "https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-3Team/blob/main/img/tree_roccurve.png" width = "300" height = "250">|
|랜덤 포레스트|<img src = "https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-3Team/blob/main/img/forest_output.png">|<img src = "https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-3Team/blob/main/img/forest_roccurve.png" width = "300" height = "250">|
|XGBoost|<img src = "https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-3Team/blob/main/img/xgb_output.png">|<img src = "https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-3Team/blob/main/img/xgb_roccurve.png" width = "300" height = "250">|

## Reference
- 자료 출처

  https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

  
- 참고 사이트

  https://www.kaggle.com/code/yasinmhmd/e-commerce-churn-99-accuracy

  https://www.epnc.co.kr/news/articleView.html?idxno=303682

  https://www.investkorea.org/ik-kr/bbs/i-112/detail.do?ntt_sn=491181
