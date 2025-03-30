# 1주차 타이타닉 캐글 필사

## 참고 노트북

[Titanic Top 4% with ensemble modeling - Yassine Ghouzam](https://www.kaggle.com/code/yassineghouzam/titanic-top-4-with-ensemble-modeling/notebook)


## 내용 정리

### 이상치 처리

```PYTHON
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])

=> 2개 이상 이상치를 가진 행 제거
```

---
### 결측치 처리

```PYTHON
# Filling missing value of Age 

## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med

=> Pclass, SibSp, Parch 기준 median으로 예측하여 채움
```

---
### 모델링

#### 교차검증을 통한 분류 모델 성능 평가
- 여러 개의 분류 모델을 **StratifiedKFold 교차검증**을 통해 비교 평가함
- 교차검증을 통해 각 모델의 평균 정확도를 계산하여 비교함

#### 하이퍼파라미터 최적화
- 선택된 분류 모델에 대해 **GridSearchCV**를 사용하여 최적 하이퍼파라미터를 찾음
    - 사용자가 지정한 모든 하이퍼파라미터 조합을 탐색하여, 최적의 파라미터 조합을 찾아줌

        | 파라미터    | 설명                                     | 예시                                         |
        |-------------|------------------------------------------|----------------------------------------------|
        | estimator   | 사용할 모델                              | `RandomForestClassifier()`                   |
        | param_grid  | 탐색할 하이퍼파라미터의 값 범위          | `{'max_depth': [5, 10], 'n_estimators':[100,200]}` |
        | scoring     | 성능 평가 기준(정확도, 정밀도 등)        | `"accuracy"`                                 |
        | cv          | 교차검증 폴드 수                         | `5` (5-fold 교차검증)                        |
        | n_jobs      | 병렬 처리에 사용할 CPU 코어 수           | `-1` (모든 CPU 사용)                         |
        | verbose     | 진행 상황 출력 (0: 안함, 1: 간단, 2: 상세)| `verbose=1`                                  |
    
#### 학습 곡선 분석
- 학습 곡선을 통해 과적합 여부와 훈련 데이터 크기에 따른 정확도 변화를 시각적으로 분석함

#### 특성 중요도 분석
- 트리 기반 모델의 주요 특성 중요도를 분석함

#### 앙상블 모델링
- 여러 개의 개별 모델을 결합하여 더 높은 성능을 얻는 모델링 방법

    > **투표 기반 분류기(Voting Classifier):**<br>
    > 여러 모델의 예측을 조합하여 최종 결정을 내리는 방식
     
    | 유형          | 설명                                                           |
    |---------------|----------------------------------------------------------------|
    | **Hard Voting** | 개별 모델들이 예측한 결과(클래스)에서 가장 많이 투표된 클래스 선택     |
    | **Soft Voting** | 개별 모델들의 클래스별 예측 확률을 평균낸 후 가장 높은 확률의 클래스 선택 |

    ![스크린샷](../image/screenshot1.png)

    => **Soft Voting** 방식은 각 모델이 반환하는 클래스에 대한 예측 확률을 활용하여, 더욱 정밀하고 일반화된 결과를 얻을 수 있음