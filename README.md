# Image Stitching

여러 장의 이미지를 자동으로 정합하여 하나의 파노라마 이미지를 생성하는 프로그램.

## 주요 기능

### 1. 자동 이미지 로드
`data/` 폴더의 모든 이미지를 자동으로 읽어들임. 파일 개수에 무관하게 동작.

### 2. 특징점 기반 매칭
- **BRISK** 특징점 검출
- **Lowe's ratio test** (0.75)로 신뢰도 낮은 매치 필터링
- **RANSAC** 기반 호모그래피 추정으로 outlier 제거

### 3. 자동 순서 결정
사용자가 이미지 순서를 지정하지 않아도 알고리즘이 자동으로 좌→우 배치 결정.
- 모든 이미지 쌍의 RANSAC inlier 수 계산
- 합계가 가장 큰 이미지를 기준(ref)으로 선택
- 매칭 점들의 평균 x좌표(`dx`)로 좌우 위치 판별
- ref 기준으로 정렬

### 4. Multi-band Blending
경계선이 보이지 않도록 라플라시안 피라미드 기반 블렌딩 적용.
- 저주파 대역(색감) → 넓은 영역에서 부드럽게 섞기
- 고주파 대역(디테일) → 좁은 영역에서 빠르게 전환
- Distance transform 기반 가중치 마스크 사용

## 데모

### 1. Ratio Test 적용 전
모든 매치를 그대로 `findHomography`에 넘기던 초기 버전.

```python
match = fmatcher.match(descriptors_l, descriptors_r)
```

inlier 비율이 매우 낮아 RANSAC이 잘못된 호모그래피로 수렴 → 빌딩이 사선으로 휘는 결과.

<img src="assets/1.png" alt="Before ratio test" width="100%">


### 2. Ratio Test 적용 후
1등 매치가 2등 매치보다 25% 이상 가까울 때만 사용 (Lowe's ratio test).

```python
knn = fmatcher.knnMatch(descriptors_l, descriptors_r, k=2)
match = [m for m, n in knn if m.distance < 0.75 * n.distance]
```

<img src="assets/2.png" alt="After ratio test" width="100%">


### 3. Multi-band Blending 적용 결과
라플라시안 피라미드 기반 블렌딩.

<img src="assets/3.png" alt="Multi-band blending result" width="100%">


