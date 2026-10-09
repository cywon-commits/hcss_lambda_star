# tm_cpp — C++ 전달 행렬 생성기 (Phase A)

`reference/python/cyl_ref.py`의 `build()`를 같은 순서·규칙으로 옮긴 것. 상태 번호·전이 순서까지 Python과 같다.

## 빌드·사용
```
make -C tm_cpp                                   # g++ -O2 -ffp-contract=off -fopenmp
tm_cpp/tm_gen regions                            # 영역 계수 8개 검산
tm_cpp/tm_gen d0 a0 a1 a2 a3                     # 시작 앞면 (cyl_ref.py와 같은 permutations 규칙)
tm_cpp/tm_gen build a0 a1 a2 a3 runs/raw_X [d0]  # 원시 디렉터리 (d0 생략 시 자동 탐색)
python3 tools/bin2npz.py runs/raw_X runs/X.npz
```
## 검증
```
bash tm_cpp/test_ref.sh      # 기준 원통 5개 compare_tm.py PASS + d0 탐색 일치 + 영역 계수
python3 tm_cpp/check_s.py    # tilt.py 장 0의 s 대 expected.json
```
## 구현 메모
- 꼭짓점은 정수 4성분. 표준 시작 꼭짓점(높이 최소, 닫힌 고리는 (Im, Re) 최소)은 Q(√3)에서 **정확 산술**로 비교(Python의 round(,9) 대체). 교차 검사만 double + 참조와 같은 허용오차.
- 분할이 교차 검사보다 먼저(참조와 동일). 구멍 계수는 스레드별 메모이제이션, 곱은 uint64 오버플로 검사.
- OpenMP: 상태를 2048개씩 병렬 전개한 뒤 **순차로** 번호 부여 → 결정적이며 단일 스레드와 같은 결과.
- 국소화 등 추가 최적화는 아직 적용하지 않음(필요 없었음).
