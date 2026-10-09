# hcss_lambda_star

2D 강체 코어 + 계단 어깨(HCSS) 모형의 λ* = 2cos15° 축퇴점에서 나타나는 **정삼각형 + 30° 마름모 타일링**의 후속 연구 저장소.
모형 정의, 논문 원고(hcss_tiling.tex), 예비 실행 1–10은 [cywon-commits/hardcore_square_shoulder](https://github.com/cywon-commits/hardcore_square_shoulder)에 있다.

## 지금까지의 결과 (2026-10-09 기준)

| 주제 | 결과 |
|---|---|
| 치환 대칭 | 등변 치환 6가지, C₁₂ 고정점이 껍질 안에 있는 것 3가지, 치환은 본질적으로 카이랄 |
| 모형집합 | (6,15), (6,4) 모두 순수점 — 프랙탈 지지집합 겹침 판정으로 엄밀(비일치 잠재 겹침 ρ = 5 < λ²). 창 경계 차원 (6,15) ≈ 1.22(추측 log5/logλ), (6,4) ≈ 1.5–1.6(수치) |
| 꼭짓점 빈도 | (6,15) 6유형, 정확한 값 ∈ Q(√3) (예: A·R·R = 9√3 − 15) |
| 엔트로피 | 무작위 타일링 s∞ = 0.594 ± 0.004 k_B/입자, 3.12.12 섹터 ≈ 0.591 |
| 페이존 탄성 | K_α ≈ 1.75, K_r ≈ 1.70, K_i ≈ 1.82 (±0.15) — 등방성 판정은 의뢰 1 |
| 상도 | λ*에서 타일링 창 P* − 1.061T < P < P* + 0.547T (배치 엔트로피 0.594로 정정), 조성 고정선 −0.623T ~ +1.125T. figures/phase_diagram.png |
| 네마틱장 | 접촉 결합 네마틱 χ₂ ≈ 0.62, 유도 변형 α ≈ Ψ₂², h ≤ 1.5에서 페이존 연화 없음 |
| 결합 방식 | DDT 클러스터 전수 분류: 가로지름 집합 29 = 비결합 1 + 결합 6 + 꼭짓점 22(15.2%, 모두 접합점 비움·카이랄 짝). DDDT는 의뢰 2 |
| 양자 뒤집기 모형 | stoquastic 기저상태 A₁, 첫 들뜸 E₅(수직 공간 벡터), 부호 반전 시 B₂ |

전체 정리 문서: Claude Doc "HCSS λ* 타일링: 치환 대칭·프랙탈 창·엔트로피·양자 대칭 분류 — 작업 정리".

## 구성

```
TASK.md            의뢰 1: C++ 전달 행렬 생성기 + 페이존 강성 등방성 판정 → results01.md
TASK2.md           의뢰 2: 3.12.12 꼭짓점 환경(DDDT) 결합 방식 전수 분류 → results02.md
reference/         검증 기준 (수정 금지): Python 참조 구현, 기준 전달 행렬, expected.json
tools/             compare_tm.py (비트 비교), bin2npz.py (C++ 원시 출력 → npz)
tm_cpp/            C++ 생성기 자리 (비어 있음)
runs/              대형 계산 출력 (git 제외)
analysis/          이 연구의 계산 스크립트 스냅숏 (INDEX.md 참조)
figures/           그림 5장
```

## 빠른 시작

```
cd reference/python
python3 region_checks.py                                              # ALL OK
python3 cyl_ref.py 2 2 0 -1 /tmp/a.npz
python3 ../../tools/compare_tm.py ../data/ref_2_2_0_-1.npz /tmp/a.npz  # PASS
```
필요 패키지: Python 3, numpy, scipy (analysis/ 일부는 shapely, sympy, matplotlib).
