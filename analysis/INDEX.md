# analysis/ — 스크립트 색인

채팅 샌드박스에서 쓴 스크립트의 스냅숏이다. 서로 중간 파일(pkl, npz)을 주고받으므로 아래 순서대로 같은 디렉터리에서 실행한다.
검증된 기준 구현은 `reference/python/`에 있으며, 여기 파일과 내용이 겹치면 그쪽이 우선이다.

| 순서 | 스크립트 | 하는 일 | 생성물 / 핵심 결과 |
|---|---|---|---|
| 0 | zz.py | Z[ζ12] 정수 산술, 별 사상 | — |
| 1 | check2.py 3 | 부풀린 삼각형·마름모의 정확 덮개 열거 | chk3.pkl, 20 / 23 |
| 2 | sym2.py | 채우기의 C₃ / C₂ 불변성 | fills.pkl, 불변 #6,#12 / #4,#15,#20 |
| 3 | subst.py, legal.py 4 | 등변 치환 6가지, 12-별 합법성, 꼭짓점 유형 | vtypes.pkl |
| 4 | vec.py, win2.py 6 15 5 | 벡터화 치환, 내부 공간 창 검정 | pts_*.npz |
| 5 | chaos.py 6 15 30 | 르베그 가중 카오스 게임, 창 경계 상자 차원 | — |
| 6 | overlap.py 6 15 300 | 겹침 알고리즘, 다각형 대리판 (비엄밀) | (6,4)의 8.8185는 인공물 |
| 6b | rig_overlap.py 6 15 | 겹침 알고리즘, 프랙탈 지지집합 엄밀판 (δ = 0.1830 불림, 상위 집합) | 두 치환 모두 ρ = 5 < λ² → 순수점 |
| 7 | freq.py 6 15 | 꼭짓점 유형 재귀(추출 사상) | 빈도는 정리 문서의 손 풀이 참조 |
| 8 | diffr.py | 변 곡선 자기 회피·차원, 회절 세기 기울기 | figures/bragg_vs_kperp.png |
| 9 | front.py | 검증된 영역 계수기 | 20 / 23 / 5827 |
| 10 | cyl.py, tm_save.py, chi.py, cusp.py | 원통 전달 행렬, 조성 감수율 | χ ∝ \|g\|² |
| 11 | cyl2.py, tilt.py, tiltscan.py, kfit.py | 기울기장, 강성 텐서 맞춤 | K_α, K_r, K_i |
| 12 | pairs.py, pair12.py, chains.py, chainsDT.py, clusters.py, bondmodel.py | 금지쌍, 결합 방식, 클러스터 계수 | D–D 2모드, D–T 1모드, DDT 15% |
| 12b | tmgen.py, spec.py | 타일 집합 일반화 원통 TM, 복소 간격비 통계(베테 가설 진단) | 사각형–삼각형: 푸아송형 0.655 / 우리 모형: 지니브르형 0.746 |
| 12c | cyl3.py, nematic.py, hstab.py | 접촉 결합 네마틱장: χ₂, 변형 결합, 장 속 페이존 강성 | χ₂ ≈ 0.62, α ≈ 1.1·Ψ₂², 연화 없음(h ≤ 1.5) |
| 12d | vmodes.py, vclass.py, vfig.py (vmodes3.py, vclass3.py) | DDT 클러스터의 골격 가로지름 집합 전수 분류 (DDDT는 진행 중) | 29가지 = 비결합 1 + 결합 6 + 꼭짓점 22, 꼭짓점 방식 15.2% |
| 13 | dodec_rep.py, moves.py, flipgraph.py, qflip.py, qflip2.py, qflip3.py | D₁₂ 분해, 뒤집기 그래프, 양자 대각화 | A₁ 260 …, E₅ 들뜸, B₂ |

주의: 첫 경계 절단 열거기(tenum.py)는 삼각형을 18개로 세는 버그가 있어 제외했다(분할 전에 교차 검사를 한 것이 원인).
| 12e | vmodes_fast.py, vclass3x.py, vstats3.py, vfig3.py | DDDT 전수 열거(7.50×10¹¹, 948집합), 독립성 검사(협동 14.42%), C₃v 분해 | results02.md |
