"""조정 계층(Supervisor). 하위 에이전트(agents/)는 이 패키지를 import하지 않는다.

- policy.py : State 제어 필드 → 다음 실행 노드 결정(순수 함수)
- router.py : supervisor 노드(결정 + 결정 로그) 와 단일 conditional edge 라우터
- guard.py  : 에이전트 예외를 node_status=failed / last_error로 바꾸는 래퍼
"""
