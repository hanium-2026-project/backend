"""실차 계측·기록 도구 모음.

패키지로 두는 이유는 하나다: 이게 없으면 ``unittest discover`` 가 tools/ 를
아예 건너뛴다. test_run_recorder / test_analyze_vehicle_dynamics /
test_calibrate_speed_wiring 이 전부 전체 suite 밖에 있었다.
"""
