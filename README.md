# [FOSCAR] winter project1
영상에서 특정 색상을 검출한 뒤, 검출된 위치로 터틀심 거북이를 이동해보는 프로젝트


## 프로젝트 소개
- 웹캠을 통해 이미지를 받아온다.
- 이미지를 9등분으로 나누고 특정 색상(green)을 검출한 뒤, 그 위치 좌표를 추출한다.
- 현재 거북이가 있는 좌표에서 특정 색상이 존재하는 좌표로 이동시킨다. (publish)
  - angular 속도 설정
    - atan2(목표 좌표 - 현재좌표) 값보다 현재 theta가 크면 1
    - 현재 theta보다 작으면 -1
  - linear 속도는 3으로 고정

## 실행 방법
`roslaunch foscar_winter_project1 project1.launch`


## RQT_GRAPH
![rosgraph](https://user-images.githubusercontent.com/45509381/108341240-cf72f380-721c-11eb-8c3a-2b1873ac81f4.png)

## DEMO
[![[FOSCAR] winter project1](https://img.youtube.com/vi/SaA8JoiG_XY/0.jpg)](https://youtu.be/SaA8JoiG_XY)
