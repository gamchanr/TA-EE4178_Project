# EE 4178 (2019 Fall) - 기말 프로젝트

## Final Project
1. [프로젝트개요](https://drive.google.com/open?id=1VYOuNUQQynr9hX2WcEqzAGGCBl5vukRH)
2. 데이터셋 - [[train](https://drive.google.com/open?id=1Gx-1Gj3YLR7r4kYIMDJMnF1GtKYPMvbQ)] / [[validation](https://drive.google.com/open?id=1T8KSOgAVpKsJFWgNMeVfLgTnKQSp1VeB)] / [[test](https://drive.google.com/open?id=1b5-v3h-EIqO00vel7MRSTucGp856BymK)]
3. [데이터 로드를 위한 참고 코드 (font_dataset.py)](https://github.com/gamchanr/TA-EE4178/blob/master/utils/font_dataset.py)

## 프로젝트 추가공지 (GPU환경 확인 및 실행 시간 출력)
* Colab에서 제공되는 GPU종류가 총 5종류(NVIDIA Tesla K80, P100, P4, T4, V100 GPU)로 확장되었고, 임의로 지정할 수 없으며 세션이 연결될 시 랜덤으로 배정되는 것으로  확인됩니다. 이에 따라 GPU 동작시간에 차이가 발생할 수 있습니다.
* 현재 할당된 GPU가 아닌 다른 GPU를 할당받고 싶으신 경우, 세션을 끊었다가 다시 연결하면 변경되는 경우가 있습니다. (배정되는 GPU를 그냥 사용하실 분은 그대로 진행하 시면 됩니다.)
* <u>**train.ipynb 제출 시 아래 두 코드도 추가하여, 보고서에 '어떤 환경에서 소요시간이 얼마인지(ex. Tesla T4 환경에서 2.003초)' 함께 기재해 주시기 바랍니다.**</u>
1. GPU 환경 확인 (Colab GPU: NVIDIA Tesla K80, P100, P4, T4, V100 GPU 중 실행됨): `!nvidia-smi`
<img src="./assets/pj_gpu.png" alt="Drawing" style="width: 800px;" align="left"/>  <br>
2. 코드 실행 시간 출력
   ```
   import time

   start_time = time.time()

   ### 학습코드 예시
   time.sleep(2)
   ###

   duration = time.time() - start_time
   print(duration) # 2.003 (초)가 출력됨
   ```  
