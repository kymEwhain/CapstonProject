# VII. DISCUSSION

실험 결과를 바탕으로 **링크 변동성, 분할 방식, 그리고 뉴런-디바이스 할당**과 관련된 논의를 다룸

<br>

### ❤️ A. **링크 변동성에서의 온디바이스 학습**

**무선 네트워크의 변동성** 때문에 기존의 분산 학습 방법들을 그대로 적용하기 어렵다. 특히, 무선 통신은 전력 소모에도 영향을 주기 때문에, **무선 엣지 환경**에서는 학습이 더 어려워짐.

⇒ **EdgePipe의 뉴런 기반 모델 분할과 파이프라인 가속**은 이런 변동성에도 **예측 성능을 유지**하는 데 효과적인 방법으로 검증됨

<br>

### ❤️ B. **Implication**

1. Vertical vs Horizontal
    - Vertical: 파이프라인 가속을 통해 짧은 학습 시간과 높은 정확도를 달성할 수 있다.
        - 하지만 한 디바이스가 전체 레이어를 처리하기 때문에 디바이스간 통신 손실이 발생하면 전체 학습이 중단될 위험이 크다.
    - Horizontal: 각 레이어를 여러 디바이스에 나누어 처리한다.
        - 이런 방식은 인접한 레이어간 통신이 잦아져, 통신 손실이 발생할 경우 전체 학습이 실패할 가능성이 매우 높아진다. 실제 실험에서도 수평 분할 방식은 매우 낮은 성능을 보임
        - 파이프라인 가속 효과를 거의 활용하지 못하기 때문에 학습 성능이 떨어진다(해당 레이어를 가지는 모든 디바이스가 작업을 완료한 후 그 결과를 통합해야만 다음 레이어로 넘어갈 수 있기 때문임). 따라서 한 디바이스에 연속적인 여러 레이어를 배분하는 방식은 무선 네트워크에서 적절하지 않다는 결론을 내림
    
2. Desirable Partitioning
    
    Vertical Allocation과 Horizontal Allocation을 결합한 Hybrid Allocation방식이 가장 안정적인 학습과 추론 성능을 제공한다. 
    
    - Hybrid Allocation은 파이프라인 가속과 레이어간 통신 감소 효과를 가진다.
    - 적어도 두 개의 디바이스가 하나의 레이어를 함께 처리하도록 함으로써, 안정성을 유지하면서도 속도와 효율성을 극대화할 수 있음
3. Neural Network-to-Device Allocation
무선 네트워크 환경에서 디바이스 간 연결성이 달라지므로, 적절한 디바이스에 모델을 배분하는 것이 매우 중요하다. 
    - 순전파와 역전파가 순서대로 실행되어야 하기 때문에 연결성이 강한 디바이스 쌍을 인접한 레이어에 할당하는 것이 중요하며 이를 통해 더 안정적이고 빠른 분산 학습을 구현할 수 있다.