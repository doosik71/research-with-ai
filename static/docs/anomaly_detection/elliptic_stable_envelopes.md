# Elliptic stable envelopes

- **저자**: Mina Aganagic, Andrei Okounkov
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1604.00423

## 1. 논문 개요

본 논문의 목표는 Nakajima quiver varieties의 equivariant elliptic cohomology에서 stable envelopes를 구성하는 것이다. 이 연구는 이전 논문 [MO1]의 결과를 elliptic cohomology로 확장하며, 특히 rational curves의 enumerative K-theory에서 발생하는 q-difference equations의 monodromy 계산에 응용된다.

연구의 핵심 문제는 symplectic resolution上でtorus 작용에 대한 attracting cycles의 안정화이다. 안정화된 cycles는 원래의 비섭동 작용만을 참조하여 특성화되어야 하며, 이는 작은 섭동에 의존하지 않아야 한다. Stable envelopes는 정확히 이러한 목표를 달성한다.

연구의 중요성은 양자군(quantum groups)의 기하학적 표현론과 3차원 다양체 위의 sheaf에 대한 enumerative geometry에서의 응용에 있다. 특히 quantum cohomology와 K-theoretic Donaldson-Thomas theory에 직접적인 응용을 가진다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 K-theory에서의 stable envelopes를 elliptic cohomology로 올리는 것이다. 이 과정에서 새로운 수학적 구조가 나타나는데, slope parameter $\mathscr{L}$에 대한 piecewise constant dependence가 elliptic curve $E = \mathbb{C}^\times / q^\mathbb{Z}$ 위의 meromorphic dependence로 대체된다.

기존 접근 방식과의 주요 차별점은 다음과 같다. 첫째, cohomology 수준의 stable envelopes는 정상적인 극한을 통해 K-theory로 전환되지만, K-theory에서 elliptic cohomology로의 전환은 새로운 analytic structure를 야기한다. 둘째, K-theory에서 slope $\mathscr{L}$에 대한 불연속적인 dependence가 elliptic cohomology에서는 $z \in \text{Pic}(X) \otimes_\mathbb{Z} E$에 대한 유리형 함수(meromorphic function) dependence로 변환된다. 셋째, 이 변환을 통해 quantum difference equations의 pole subtraction 문제를 풀 수 있게 된다.

## 3. 상세 방법 설명

### 3.1 설정: Elliptic curve

$E = \mathbb{C}^\times / q^\mathbb{Z}$로 정의되는 elliptic curve를 사용한다. 이는 $0 < |q| < 1$인 punctured disc 위의 complex elliptic curves의 family이다.

Theta function은 다음과 같이 정의된다:

$$\vartheta(x) = (x^{1/2} - x^{-1/2}) \prod_{n>0}(1 - q^n x)(1 - q^n / x)$$

이 함수는 다음 성질을 만족한다:

$$\vartheta(q^k x) = (-1)^k q^{-k^2/2} x^{-k} \vartheta(x), \quad k \in \mathbb{Z}$$

### 3.2 Equivariant elliptic cohomology

Equivariant elliptic cohomology는 torus-equivariant elliptic cohomology를 $\mathbb{C}$ 위에서 정의한다. 주요 functor는 다음과 같이 주어진다:

$$\text{Ell}_{\mathsf{T}}: \{\mathsf{T}\text{-spaces } X\} \rightarrow \{\text{schemes}\}$$

특히 $\text{Ell}(\mathbb{C}^\times)^n(\text{pt}) \cong E^n$이고, $\text{Ell}_{\mathsf{T}}(\text{pt}) = \mathsf{T}/q^{\text{cochar}(\mathsf{T})} =: \mathscr{E}_{\mathsf{T}}$이다.

### 3.3 Stable envelopes의 정의

Attracting correspondence는 다음과 같이 정의된다:

$$\text{Attr} = \{(x,y), \lim_{a \to 0} a \cdot x = y\} \subset X \times X^{\mathsf{A}}$$

Stable envelope는 이 attracting set 위에서 $\text{Aut}(X)^\mathsf{A}$-invariant인 cycle의 개선된 버전이다. 구체적으로, 원래의 비섭동 작용만을 참조하여 특성화되며, small perturbation에 의존하지 않는다.

### 3.4 Elliptic stable envelopes의 핵심 특성

Elliptic stable envelopes는 $z \in \text{Pic}(X) \otimes_\mathbb{Z} E$에 대한 meromorphic dependence를 가진다. 이는 K-theory에서의 piecewise constant dependence를 대체한다. 극한 $q \to 1$ (또는 $-\Re \frac{\ln z}{\ln q} \to \mathscr{L}$)에서 piecewise analytic limit를 통해 K-theoretic stable envelopes를 재구성한다.

### 3.5 Pole subtraction과 monodromy

Quantum difference equations는 $K_{\mathsf{T}}(X)$-값 함수에 대한 선형 차분 방정식이다. 이 방정식들은 $z \mapsto q^{\mathscr{L}} z$ 형태의 shift를 가지며, Kähler 변수와 equivariant 변수에서 각각 regular singularities를 가진다.

pole subtraction matrix $\mathfrak{P}$는 $z$-solutions에서 $a$-solutions로의 transition matrix이다. Theorem 5에 따르면, 적절한 정규화 하에서 이 matrix는 elliptic stable envelopes로 주어진다.

### 3.6 R-matrices

Tensor product of Nakajima varieties의 경우, elliptic R-matrix는coproduct of monodromy와 관련된다. 이는 $\mathscr{U}_\hbar(\widehat{\mathfrak{g}})$-modules의 범주에서 braid group action을 정의한다.

## 4. 실험 및 결과

### 4.1 주요 결과 정리

이 논문의 주요 결과는 Theorem 3으로, Nakajima varieties에 대한 elliptic stable envelopes의 존재성과 구성을 제시한다. 이는 $\mathscr{U}_\hbar(\widehat{\mathfrak{g}})$-action을 elliptic quantum group으로 올린다.

### 4.2 구체적인 예시

$T^* \mathbb{P}(W)$와 $T^* \mathsf{Gr}(k,n)$ (cotangent bundle of Grassmannian)에 대한 구체적인 계산이 제공된다. These examples는 일반적인 theory의 응용을 보인다.

### 4.3 Monodromy 결과

Corollary 6.2는 qKZ 방정식의 monodromy가 elliptic R-matrix와 명시적 gauge transformation만큼 차이남을 보여준다. 이는 다음 commutative square로 요약된다:

vertices와 flop 사이의 monodromy가 stable envelopes를 통해 연결된다.

## 5. 강점, 한계

### 5.1 강점

이 논문의 강점은 다음과 같다. 강력한 수학적 엄밀성을 가지고 있으며, 고전적인 결과를 새로운 수준으로 확장한다. Nakajima varieties라는 가장 크고 풍성한 family of equivariant symplectic resolutions을 다룬다. 양자군 표현론, enumerative geometry, 수리물리학 사이의 깊은 연결을 제공한다. 3d 초대칭 gauge theories에서의 dualities와 직접적으로 연결된다.

### 5.2 한계 및 미해결 질문

이 연구의 한계와 향후 연구 방향은 다음과 같다. 현재는 algebraic setting에서만 작동하며, 더 일반적인 topological setting으로의 확장이 필요하다. Categorification (양자군 작용을 범주화) 문제는 companion paper에서 다뤄진다. 비가환적 deformation의 상황에서의 분석은 아직 완전히 개발되지 않았다.

## 6. 결론

본 논문의 주요 기여는 Nakajima quiver varieties의 equivariant elliptic cohomology에서 stable envelopes를 구성한 것이다. 이 구성은 다음과 같은 중요한 결과를 이끈다: $\mathscr{U}_\hbar(\widehat{\mathfrak{g}})$-action의 elliptic quantum group으로의 확장, quantum difference equations의 monodromy에 대한 명시적 계산, 3d mirror symmetry와 symplectic duality 사이의 대응의 정식화.

향후 연구 방향으로, 3차원 초대칭 gauge theories에서의 경계 조건에 대한 correspondence, knot theory와 categorification으로의 응용, 그리고 6d "little string theory"와의 연결이 있다. 이 연구는 수리물리학에서 중요한 역할을 하며, 특히 양자군, 초대칭 gauge theory, enumerative geometry의 교차점에서 새로운 통찰을 제공한다.
