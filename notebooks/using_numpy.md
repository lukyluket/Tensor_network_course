# **Background Notes for `using_numpy.ipynb`**
*Using numpy — Session 1*

These notes provide the background needed to understand the material in the notebook **`first_class.ipynb`**. The goal is to connect basic quantum mechanics and linear algebra with the numerical tools used in the notebook, and to motivate why tensor networks are needed to describe quantum many-body systems efficiently.

This document is self-contained and written for **Master students in Quantum Technologies**.

---

# **1. Single Spin-1/2 Systems**

A single quantum two-level system (spin-1/2 or qubit) lives in a **two-dimensional Hilbert space**:

$$
\mathcal{H} \cong \mathbb{C}^2.
$$

We typically choose the computational basis

$$
|0\rangle =
\begin{pmatrix}1 \\
 0\end{pmatrix},
\\quad
|1\rangle =
\begin{pmatrix}0 \\
 1\end{pmatrix}.
$$

A general pure state is a column vector

$$
|\phi\rangle = c_0\,|0\rangle + c_1\,|1\rangle,
$$

with **complex amplitudes** $c_0, c_1 \in \mathbb{C}$.  
The state must be **normalized**:

$$
|c_0|^2 + |c_1|^2 = 1.
$$

In the notebook:

- Students sample random complex values for $c_0, c_1$.  
- Form a vector `phi`.  
- Normalize it with `phi / LA.norm(phi)`.

This ensures the state is valid.

---

# **2. Operators and the Pauli Basis**

An **observable** in quantum mechanics is represented by a **Hermitian operator** $O = O^\dagger$.  
On a single spin, any operator is a $2 	\times 2$ complex matrix.

## **2.1 Pauli matrices**

A convenient basis for all single-spin operators is the **Pauli basis**:

$$
\sigma_0 = I =
\begin{pmatrix}1&0 \\
 0&1\end{pmatrix},\\quad
\sigma_x =
\begin{pmatrix}0&1 \\ 
1&0\end{pmatrix},\\quad
\sigma_y =
\begin{pmatrix}0&-i \\
 i&0\end{pmatrix},\\quad
\sigma_z =
\begin{pmatrix}1&0 \\
 0&-1\end{pmatrix}.
$$

Any Hermitian operator on $\mathbb{C}^2$ can be written as

$$
O = c_0 \sigma_0 + c_1\sigma_x + c_2\sigma_y + c_3\sigma_z,
$$

with **real** coefficients $c_\mu$.

## **2.2 Extracting coefficients by trace identities**

The Pauli matrices are orthogonal under the Hilbert–Schmidt inner product:

$$
\text{Tr}(\sigma_\mu \sigma_\nu) = 2\,\delta_{\mu\nu}.
$$

Thus,

$$
c_\mu = \frac{1}{2} \mathrm{Tr}(\sigma_\mu O).
$$

The notebook uses this identity to reconstruct the coefficients from a given operator — a useful exercise in working with operator bases.

---

# **3. Expectation Values and Basis Changes**

Given a normalized state $|\phi\rangle$ and observable $O$, the expectation value is:

$$
\langle O \rangle_\phi = \langle \phi | O | \phi \rangle = \phi^\dagger O \phi.
$$

In NumPy:

```python
exp_O = phi.T.conj() @ operator @ phi
```

## **3.1 Changing basis**

Sometimes it is useful to change basis using a **unitary matrix** $U$.  
For example, $\sigma_x$ is diagonal in the basis

$$
|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt 2},
\\quad
|-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt 2}.
$$

Define the matrix with these eigenvectors as columns:

$$
U_x = \frac{1}{\sqrt 2}
\begin{pmatrix}
1 & 1 \\
1 & -1
\end{pmatrix}.
$$

The transformed state and operator are:

$$
|\phi_x\rangle = U_x^\dagger |\phi\rangle,
\\quad
O_x = U_x^\dagger O U_x.
$$

A key physical point:  
**All expectation values are basis-independent.**  

The notebook asks you to verify this computationally.

---

# **4. Two Spins and Tensor Products**

For two spins, the total Hilbert space is the **tensor product**:

$$
\mathcal{H}_{12} = \mathbb{C}^2 \otimes \mathbb{C}^2 \cong \mathbb{C}^4.
$$

The basis is:

$$
|00\rangle,\; |01\rangle,\; |10\rangle,\; |11\rangle.
$$

## **4.1 Product states**

Given single-spin states $$|\phi\rangle$$ and $$|\psi\rangle$$,

$$
|\Phi\rangle = |\phi\rangle \otimes |\psi\rangle.
$$

In NumPy:

```python
Phi = np.kron(phi, psi)
```

## **4.2 Operators on two spins**

Operators on multi-spin systems are built from tensor products:

$$
O_1 = O \otimes I, \\
O_2 = I \otimes O, \\
O_{12} = O \otimes O.
$$

These correspond to:

- an operator acting only on spin 1,  
- only on spin 2,  
- or a two-body operator acting on both.

Expectation values are computed in the same way as in the single-spin case.

---

# **5. Exponential Growth of Hilbert Space**

A crucial insight behind tensor networks:

> The Hilbert space of $N$ qubits has dimension $2^N$.

To store a **general** state vector you need $2^N$ complex numbers.  
In the notebook, you build 2-, 3-, 4-, …, $N$-spin states using repeated Kronecker products and record the size of the resulting array.

This growth is **exponential**, and even for moderate $N$, the number of parameters becomes enormous.

The notebook visualizes this both in:

- **memory cost** (`np.size(phi_n_phi)`),  
- **runtime cost** (timing expectation-value computations).

This motivates the need for **compressed representations** — eventually tensor networks.

---

# **6. Correlations, Product States, and the First Glimpse at Tensor Networks**

## **6.1 Connected correlation functions**

For operators $O_1$, $O_2$ acting on different sites, the **connected correlation** is:

$$
C_{12} = \langle O_1 O_2 \rangle - \langle O_1\rangle \langle O_2\rangle.
$$

- If $C_{12} \neq 0$, measurements are correlated.  
- If $C_{12} = 0$, outcomes are independent.

In the notebook, you compute these quantities for product states of three spins.

## **6.2 Why product states are simple**

For a product state

$$
|\psi\rangle = |\phi_1\rangle \otimes \cdots \otimes |\phi_N\rangle,
$$

any operator acting on disjoint subsets of spins factorizes:

$$
\langle O_1 O_2 \rangle
= \langle O_1 \rangle \langle O_2 \rangle.
$$

Thus **all connected correlations vanish**, and the state has:

- **no entanglement**,  
- **simple structure**,  
- **only $O(N)$ parameters**.

## **6.3 Product states as trivial tensor networks**

A product state corresponds to a tensor network where:

- each site has a local tensor (its state vector),  
- **no bonds** connect the tensors.

In tensor-network language, the **bond dimension** is 1.  
This is the simplest example of a compressed representation.

Later, Matrix Product States (MPS) extend this to include *limited* correlations and entanglement by introducing virtual bonds of small dimension.

---

# **7. Learning Goals of the Notebook**

After working through `first_class.ipynb`, students should be able to:

- Represent states and operators using NumPy arrays.  
- Normalize quantum states.  
- Verify Hermiticity numerically.  
- Expand operators in the Pauli basis.  
- Compute quantum expectation values.  
- Perform basis changes using unitary matrices.  
- Build multi-spin product states via tensor products.  
- Construct operators such as $O \otimes I$ and $I \otimes O$.  
- Observe exponential scaling of Hilbert-space dimension.  
- Understand why product states have no connected correlations.  
- Appreciate the need for efficient representations → **tensor networks**.

