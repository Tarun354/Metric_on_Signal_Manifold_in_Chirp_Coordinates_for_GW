# 2D Metric on Signal Manifold in Chirp Coordinates for GWs

This repository computes the **2D metric on the signal manifold** in dimensionless chirp coordinates using the **Fisher Information Matrix**, and verifies the metric approximation by comparing exact and metric-based matches.

---

## Waveform Model

The waveform is parameterized as:

$$
h(\vec{t}, \vec{\lambda}) = A(\vec{t}, \vec{\lambda}) e^{-i \phi(\vec{t}, \vec{\lambda})}
$$

where the parameters are:

- **Intrinsic parameters** (lambda_intr): component masses (m1, m2) and spin components (s1z, s2z, ...)
- **Extrinsic parameters** (lambda_extr): sky position, orientation (t0, phi0)

---

## Match Between Templates

The match between two neighboring templates in parameter space is defined as:

$$
M(\vec{\lambda}, \vec{\lambda} + \Delta \vec{\lambda}) 
= \max_{\Delta \vec{\lambda}_{\rm extr}} \langle \hat{h}(\vec{\lambda}), \hat{h}(\vec{\lambda} + \Delta \vec{\lambda}) \rangle
$$

where:

- $\hat{h} = h / ||h||$ is the normalized waveform  
- ⟨a, b⟩ = ∫₍f_low₎⁽f_high⁾ [(a*(f)b(f) + a(f)b*(f)) / S_h(f)] df  
is the PSD-weighted inner product.


---

## Approximate Match Expression

Using a Taylor expansion around $\Delta \vec{\lambda} = 0$ up to quadratic terms:

$$
M(\vec{\lambda}, \vec{\lambda} + \Delta \vec{\lambda}) 
= M|_{\Delta \vec{\lambda} = 0} 
{} + \Delta \vec{\lambda}^{\rm T} \frac{\partial M}{\partial \vec{\lambda}} \bigg|_{\Delta \vec{\lambda} = 0} 
{} + \frac{1}{2} \Delta \lambda_i \Delta \lambda_j \frac{\partial^2 M}{\partial \lambda_i \partial \lambda_j} \bigg|_{\Delta \vec{\lambda} = 0}
$$

With initial conditions:

- $M|_{\Delta \vec{\lambda} = 0} = 1$  
- $\frac{\partial M}{\partial \vec{\lambda}} \big|_{\Delta \vec{\lambda} = 0} = 0$

we get the **simplified match**:

$$
M(\vec{\lambda}, \vec{\lambda} + \Delta \vec{\lambda}) = 1 + \frac{1}{2} \frac{\partial^2 M}{\partial \lambda_i \partial \lambda_j} \Delta \lambda_i \Delta \lambda_j
$$

---

## Metric Definition

The **template-space metric** is defined as:

$$
M(\vec{\lambda}, \vec{\lambda} + \Delta \vec{\lambda}) = 1 - g_{ij} \Delta \lambda_i \Delta \lambda_j = 1 - \Delta s^2
$$

with

$$
g_{ij} = -\frac{1}{2} \frac{\partial^2 M}{\partial \lambda_i \partial \lambda_j} \Big|_{\Delta \vec{\lambda} = 0}
$$

---

## Fisher Information Matrix

The **Fisher matrix elements** are:

$$
\tilde{\Gamma}_{ab} = \langle \partial_a \hat{h}, \partial_b \hat{h} \rangle
$$

where $\partial_a$ denotes derivative with respect to parameter $a$.  

### Numerical Derivatives

Finite-difference approximation is used:

$$
f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} + O(h^2)
$$

### Fisher Matrix for Normalized Waveforms

For $\hat{h} = h / ||h||$:

$$
\tilde{\Gamma}_{ab} = \frac{1}{||h||^2} 
\Big[ \langle A e^{-i\phi}(-i)\partial_a \phi, A e^{-i\phi}(-i)\partial_b \phi \rangle 
{} + \langle \partial_a A, \partial_b A \rangle 
{} + \text{cross terms} \Big]
$$

Neglecting small cross terms, the **adjusted Fisher matrix** becomes:

$$
\tilde{\Gamma}_{ab} \approx \frac{1}{2 ||h||^2} \big[ \langle A \partial_a \phi, A \partial_b \phi \rangle + \langle \partial_a A, \partial_b A \rangle \big]
$$

---

## Parameter Space

### Intrinsic Parameters

We work in a **5-dimensional intrinsic parameter space**:

- **Chirp mass** $\mathcal{M}_c$  
- **Mass ratio** $\eta = \frac{m_1 m_2}{(m_1 + m_2)^2}$  
- **Mass asymmetry** $\delta = \frac{m_1 - m_2}{m_1 + m_2}$  
- **Reduced spin** $\chi = \frac{\chi_1 + \chi_2}{2}$ (assuming $\chi_1 = \chi_2$ for $m_1 \leq 3$, and $\chi_2 = 0$ otherwise). 

### Metric Projection

The **template-space metric** is obtained by projecting out extrinsic parameters:

$$
g = \Gamma_1 - \Gamma_2 \Gamma_3^{-1} \Gamma_4
$$

where $\Gamma_1, \Gamma_2, \Gamma_3, \Gamma_4$ are blocks of the Fisher matrix corresponding to intrinsic/extrinsic parameters.

---

**Author:** Tarun Kumar  
**Affiliation:** IIT Gandhinagar
