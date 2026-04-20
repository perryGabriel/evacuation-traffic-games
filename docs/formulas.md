# Formulas

For schedule `S` over horizon `T`, service tokens are `r=(i,b,t)`.

- Route token count for generator `k`:
  - `N_k(S) = number of tokens (i,b_i(k),t) selected on route P_k`.
- Surrogate welfare:
  - `W(S) = sum_k alpha_k * phi_k(N_k(S))`.
- Intersection utility:
  - `u_i(S) = W(S) - W(S_{-i})`.

Supported concave families:

- capped: `phi_k(n)=min(n,D_k)`
- geometric: `phi_k(n)=sum_{m=1}^n beta_k^(m-1)`

Important: the revised theorem applies to this surrogate welfare construction, **not** the original max-delay objective from the operational model.
