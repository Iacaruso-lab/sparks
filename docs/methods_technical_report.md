# SPARKS — Model and Controls: A Technical Description of the Methods

## Scope

This document provides a complete technical description of the SPARKS architecture, its training
objective, and the alternative attention mechanisms used as controls, as currently implemented.
It is written to serve as the source material for a Methods section, and it therefore follows the
notation and organisation of the corresponding sections of Skatchkovsky, Glazman, Egea-Weiss,
Sadeh and Iacaruso, *A Biologically Inspired Attention Model for Neural Signal Analysis*
(hereafter, the original description).

Several components of the implementation have been extended or superseded since the original
description was written. Rather than annotating these individually in the running text, each is
introduced in its natural place with an explicit statement of what it replaces, and Section 10
summarises the full set of differences in tabular form. Equation numbering is internal to this
document and does not correspond to that of the original manuscript. Reference numbers in the
ranges 26–68 are those of the original bibliography; sources introduced here are marked as such
in Section 12.

**Notation.** Throughout, $N$ denotes the number of simultaneously recorded neurons, $T$ the
number of time-steps in a sequence, and $B$ the number of sequences in a mini-batch. Neural
activity is written $s_{i,t}$ for neuron $i$ at time $t$, taking values in $\{0,1\}$ for
electrophysiological recordings and in $\mathbb{R}$ for fluorescence signals. The Hebbian
attention layer produces a coefficient matrix $\boldsymbol{A}_t \in \mathbb{R}^{N \times N}$ with
entries $a_{ij,t}$, which is projected onto an embedding of dimension $d_v$ and compressed onto
$M$ latent slots; the resulting representation is mapped onto a latent space of dimension $d$.
The eligibility-trace time constant is $\tau_s$, the sampling period is $dt$, the decoder reads a
window of $\tau_p$ past latent samples and predicts $\tau_f$ future samples of the reference
signal, and $K$ denotes the number of Hebbian attention heads. We write $\mathrm{sp}(\cdot)$ for
the softplus function and $\mathrm{sigm}(\cdot)$ for the logistic sigmoid, reserving
$\sigma(\cdot)$ for the standard deviation of the latent distribution as in the original
description. Tensors are laid out with time along the second axis, so that a sequence of
coefficient matrices has shape $B \times T \times N \times N$.

---

## 1. Predictive causally conditioned distributions

For any two jointly distributed random vectors $\boldsymbol{a} = (\boldsymbol{a}_1, \dots,
\boldsymbol{a}_T)$ and $\boldsymbol{b} = (\boldsymbol{b}_1, \dots, \boldsymbol{b}_T)$ from time
$1$ to $T$, at every time-instant $t$ we define the *predictive causally conditioned* (PCC)
distribution of $\boldsymbol{a}$ given $\boldsymbol{b}$ as

$$
p^{(\tau_1,\tau_2)}(\boldsymbol{a}_t \,\|\, \boldsymbol{b}_t) \;=\; p\big(\boldsymbol{a}_t^{\,t+\tau_2-1} \mid \boldsymbol{a}^{t-1},\, \boldsymbol{b}_{t-\tau_1+1}^{\,t}\big).
\tag{1}
$$

This distribution captures the causal dependence, in the sense of the directed information flow
in time, of the $\tau_2$ future samples of $\boldsymbol{a}_t^{\,t+\tau_2}$ on the $\tau_1$ past
samples of $\boldsymbol{b}_{t-\tau_1}^{\,t}$. Setting $\tau_1 = T$ and $\tau_2 = 1$ recovers the
causally conditioned distribution of Kramer (27).

## 2. Problem definition

We consider the autoencoder architecture of Figure 1A, whose purpose is to train an encoding
network to produce a representation $\boldsymbol{z}$ of a neural input $\boldsymbol{x}$ that is
informative about a reference signal $\boldsymbol{r}$. The input $\boldsymbol{x}$ is a neural
signal obtained by electrophysiology or by optical imaging. The reference signal $\boldsymbol{r}$
represents the corresponding target, which may be a label describing a behaviour of interest, a
set of continuous behavioural variables, a sensory stimulus, or the input itself in the
unsupervised case.

Encoder and decoder are optimised jointly on a data set of pairs $(\boldsymbol{x},
\boldsymbol{r})$ drawn from an unknown population distribution $p(\boldsymbol{x},
\boldsymbol{r})$. Inputs take the form of a discrete collection of vectors over time
$\boldsymbol{x} = (\boldsymbol{x}_1, \dots, \boldsymbol{x}_T)$ with $\boldsymbol{x}_t \in
\mathbb{R}^{N}$, and reference signals $\boldsymbol{r} = (\boldsymbol{r}_1, \dots,
\boldsymbol{r}_T)$ with $\boldsymbol{r}_t \in \mathbb{R}^{N_R}$ define general target signals;
all signals are assumed to be uniformly sampled at a fixed rate $dt$.

Following ideas from predictive coding, we maximise the causally conditioned distribution of
$\tau_f$ future samples of the reference signal given past input activity, averaged over
time-steps,

$$
\frac{1}{T}\sum_{t=1}^{T} \log p\big(\boldsymbol{r}_t^{\,t+\tau_f-1} \,\|\, \boldsymbol{x}_t\big)
\;=\; \frac{1}{T}\sum_{t=1}^{T} \log \mathbb{E}_{\boldsymbol{z}^t}\, p\big(\boldsymbol{r}_t^{\,t+\tau_f-1} \mid \boldsymbol{z}^t, \boldsymbol{x}^t, \boldsymbol{r}^{t-1}\big).
\tag{2}
$$

Objective (2) is intractable for problems of practical size (63), and is optimised through the
variational approximation developed in Section 6.

A single model may be trained jointly on recordings from several sessions or animals, each with
its own number of recorded neurons. In this case the problem above is instantiated once per
session, with a session-specific set of Hebbian attention parameters and a parameter vector that
is otherwise shared across sessions (Section 4.7).

## 3. Encoding network

The encoder implements a causal stochastic mapping between the input $\boldsymbol{x}$ and the
latent signal $\boldsymbol{z}$, parameterised by a vector $\boldsymbol{w}^e$,

$$
p_{\boldsymbol{w}^e}(\boldsymbol{z} \,\|\, \boldsymbol{x}) \;=\; \prod_{t=1}^{T} p_{\boldsymbol{w}^e}\big(\boldsymbol{z}_t \,\|\, \boldsymbol{x}^t\big),
\tag{3}
$$

where each factor is obtained through the reparameterisation trick (67) as

$$
p_{\boldsymbol{w}^e}\big(\boldsymbol{z}_t \,\|\, \boldsymbol{x}^t\big) \;\sim\; \mathcal{N}\Big(\mu\big(f_{\boldsymbol{w}^e}(\boldsymbol{x}^t)\big),\; \sigma^2\big(f_{\boldsymbol{w}^e}(\boldsymbol{x}^t)\big)\Big),
\tag{4}
$$

with $f_{\boldsymbol{w}^e}(\cdot)$ the output of the encoding network, so that samples are drawn
as $\boldsymbol{z}_t = \mu(f_{\boldsymbol{w}^e}(\boldsymbol{x}^t)) +
\sigma(f_{\boldsymbol{w}^e}(\boldsymbol{x}^t)) \odot \boldsymbol{\epsilon}$ with
$\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$.

The encoding network $f_{\boldsymbol{w}^e}$ comprises, in order: a session-specific Hebbian
attention block (Section 4), which maps the $N$-dimensional input at each time-step onto a
representation of dimension $M d_v$; zero or more session-shared conventional attention blocks
operating on that representation; and a final layer normalisation. In the original description,
the population read-out consisted of flattening the output of the last attention block and
projecting it linearly onto the latent dimension. Here, this role is taken by a latent
cross-attention bottleneck internal to the Hebbian attention block (Section 4.5), which
compresses the $N$ neuron-indexed embeddings onto a fixed number $M$ of latent slots irrespective
of $N$. A single downstream architecture can consequently serve recordings with differing neuron
counts without any change of shape, which is what makes the multi-session and session-transfer
procedures of Section 4.7 possible.

The optional conventional blocks implement causal dot-product self-attention as in (26), with two
modifications relative to the original description: positions are encoded by rotary position
embeddings applied to the queries and keys (new ref. 69) in place of additive positional
encodings, and each residual branch is scaled by a learnable per-channel gain initialised near
zero (new ref. 70), so that a deep stack begins close to the identity mapping and the
contribution of each block grows only as optimisation finds it useful. The default configuration
uses no conventional blocks, in which case the Hebbian attention block alone constitutes
$f_{\boldsymbol{w}^e}$.

The functions $\mu(\cdot)$ and $\sigma(\cdot)$ are each implemented as a single linear layer
reading off $f_{\boldsymbol{w}^e}(\boldsymbol{x}^t)$. The log-variance is constrained to the
interval $[-4, 2]$. This constraint bounds both the magnitude of the injected reparameterisation
noise and the argument of the exponential appearing in the Kullback–Leibler term of Section 6,
neither of which is otherwise controlled when the weight $\beta$ of that term is small; it
therefore decouples the numerical stability of the encoder from the tuning of $\beta$. A layer
normalisation is applied immediately before these two heads, because neither the Hebbian
attention block nor the conventional blocks normalise their own output, and the scale reaching
the projection heads would otherwise depend on the accumulated scale of the residual stream.

## 4. Hebbian attention

### 4.1 Coefficients from spike-timing-dependent plasticity

We apply online spike-timing-dependent plasticity (STDP) at every time-step $t$ to compute a
matrix $\boldsymbol{A}_t \in \mathbb{R}^{N \times N}$ in which each element $a_{ij,t}$ defines a
coefficient between neurons $i$ and $j$. Coefficients for each ordered pair $(i \rightarrow
j)_{1 \le i,j \le N}$ are obtained by maintaining over time a pair of pre- and post-synaptic
*eligibility traces* $a^{\mathrm{pre}}_{ij,t}$ and $a^{\mathrm{post}}_{ij,t}$. When the neuron on
either side of the pair fires, the corresponding trace is incremented by an amount determined by
weights $w^{\mathrm{pre}}_{ij}$ and $w^{\mathrm{post}}_{ij}$, and in the absence of a spike each
trace decays exponentially at a rate determined by $\gamma^{\mathrm{pre}}_{ij}$ and
$\gamma^{\mathrm{post}}_{ij}$. The coefficient $a_{ij,t}$ is potentiated, respectively depressed,
by the value of the pre-, respectively post-, synaptic trace when the post-, respectively pre-,
synaptic neuron fires.

At each time-step the three quantities are updated in the following order. The traces first decay,

$$
\bar{a}^{\mathrm{pre}}_{ij,t} = \gamma^{\mathrm{pre}}_{ij}\, a^{\mathrm{pre}}_{ij,t-1},
\qquad
\bar{a}^{\mathrm{post}}_{ij,t} = \gamma^{\mathrm{post}}_{ij}\, a^{\mathrm{post}}_{ij,t-1};
\tag{5}
$$

the coefficient is then updated using these decayed values,

$$
a_{ij,t} = \mathrm{clip}\Big(a_{ij,t-1} + \big(1 - a_{ij,t-1}\big)\,\bar{a}^{\mathrm{pre}}_{ij,t}\, s_{j,t} \;-\; a_{ij,t-1}\, \bar{a}^{\mathrm{post}}_{ij,t}\, s_{i,t},\;\; -0.5,\; 1.5\Big);
\tag{6}
$$

and only afterwards is the current time-step's activity incorporated into the traces,

$$
a^{\mathrm{pre}}_{ij,t} = \bar{a}^{\mathrm{pre}}_{ij,t} + w^{\mathrm{pre}}_{ij}\, s_{i,t},
\qquad
a^{\mathrm{post}}_{ij,t} = \bar{a}^{\mathrm{post}}_{ij,t} + w^{\mathrm{post}}_{ij}\, s_{j,t}.
\tag{7}
$$

The ordering of (5)–(7) is what enforces the causal asymmetry of the rule. Because the update in
(6) reads the traces before the current spikes have been added to them, potentiation requires
pre-synaptic activity that is strictly *prior* to a post-synaptic spike, and depression requires
the converse; two neurons firing in the same bin therefore produce no potentiation. This
asymmetry is the property that distinguishes the mechanism from generic co-activation measures,
and it is the variable isolated by the symmetric control of Section 7.1.

The multiplicative factors $(1 - a_{ij,t-1})$ and $a_{ij,t-1}$ in (6) implement soft bounds that
restrict coefficients to approximately the interval $[0,1]$, and were introduced in the original
description to prevent coefficients from diverging for neurons firing at high rates during long
recordings. The present implementation additionally imposes the hard bound
$\mathrm{clip}(\cdot, -0.5, 1.5)$, which was found necessary for the recordings of longest
duration, where the soft bounds alone permit slow drift beyond the intended range. The resulting
saturation is reminiscent of the bounded dynamic range of biological synaptic efficacy, and is
the second variable isolated by the control of Section 7.1.

### 4.2 Parameterisation and parameter sharing

All decay and weight parameters are strictly positive by construction, being obtained from
unconstrained real-valued parameters through a softplus transformation,

$$
\gamma^{\mathrm{pre}}_{ij} = \exp\!\Big(-\mathrm{sp}\big(\delta^{\mathrm{pre}}_{ij}\big)\,\frac{dt}{\tau_s}\Big),
\qquad
w^{\mathrm{pre}}_{ij} = \mathrm{sp}\big(\theta^{\mathrm{pre}}_{ij}\big)\, w_{+},
\qquad
w^{\mathrm{post}}_{ij} = \mathrm{sp}\big(\theta^{\mathrm{post}}_{ij}\big)\, w_{+}\, \alpha,
\tag{8}
$$

and analogously for $\gamma^{\mathrm{post}}_{ij}$, where $w_{+}$ and $\alpha$ are fixed
hyperparameters. The unconstrained parameters are initialised at $\mathrm{sp}^{-1}(1) = \log(e -
1)$, so that decay rates begin at $\exp(-dt/\tau_s)$ and the softplus operates in its
approximately linear regime, where gradients are well conditioned. The factor $\alpha > 1$ biases
the post-synaptic increment above the pre-synaptic one, breaking the symmetry between the
potentiation and depression pathways at initialisation.

Two configurations of the layer differ in how these parameters are shared across pairs. In the
*full* configuration, $\delta^{\mathrm{pre}}_{ij}$, $\delta^{\mathrm{post}}_{ij}$,
$\theta^{\mathrm{pre}}_{ij}$ and $\theta^{\mathrm{post}}_{ij}$ are learned independently for every
ordered pair, as in the original description, amounting to $4N^2$ learnable parameters in the
attention layer. In the *light* configuration, which is the default in the present
implementation, these quantities are instead held fixed and shared across the population, so that
$\gamma^{\mathrm{pre}} = \gamma^{\mathrm{post}} = \exp(-dt/\tau_s)$, $w^{\mathrm{pre}} = w_{+}$
and $w^{\mathrm{post}} = w_{+}\alpha$, and the only learnable parameters of the attention layer
are those of the value projection of Section 4.3. Two consequences follow. First, the eligibility
traces need only be maintained per neuron rather than per pair, reducing the recurrent state from
$O(N^2)$ to $O(N)$ until the coefficient matrix itself is formed at the combination step (6).
Second, the interpretation of the coefficients changes: in the *light* configuration the
coefficient matrix is a fixed, parameter-free functional of the spike trains, and any structure
recovered from it reflects the recording rather than the fitted model.

### 4.3 Value projection

The coefficient matrix is mapped onto a per-neuron embedding by an affine transformation applied
along the post-synaptic index,

$$
\mathrm{HebbianAttention}(\boldsymbol{A}_t, V) = \boldsymbol{A}_t V + \boldsymbol{b}_v,
\qquad
V \in \mathbb{R}^{N \times d_v},
\tag{9}
$$

so that the embedding of neuron $i$ at time $t$ is obtained from the $i$-th row of
$\boldsymbol{A}_t$, that is, from the coefficients relating neuron $i$ to every other neuron. The
matrix $V$ is defined in analogy to the values of conventional dot-product attention (26) and is
shared across all neurons: the transformation (9) carries no dependence on the identity of the
neuron $i$ whose row it is applied to. Neuron identity is instead introduced downstream, by the
additive embedding of Section 4.5, which separates the representation of *which* neuron a row
belongs to from the transformation applied to its coefficients.

This projection was selected over higher-capacity alternatives, including a gated variant with a
learnable per-neuron feature table and a normalised value pathway. The affine map of (9) matched
or exceeded these alternatives on the tasks considered, and is used throughout.

### 4.4 Feedforward sublayer

The embedding produced by (9) is passed through a position-wise feedforward sublayer with a
residual connection,

$$
\boldsymbol{h}_{i,t} \;\leftarrow\; \boldsymbol{h}_{i,t} + \mathcal{D}\Big(W_3\big[\mathrm{SiLU}(W_1 \tilde{\boldsymbol{h}}_{i,t}) \odot W_2 \tilde{\boldsymbol{h}}_{i,t}\big]\Big),
\qquad
\tilde{\boldsymbol{h}}_{i,t} = \mathrm{RMSNorm}(\boldsymbol{h}_{i,t}),
\tag{10}
$$

where the gated linear unit of (10) has inner width $\tfrac{8}{3}d_v$ and no biases (new ref. 71),
and $\mathcal{D}(\cdot)$ denotes stochastic depth (new ref. 72), which independently drops the
entire residual branch for each sequence in a mini-batch during training and rescales the
surviving branches so that the expectation is unchanged. This replaces the fully connected
sublayer with batch normalisation of the original description; normalisation is applied to the
input of the sublayer rather than to its output, so that the residual path remains an exact
identity mapping.

### 4.5 Latent bottleneck

The $N$ neuron-indexed embeddings are compressed onto $M \ll N$ latent slots by a cross-attention
module in the manner of the Perceiver (new ref. 73). A learnable embedding $\boldsymbol{e}_i \in
\mathbb{R}^{d_v}$, shared across time and specific to neuron $i$, is added before normalisation,

$$
\tilde{\boldsymbol{h}}_{i,t} = \mathrm{LayerNorm}\big(\boldsymbol{h}_{i,t} + \boldsymbol{e}_i\big),
\tag{11}
$$

and $M$ learnable query vectors, which do not depend on the input and are reused at every
time-step and for every session, read out the population. Two read-out mechanisms are available.

The first is multi-head softmax cross-attention with the $M$ queries attending over the $N$
neuron embeddings, computed independently at each time-step; the compression is then a stateless
function of the population activity at that instant.

The second, which is the default, replaces the softmax with a kernelised linear attention that
carries a decaying recurrent state across time (new ref. 74). With the feature map $\phi(u) =
\mathrm{elu}(u) + 1$, which guarantees positivity, applied to per-head projections
$\boldsymbol{k}_{i,t}$ and $\boldsymbol{v}_{i,t}$ of $\tilde{\boldsymbol{h}}_{i,t}$, the
per-time-step contributions of the population are

$$
\boldsymbol{\kappa}_t = \sum_{i=1}^{N} \phi(\boldsymbol{k}_{i,t}) \otimes \boldsymbol{v}_{i,t},
\qquad
\boldsymbol{\zeta}_t = \sum_{i=1}^{N} \phi(\boldsymbol{k}_{i,t}),
\tag{12}
$$

which are accumulated into a state that decays at a learnable, head-specific rate $\lambda_h =
\mathrm{sigm}(\ell_h) \in (0,1)$,

$$
\boldsymbol{S}^{\kappa}_t = \lambda_h \boldsymbol{S}^{\kappa}_{t-1} + \boldsymbol{\kappa}_t,
\qquad
\boldsymbol{S}^{\zeta}_t = \lambda_h \boldsymbol{S}^{\zeta}_{t-1} + \boldsymbol{\zeta}_t,
\tag{13}
$$

and read out by the $M$ queries as

$$
\boldsymbol{o}_{m,t} = \frac{\phi(\boldsymbol{q}_m)^{\!\top} \boldsymbol{S}^{\kappa}_t}{\phi(\boldsymbol{q}_m)^{\!\top} \boldsymbol{S}^{\zeta}_t + \varepsilon},
\qquad m = 1, \dots, M.
\tag{14}
$$

The rates $\lambda_h$ are initialised at $\mathrm{sigm}(0) = 1/2$, a neutral value from which
optimisation may sharpen the state in either direction. The motivation for this variant is
structural: with the softmax read-out, the compression step is the only stateless operation
between two recurrent stages, namely the eligibility traces upstream and the decoder's latent
window downstream, and the population read-out is therefore forced to re-derive its summary of
the population from scratch at each instant. Equations (13)–(14) instead endow it with a decaying
memory of the same form as the eligibility traces themselves.

Independently of which read-out is used, the $M$ resulting slots subsequently attend to one
another through multi-head self-attention, so that the compressed summaries are not independent,
and pass through a further feedforward sublayer of the form (10). Both operations are applied
with pre-normalised residual connections. The $M$ slots are finally concatenated into a vector of
dimension $M d_v$, which is the quantity denoted $f_{\boldsymbol{w}^e}(\boldsymbol{x}^t)$ in
Section 3.

### 4.6 Multiple timescales

The mechanism above may be replicated to obtain several parameter sets
$\big(T^{\mathrm{pre},(k)}, T^{\mathrm{post},(k)}, W^{\mathrm{pre},(k)}, W^{\mathrm{post},(k)},
V^{(k)}\big)$ for $k = 1, \dots, K$, each with its own trace time constant $\tau_s^{(k)}$ and
therefore its own eligibility dynamics and value projection. The resulting multi-headed module
allows the model to attend jointly to information from different representation subspaces (26),
and here specifically to correlation structure at several eligibility timescales simultaneously
rather than committing to a single $\tau_s$. Head outputs are concatenated and projected back
onto $d_v$ by a learnable matrix $W_o \in \mathbb{R}^{K d_v \times d_v}$,

$$
\mathrm{MultiHeadHebbianAttention}_t = W_o^{\!\top}\,\mathrm{Concat}\Big(\mathrm{HebbianAttention}\big(\boldsymbol{A}^{(1)}_t, V^{(1)}\big), \dots, \mathrm{HebbianAttention}\big(\boldsymbol{A}^{(K)}_t, V^{(K)}\big)\Big).
\tag{15}
$$

With $K = 1$ this reduces to the single-head mechanism of Sections 4.1–4.3.

### 4.7 Session-specific attention and transfer

When a model is trained on recordings from several sessions or animals, a separate Hebbian
attention block, sized to that session's neuron count, is instantiated for each session, while
the feedforward sublayer parameters of the shared stack, any conventional attention blocks, and
the projection heads $\mu(\cdot)$ and $\sigma(\cdot)$ are common to all sessions. Because the
latent bottleneck of Section 4.5 emits a representation of fixed dimension $M d_v$ regardless of
$N$, no shared component depends on the number of neurons in any individual session. A previously
unseen session may accordingly be introduced into a trained model by instantiating an additional
Hebbian attention block for it and fine-tuning, leaving the shared components and the blocks of
all other sessions structurally unchanged.

### 4.8 Exact evaluation of the recurrences

Every recurrent quantity introduced above — the eligibility traces (5) and (7), the recurrent
state (13) of the linear-attention read-out, the calcium traces of Section 5, and the traces of
the control mechanisms of Section 7 — satisfies an affine recurrence of the form $x_t = a_t
x_{t-1} + b_t$ in which $a_t$ and $b_t$ are known in advance of the recursion. The composition of
affine maps being associative, such a recurrence admits an exact closed-form solution computable
by a parallel prefix scan (new ref. 75; see also new refs. 76, 77) in $O(\log_2 T)$ sequential
rounds of elementwise operations over the full sequence, in place of $T$ strictly sequential
steps. The initial condition $x_{-1}$ enters this construction exactly, through the first
time-step's coefficient $b_0$, which is what permits the recurrences to be resumed from a stored
state at an arbitrary point in a recording (Section 6.6).

The coefficient update (6) is the sole exception. Its hard bound is a genuine per-step
nonlinearity, and associativity fails; no closed form of the above type is available. This
recurrence is therefore evaluated sequentially, but with $a_t$ and $b_t$ precomputed for all $t$
so that each step reduces to a single fused multiply–add–clip over the batch and pair dimensions,
rather than a full trace-decay-and-update. The two evaluation strategies are numerically
equivalent reimplementations of the same recurrence, agreeing to floating-point precision; the
choice between them, which is made on the basis of $N$, is a compute-and-memory trade-off and not
an approximation. For small $N$ the per-step arithmetic is slight and dispatch latency dominates,
favouring the parallel form; for large $N$ the additional $O(T \log T)$ total work of the
parallel form outweighs the reduction in sequential depth. The crossover was measured on a single
GPU at a fixed batch size and embedding width and is exposed as a configurable threshold, as it
depends on the hardware and on the problem dimensions.

## 5. Calcium imaging data

The original description proposes, for calcium recordings, to use the fluorescence traces of each
neuron directly as eligibility traces, thereby dispensing with spike deconvolution and its
associated processing. Writing $\boldsymbol{x}$ for the fluorescence signal, this amounts to
setting $a^{\mathrm{pre}}_{ij,t} := x_{i,t}$ and $a^{\mathrm{post}}_{ij,t} := x_{j,t}$, with
coefficients accumulated as $a_{ij,t} := a_{ij,t-1} + a^{\mathrm{post}}_{ij,t} -
a^{\mathrm{pre}}_{ij,t}$. The justification is twofold. Since an eligibility trace is
fundamentally a decaying memory of a past event, a raw calcium trace may be regarded as a
biologically generated eligibility trace, whose relative timing across pairs of neurons provides
a robust, if temporally smoothed, proxy for the spike-timing relationships the mechanism is built
to capture; and correlated fluctuations in calcium activity are known to reflect underlying
functional connectivity even on comparatively slow timescales.

The present implementation refines this construction in two respects, motivated by the
observation that a single raw trace conflates fluorescence dynamics occurring on very different
timescales, and that the unbounded accumulation above is numerically fragile over long
recordings. First, each neuron maintains $N_\tau$ parallel leaky integrators at logarithmically
spaced anchor timescales $\tau_1, \dots, \tau_{N_\tau}$ spanning $10\,\mathrm{ms}$ to
$1\,\mathrm{s}$,

$$
c^{(k)}_{i,t} = \mathrm{clip}\Big(\gamma^{(k)} c^{(k)}_{i,t-1} + x_{i,t},\; v_{\min},\; v_{\max}\Big),
\qquad
\gamma^{(k)} = \exp\!\Big(-\mathrm{sp}\big(\delta^{(k)}\big)\frac{dt}{\tau_k}\Big),
\tag{16}
$$

with $\delta^{(k)}$ learnable but shared across the population. The bound in (16) is required
here, unlike in (6), because a leaky integrator possesses no intrinsic saturation and its
steady-state gain approaches $1/(1 - \gamma^{(k)})$, of order $10^3$ at the slowest timescale
considered; unbounded, the trace grows to a magnitude that destabilises the gradients of the
downstream read-out.

Second, coefficients are obtained from a learnable bilinear read-out of the resulting
multi-timescale trace $\boldsymbol{c}_{i,t} \in \mathbb{R}^{N_\tau}$ rather than as a difference
of raw values. Two small multilayer perceptrons, each of the form $\mathrm{Linear}
\rightarrow \mathrm{SiLU} \rightarrow \mathrm{Linear}$ mapping $N_\tau \rightarrow 4 d_v
\rightarrow d_v$, play roles analogous to queries and keys, and their outputs are
root-mean-square normalised before contraction,

$$
\boldsymbol{f}_{i,t} = \mathrm{RMSNorm}\big(\mathrm{MLP}_f(\boldsymbol{c}_{i,t})\big),
\qquad
\boldsymbol{g}_{j,t} = \mathrm{RMSNorm}\big(\mathrm{MLP}_g(\boldsymbol{c}_{j,t})\big),
\qquad
a_{ij,t} = \frac{\boldsymbol{f}_{i,t}^{\!\top} \boldsymbol{g}_{j,t}}{\sqrt{d_v}}.
\tag{17}
$$

The normalisation in (17) renders the magnitude of the coefficient independent of the raw scale
of the trace, and thus provides a safeguard on the read-out that is independent of the bound
imposed on its input in (16). The matrix $\boldsymbol{A}_t$ so obtained enters the remainder of
the architecture exactly as in the spiking case, through the value projection (9), the
feedforward sublayer (10) and the latent bottleneck of Section 4.5. We note that (17) is
symmetric in its treatment of the two neurons in the sense that no temporal offset is imposed
between them, the directionality of the mechanism residing entirely in the causality of the
traces (16); the sharp causal asymmetry of the spiking rule (5)–(7) has no counterpart here,
consistent with the absence of precisely timed discrete events in the fluorescence signal.

## 6. Optimisation

### 6.1 Variational predictive likelihood

To address the maximisation of (2), we adopt a variational formulation resting on a decoding
network that implements at every time-step the PCC distribution

$$
q^{(\tau_p, \tau_f)}_{\boldsymbol{w}^d}\big(\boldsymbol{r}_t \,\|\, \boldsymbol{z}_t\big) = q_{\boldsymbol{w}^d}\big(\boldsymbol{r}_t^{\,t+\tau_f-1} \mid \boldsymbol{r}^{t-1}, \boldsymbol{z}^{\,t}_{t-\tau_p+1}\big)
\tag{18}
$$

between the latent representation and the reference signal (28, 64). The decoder (18) is causal,
with memory given by the integer $\tau_p \ge 1$, and is parameterised by a vector
$\boldsymbol{w}^d$. In practice it is implemented as $q_{\boldsymbol{w}^d}\big(\boldsymbol{r}_t^{\,
t+\tau_f-1} \mid \boldsymbol{z}^{\,t}_{t-\tau_p+1}\big)$, realised by the output layer of an
artificial neural network whose input is the window $\boldsymbol{z}^{\,t}_{t-\tau_p+1}$ of latent
samples; alternatively, a recurrent or probabilistic spiking network (65) may be used to model
the kernel (18) directly. The default decoder is a multilayer perceptron of one hidden layer of
width $4 \tau_p d$ with a zero-initialised output layer, so that optimisation begins from an
exactly constant prediction and the reconstruction pathway develops only as the encoder acquires
informative structure. The output dimension is scaled by $\tau_f$, and the corresponding
$\tau_f$-step target windows are formed by unfolding the reference sequence. Either a separate
output layer per session or a single shared output layer may be used.

Applying a standard variational inequality (63) to lower-bound the decoder log-likelihood at each
time-step and averaging over time-steps yields the *variational predictive likelihood*

$$
\mathcal{L}_{\boldsymbol{w}}(\boldsymbol{x}, \boldsymbol{r}) = \mathbb{E}_{p_{\boldsymbol{w}^e}(\boldsymbol{z}\|\boldsymbol{x})}\Bigg[\frac{1}{T}\sum_{t=1}^{T} \log q^{(\tau_p,\tau_f)}_{\boldsymbol{w}^d}\big(\boldsymbol{r}_t \,\|\, \boldsymbol{z}_t\big)\Bigg] \;-\; \beta\, \mathrm{KL}\Big(p_{\boldsymbol{w}^e}\big(\boldsymbol{z} \,\|\, \boldsymbol{x}\big) \,\Big\|\, p(\boldsymbol{z})\Big),
\tag{19}
$$

where $\boldsymbol{w} = (\boldsymbol{w}^e, \boldsymbol{w}^d)$, which is maximised by stochastic
gradient descent using the Adam optimiser (66) and Monte Carlo gradients obtained by the
stochastic gradient variational Bayes estimator (67). With $\tau_p = 1$, $\tau_f = 1$ and $\beta$
equal to the reciprocal of the sequence length, (19) reduces to the standard evidence lower bound.

The original description fixes the weight of the divergence term at $1/T$, as follows from the
derivation of the bound. Here it is exposed as a free hyperparameter $\beta$, for the following
reason. When the reference signal is categorical and the reconstruction term is a cross-entropy,
hard targets admit no finite optimum: the loss decreases monotonically as the norm of the
decoder's logits, and hence of the latent mean feeding them, grows without bound. The divergence
term is then the only counterweight, and the equilibrium magnitude of the latent representation
scales approximately as $\beta^{-1}$. Two measures are used together in this regime. The weight
$\beta$ is treated as a hyperparameter to be tuned, and the cross-entropy is computed with label
smoothing (new ref. 78), which restores a finite-norm optimum by softening the target
distribution and thus removes the incentive at its origin rather than merely opposing it. For
continuous reference signals, for which the reconstruction term is a squared error or a Poisson
negative log-likelihood and no such incentive exists, $\beta = 1/T$ recovers the original
formulation.

### 6.2 Choice of latent prior

Equation (19) specifies the divergence against a prior $p(\boldsymbol{z})$ but leaves its form
open. Two choices are implemented.

The first, used in the original description, is a fixed factorised prior $\mathcal{N}(0, I)$
applied independently at each time-step, for which the divergence reduces to

$$
\mathrm{KL}^{\mathrm{iid}} = -\frac{1}{2 B T d}\sum_{b,t,k}\Big(1 + \log \sigma^2_{t,k} - \mu_{t,k}^2 - \sigma^2_{t,k}\Big).
\tag{20}
$$

The second is a learnable autoregressive prior $p(\boldsymbol{z}_t \mid \boldsymbol{z}_{t-1})$,
in the spirit of sequential latent-variable models for time series (new ref. 79). Its motivation
is that a factorised prior penalises any departure of the marginal posterior from the origin at
each instant independently, and so is indifferent to whether the latent trajectory is smooth,
whereas the objective (19) is constructed precisely on the premise that neural and behavioural
states evolve coherently over short timescales. An autoregressive prior encodes that premise in
the regulariser rather than solely in the reconstruction term. A small gated network predicts the
prior from the sampled previous latent,

$$
\boldsymbol{h}_t = \mathrm{SiLU}\big(W_h \boldsymbol{z}_{t-1}\big),
\quad
\boldsymbol{g}_t = \mathrm{sigm}\big(W_g \boldsymbol{z}_{t-1}\big),
\tag{21}
$$

$$
\boldsymbol{\mu}^p_t = \big(1 - \boldsymbol{g}_t\big) \odot A \boldsymbol{z}_{t-1} + \boldsymbol{g}_t \odot W_\mu \boldsymbol{h}_t,
\qquad
\log \boldsymbol{\sigma}^{2,p}_t = \mathrm{clip}\big(W_\sigma \boldsymbol{h}_t,\, -6,\, 2\big),
\tag{22}
$$

with $A$ initialised to the identity, so that the transition begins as an approximately
identity one-step-ahead predictor and the learnable per-dimension gate $\boldsymbol{g}_t$
interpolates between this linear persistence prediction and a nonlinear correction. The
corresponding divergence, for $t \ge 2$, is evaluated per time-step and per dimension as

$$
\mathrm{kl}_{t,k} = \frac{1}{2}\Bigg(\log \sigma^{2,p}_{t,k} - \log \sigma^2_{t,k} + \frac{\sigma^2_{t,k} + \big(\mu_{t,k} - \mu^p_{t,k}\big)^2}{\sigma^{2,p}_{t,k}} - 1\Bigg),
\tag{23}
$$

the first time-step reverting to the corresponding term of (20) since no $\boldsymbol{z}_0$ is
available to condition upon. The total is the mean of (23) over sequences, time-steps and latent
dimensions, which is the same reduction as in (20). This is deliberate: it makes $\beta$
numerically comparable between the two choices of prior, so that a value calibrated under one may
be transferred to the other. The parameters of (21)–(22) are appended to $\boldsymbol{w}$ and
optimised jointly with the encoder and decoder.

### 6.3 Free bits

When the autoregressive prior is used, each term (23) is floored at a constant $\mathrm{fb}$
before averaging (new refs. 80, 81). The floor removes the gradient incentive to compress an
individual latent dimension below a prescribed rate once its divergence has fallen beneath the
threshold, thereby guaranteeing a minimum information rate per dimension and mitigating posterior
collapse, to which the autoregressive prior is more susceptible than the factorised one because a
sufficiently expressive transition can account for the latent trajectory without recourse to the
input.

### 6.4 Gradient estimation and parameter groups

Unbiased gradient estimates are obtained from mini-batches $\mathcal{B} \subseteq \mathcal{D}$
together with random samples $\boldsymbol{\epsilon}^{(l)} \sim \mathcal{N}(0, I)$ for $1 \le l
\le L$, and are computed by backpropagation and automatic differentiation.

Parameters are partitioned into two groups for the purposes of decoupled weight decay (new ref.
82). Matrix-valued parameters are decayed; biases, normalisation scales, and the model's
scalar, gate-like and embedding-like parameters — the softplus-parameterised decay and weight
parameters of (8), the recurrent decay rates of (13), the per-neuron identity embeddings of (11)
and the latent queries of Section 4.5 — are not, none of these having a principled reason to be
regularised toward zero.

Gradients are clipped per submodule rather than jointly, each submodule that directly owns
learnable parameters being clipped independently against a common threshold. A single joint
constraint rescales every parameter by a common factor whenever the aggregate norm exceeds the
threshold, so that the component with the largest raw gradient scale determines the effective
step size of every other component, irrespective of how well conditioned those components'
own gradients may be. Clipping per submodule confines each constraint to the component that
violates it. A diagnostic reporting the per-parameter and aggregate pre-clipping gradient norms
is provided for calibrating the threshold, and a warning is raised when the pre-clipping norm
exceeds it by a large factor, which indicates that the constraint is rescaling every step rather
than intervening on outliers.

### 6.5 Online optimisation

The gradient of (19) requires backpropagation through time, whose memory cost grows with the
number of time-steps and which is therefore prohibitive for recordings of long duration. The
original description addresses this by the approximation

$$
\nabla_{\boldsymbol{w}} \mathcal{L}_{\boldsymbol{w}}\big(\boldsymbol{x}^t, \boldsymbol{r}_t^{\,t+\tau_f}\big) \approx \mathbb{E}_{p(\boldsymbol{x}^t, \boldsymbol{r}_t^{t+\tau_f})} \mathbb{E}_{p(\boldsymbol{\epsilon})}\Big[\nabla_{\boldsymbol{w}} \log q_{\boldsymbol{w}^d}\big(\boldsymbol{r}_t^{\,t+\tau_f} \mid \mu(f_{\boldsymbol{w}^e}(\boldsymbol{x}^t)) + \sigma(f_{\boldsymbol{w}^e}(\boldsymbol{x}^t)) \odot \boldsymbol{\epsilon}\big)\Big],
\tag{24}
$$

which neglects the contribution of derivatives from previous time-steps $t' < t$, permitting an
update at every time-step.

### 6.6 Truncated optimisation over chunks

The present implementation generalises (24) along a single axis. Rather than truncating the
recursion to one time-step, a recording is traversed in consecutive, non-overlapping chunks of
$C$ time-steps, within which the forward dynamics of every recurrent component evolve without
approximation, and the backward pass alone is truncated at chunk boundaries. Concretely, the
value of every recurrent state is carried from one chunk to the next, while its dependence on the
preceding chunk's computation graph is discarded, in the manner of truncated backpropagation
through time (new ref. 83). The states so propagated are the eligibility traces (5) and (7)
together with the coefficient matrix of (6); the recurrent state (13) of the linear-attention
read-out, when that read-out is used; and the trailing $\tau_p - 1$ latent samples required to
complete the decoder window (18) at the start of the following chunk. Each chunk contributes an
independent gradient estimate and parameter update, so memory consumption is independent of the
number of chunks into which the recording is divided.

The two limits of this scheme are informative. Taking $C = 1$ recovers (24) exactly. Taking $C =
T$ recovers untruncated backpropagation through the whole sequence. Intermediate values retain
$C$ time-steps of exact gradient flow through the eligibility dynamics — the mechanism by which
the model represents temporal structure — at a memory cost set by $C$ rather than by the duration
of the recording. Where a burn-in period is excluded from the reconstruction term, it is excluded
only within the first chunk of a recording, later chunks resuming from an already equilibrated
state.

The same traversal is available at evaluation time, without gradient computation, for recordings
whose per-time-step coefficient matrices would exceed available memory even for inference. In
that setting the reported loss accumulates as a sum over chunks of each chunk's own mean-reduced
loss, by analogy with the summation already performed across mini-batches, and is therefore not
directly comparable in magnitude to that of a single untruncated pass over the same data.

## 7. Alternative attention mechanisms

The mechanisms described in this section substitute for the Hebbian attention block of Section 4
while leaving every other component of the architecture and of the objective unchanged. The first
four retain the coefficient-matrix formulation, and are routed through the same value projection
(9), feedforward sublayer (10) and latent bottleneck as the reference model; they therefore
isolate a single property of the coefficient computation at a time. The three of Section 7.6
dispense with the coefficient matrix altogether and address the coarser question of whether any
explicit pairwise structure over neurons is required.

### 7.1 Symmetric Hebbian attention

To determine whether the causal asymmetry of (5)–(7) and the bound in (6) are individually
necessary, as opposed to Hebbian co-activation in general, we introduce a symmetric mechanism in
which the coefficient is potentiated irrespective of firing order. The formulation of the
original description retains per-pair traces and adds the two contributions symmetrically,

$$
a_{ij,t} := a_{ij,t-1} + s_{i,t}\, a^{\mathrm{post}}_{ij,t} + s_{j,t}\, a^{\mathrm{pre}}_{ij,t} + \big(1 - a_{ij,t-1}\big) s_{j,t} - a_{ij,t-1}\, s_{i,t}.
\tag{25}
$$

The present implementation separates the two variables under test. It removes the bounding terms
entirely, so that the accumulation is unbounded, and it maintains a single trace per neuron with
its own learnable decay and weight rather than a trace per ordered pair,

$$
\rho_{i,t} = \gamma_i \rho_{i,t-1} + w_i s_{i,t},
\qquad
a_{ij,t} = a_{ij,t-1} + \rho_{i,t}\, s_{j,t} + s_{i,t}\, \rho_{j,t},
\tag{26}
$$

with $\gamma_i$ and $w_i$ parameterised as in (8). Exchanging $i$ and $j$ in (26) permutes the two
additive contributions and leaves their sum invariant, so the mechanism is symmetric by
construction; and because the trace entering the update is the post-injection value $\rho_{i,t}$
rather than its decayed predecessor, simultaneous activity in both neurons contributes to the
coefficient, in contrast to (6). The comparison with the reference model thus tests the
asymmetric temporal offset and the bound jointly, and the comparison with (25) isolates the
bound. As in Section 4.8, the trace in (26) is evaluated by parallel prefix scan; the coefficient
accumulation, being unbounded, has unit multiplicative factor throughout and reduces to a
cumulative sum, so that no sequential pass is required for this control.

### 7.2 Hebbian attention without autoregression

To isolate the contribution of the autoregressive accumulation, coefficients may be computed
without dependence on their own previous value, retaining the bounding terms,

$$
a_{ij,t} := s_{i,t}\, a^{\mathrm{post}}_{ij,t} + s_{j,t}\, a^{\mathrm{pre}}_{ij,t} + \big(1 - a_{ij,t-1}\big) s_{j,t} - a_{ij,t-1}\, s_{i,t}.
\tag{27}
$$

Eligibility traces are computed as in Section 4.1, and all remaining components are unchanged.
This control is not at present provided as a substitutable block within the modular framework
used for those of Sections 7.1 and 7.4–7.6; comparison against the current reference model, as
opposed to the originally described one, would require reimplementing it on the shared scaffold
described in Section 7.

### 7.3 Conventional dot-product attention

For comparison with a conventional Transformer encoder, the Hebbian mechanism may be replaced by
scaled dot-product attention (26) computed independently at each time-instant,

$$
\mathrm{Attention}(Q, K, V)_t = \mathrm{softmax}\bigg(\frac{Q K^{\!\top}}{\sqrt{d_k}}\bigg) V.
\tag{28}
$$

A variant additionally endows this mechanism with autoregression, by passing the input through a
layer computing a per-neuron eligibility trace with learnable parameters $\tau_s$ and $w_i$,

$$
a_{i,t} := a_{i,t-1} \exp\!\Big({-\frac{dt}{\tau_s}}\Big) + w_i\, dt\, s_{i,t},
\tag{29}
$$

before attention is applied, thereby separating the contribution of temporal accumulation from
that of the pairwise Hebbian structure.

### 7.4 Instantaneous correlation attention

To determine whether learnable, temporally accumulating dynamics are required at all, as opposed
to a fixed short-memory statistic of smoothed activity, we introduce a mechanism whose
coefficients are a running, causal Pearson correlation between the smoothed rates of each pair of
neurons. Writing $\mathrm{EMA}_{\tau_s}(u)_t = \gamma u_{t-1} + (1 - \gamma) u_t$ with $\gamma =
\exp(-dt/\tau_s)$ for the exponential moving average at timescale $\tau_s$, the construction
proceeds in three stages:

$$
\rho_{i,t} = \mathrm{EMA}_{\tau_s}(s_i)_t,
\qquad
m_{i,t} = \mathrm{EMA}_{\tau_s}(\rho_i)_t,
\qquad
c_{i,t} = \rho_{i,t} - m_{i,t},
\tag{30}
$$

$$
\Sigma_{ij,t} = \mathrm{EMA}_{\tau_s}\big(c_i c_j\big)_t,
\qquad
a_{ij,t} = \frac{\Sigma_{ij,t}}{\sqrt{\Sigma_{ii,t}\, \Sigma_{jj,t}} + \varepsilon}.
\tag{31}
$$

Both the centring and the normalisation in (31) are necessary for the control to be meaningful.
The unnormalised outer product $\rho_{i,t}\rho_{j,t}$, which is the immediate alternative, is
non-negative everywhere and is dominated by shared population drive, so that any two neurons
firing at high rates appear strongly related irrespective of whether their activity co-fluctuates;
centring against the slower baseline $m_{i,t}$ and normalising by the running variances yields a
quantity confined to approximately $[-1, 1]$ that reflects co-fluctuation as such. On synthetic
sequences with known pairwise structure, (31) recovers values near $+1$ for identical neurons,
near $-1$ for exact complements and near zero in mean for independent neurons, once a transient
of order $\tau_s / dt$ time-steps has elapsed. This mechanism introduces no learnable parameters
of its own; only the shared downstream components are optimised.

### 7.5 Sequence-model controls

The three remaining controls map the smoothed population activity $\rho_{i,t}$ of (30) directly
onto the $M d_v$-dimensional representation, without forming a coefficient matrix and without the
latent bottleneck. They test whether explicit pairwise structure over neurons confers any
advantage over generic sequence models operating on the same smoothed input under the same causal
constraint.

The first applies a position-wise multilayer perceptron with one hidden layer of width $4 d_v$
and a Gaussian error linear unit nonlinearity, so that the only mixing across neurons is that
effected by its dense weights and the only recurrence is that of the moving average itself. The second
applies a gated recurrent unit (new ref. 84) with hidden width $d_v$ followed by a linear
read-out, providing a recurrent baseline with a learnable hidden state but no attention. The
third applies the same causal self-attention block used in the conventional stack of Section 3,
preceded and followed by linear projections.

The third control carries a caveat under the chunked traversal of Section 6.6. Only the state of
the moving average is propagated across chunk boundaries; the self-attention maintains no
key–value cache and recomputes attention over each chunk in isolation, with rotary positions
restarting at the beginning of every chunk. Its effective attention span is therefore bounded by
$C$, in a way that does not apply to the other mechanisms, and comparisons involving this control
should either hold $C$ fixed or account for this dependence. Supplying it with a key–value cache
would require modifying the attention implementation shared with the conventional stack, and has
not been undertaken.

### 7.6 Summary of the mechanisms compared

**Table 1 | Properties of the attention mechanisms.** Learnable dynamics refers to whether the
parameters governing the temporal accumulation are optimised, as distinct from those of the
shared downstream components, which are optimised in every case. Parameter counts are those of
the coefficient computation alone.

| Mechanism | Coefficient structure | Learnable dynamics | Causal asymmetry | Bounded | Own parameters |
|---|---|---|---|---|---|
| Hebbian attention, *full* (Sect. 4) | STDP, traces per ordered pair | yes | yes | soft and hard | $4N^2$ |
| Hebbian attention, *light* (Sect. 4) | STDP, traces per neuron | no | yes | soft and hard | none |
| Hebbian attention, calcium (Sect. 5) | multi-timescale traces, bilinear read-out | yes | no | trace bound | $N_\tau$ and two perceptrons |
| Symmetric Hebbian (Sect. 7.1) | one trace per neuron, symmetric sum | yes | no | none | $2N$ |
| Without autoregression (Sect. 7.2) | instantaneous STDP, traces per pair | yes | yes | soft only | $4N^2$ |
| Dot-product attention (Sect. 7.3) | learned query–key similarity | no, or (29) | no | softmax | projections |
| Instantaneous correlation (Sect. 7.4) | running Pearson correlation | no | no | by normalisation | none |
| Perceptron, recurrent unit, Transformer (Sect. 7.5) | none | no; yes; yes | — | — | see text |

## 8. Model evaluation

For unsupervised learning, results are evaluated through co-smoothing. The metric is based on the
difference between the log-likelihood of a Poisson model with rates $\boldsymbol{\lambda}$
corresponding to the model's predictions for held-out neurons $\hat{\boldsymbol{y}}$, and that of
a mean model with rate $\bar{\lambda} = \tfrac{1}{T}\sum_t \hat{y}_t$ equal to the mean firing
rate of those neurons, expressed in bits per spike,

$$
\text{bits/spike} = \frac{1}{n_{sp} \log 2}\Big(\mathcal{L}\big(\boldsymbol{\lambda}; \hat{\boldsymbol{y}}_{n,t}\big) - \mathcal{L}\big(\bar{\lambda}_n; \hat{\boldsymbol{y}}_{n,t}\big)\Big),
\tag{32}
$$

where $n_{sp}$ is the total number of spikes. Poisson, Gaussian and Bernoulli observation models
are available for the likelihood in (32), and the metric may be reported either aggregated over
the population or per neuron.

## 9. Default hyperparameters

**Table 2 | Default hyperparameter values.** Values in the upper block define the attention
mechanism and its numerical safeguards; those in the lower block define the objective and the
optimisation procedure.

| Symbol or name | Default | Description |
|---|---|---|
| $dt$ | $1\,\mathrm{ms}$ | sampling period |
| $\tau_s$ | $1\,\mathrm{s}$ | eligibility-trace and moving-average timescale |
| $w_+$ | $10^{-3}$ | base trace increment |
| $\alpha$ | $1.1$ | ratio of post- to pre-synaptic increment |
| configuration | *light* | parameter sharing in the attention layer (Sect. 4.2) |
| $K$ | $1$ | Hebbian attention heads (Sect. 4.6) |
| $H$ | $1$ | heads in the latent bottleneck (Sect. 4.5) |
| read-out | linear attention | bottleneck read-out mechanism (Sect. 4.5) |
| $N_\tau$ | $5$ | calcium timescales, log-spaced over $10\,\mathrm{ms}$–$1\,\mathrm{s}$ |
| $[v_{\min}, v_{\max}]$ | $[-10, 10]$ | calcium trace bound, eq. (16) |
| $\log\sigma^2$ range | $[-4, 2]$ | encoder log-variance constraint (Sect. 3) |
| prior | factorised | latent prior (Sect. 6.2) |
| $\mathrm{fb}$ | $0.5$ | free-bits floor, autoregressive prior only (Sect. 6.3) |
| clipping threshold | $5$ | per-submodule gradient norm (Sect. 6.4) |
| weight decay | $0.1$ | decayed parameter group only (Sect. 6.4) |

## 10. Relationship to the original description

**Table 3 | Differences between the original description and the present implementation.**

| Component | Original description | Present implementation |
|---|---|---|
| Coefficient bound | soft bounds, eq. (6) | soft bounds and hard bound $[-0.5, 1.5]$ |
| STDP parameter sharing | learned per ordered pair | *full* as described; *light* (default) fixed and shared |
| Population read-out | flatten and project linearly | latent cross-attention bottleneck, $M$ slots (Sect. 4.5) |
| Bottleneck read-out | — | softmax cross-attention, or decaying linear attention (default) |
| Conventional blocks | additive positional encoding, plain residual | rotary embeddings, gated residual branches |
| Feedforward sublayer | fully connected, batch normalisation | gated linear unit, root-mean-square normalisation, stochastic depth |
| Calcium coefficients | difference of raw traces | $N_\tau$ bounded traces and bilinear read-out, eqs. (16)–(17) |
| Latent prior | factorised $\mathcal{N}(0, I)$ | factorised, or learned autoregressive, eqs. (21)–(23) |
| Divergence weight | $1/T$ | free hyperparameter $\beta$ |
| Free bits | — | per-dimension floor under the autoregressive prior |
| Encoder log-variance | unconstrained | constrained to $[-4, 2]$ |
| Recurrence evaluation | sequential | exact parallel prefix scan, or vectorised sequential (Sect. 4.8) |
| Gradient truncation | one time-step, eq. (24) | chunks of $C$ time-steps; $C = 1$ recovers eq. (24) |
| Gradient clipping | — | per submodule |
| Controls | symmetric, without autoregression, dot-product with and without autoregression | additionally instantaneous correlation, perceptron, recurrent unit, Transformer |

## 11. Implementation notes and limitations

A sliding-window configuration of the attention layer, restricting each neuron's coefficients to
a local block of the population, is present in the code base and was described as a
scalability measure in the original release notes. It is not currently functional, and the
results reported here do not use it; the two configurations of Section 4.2 are the supported
options.

The stochastic latent sample $\boldsymbol{z}_t$ is not reproducible across different chunkings of
the same input, since fresh reparameterisation noise is drawn on each traversal. The
distributional parameters $\mu$ and $\log\sigma^2$ and the decoder outputs are reproducible, and
have been verified to be numerically identical between chunked and untruncated traversal given
the same propagated state. Comparisons across chunkings should therefore be made on the
distributional parameters rather than on individual samples.

The parallel and sequential evaluations of every recurrence in Section 4.8 have been verified
against independent sequential reference implementations, to floating-point precision, including
under resumption from a stored state at a chunk boundary.

The quadratic dependence of the coefficient matrix on $N$ remains the principal constraint on
scalability, and bounds the size of population that can be processed simultaneously.

## 12. References

References numbered 26–68 follow the numbering of the original bibliography. Those numbered from
69 are introduced in this document and should be renumbered on integration.

- 26. A. Vaswani *et al.*, Attention is all you need.
- 27. G. Kramer, Directed information for channels with feedback. Thesis, ETH Zurich (1998).
- 28. N. Skatchkovsky, O. Simeone, H. Jang, Learning to time-decode in spiking neural networks
  through the information bottleneck. *Adv. Neural Inf. Process. Syst.* **34** (2021).
- 63. O. Simeone, *Machine Learning for Engineers*. Cambridge University Press (2022).
- 64. A. A. Alemi, I. Fischer, J. V. Dillon, K. Murphy, Deep variational information bottleneck.
  arXiv:1612.00410.
- 65. H. Jang, O. Simeone, B. Gardner, A. Gruning, An introduction to probabilistic spiking neural
  networks. *IEEE Signal Process. Mag.* **36** (2019).
- 66. D. P. Kingma, J. Ba, Adam: a method for stochastic optimization. arXiv:1412.6980.
- 67. D. P. Kingma, M. Welling, Auto-encoding variational Bayes. arXiv:1312.6114.
- 68. J. Sjöström, W. Gerstner, Spike-timing dependent plasticity. *Scholarpedia* **5**, 1362 (2010).
- 69. J. Su *et al.*, RoFormer: enhanced Transformer with rotary position embedding (2021).
- 70. H. Touvron *et al.*, Going deeper with image Transformers (2021).
- 71. N. Shazeer, GLU variants improve Transformer (2020).
- 72. G. Huang *et al.*, Deep networks with stochastic depth (2016).
- 73. A. Jaegle *et al.*, Perceiver: general perception with iterative attention (2021).
- 74. A. Katharopoulos *et al.*, Transformers are RNNs: fast autoregressive Transformers with
  linear attention (2020).
- 75. G. E. Blelloch, Prefix sums and their applications (1990).
- 76. E. Martin, C. Cundy, Parallelizing linear recurrent neural networks over sequence length (2018).
- 77. J. T. H. Smith, A. Warrington, S. W. Linderman, Simplified state space layers for sequence
  modeling (2023).
- 78. C. Szegedy *et al.*, Rethinking the Inception architecture for computer vision (2016).
- 79. J. Chung *et al.*, A recurrent latent variable model for sequential data (2015).
- 80. D. P. Kingma *et al.*, Improved variational inference with inverse autoregressive flow (2016).
- 81. X. Chen *et al.*, Variational lossy autoencoder (2016).
- 82. I. Loshchilov, F. Hutter, Decoupled weight decay regularization (2019).
- 83. R. J. Williams, J. Peng, An efficient gradient-based algorithm for on-line training of
  recurrent network trajectories. *Neural Comput.* **2** (1990).
- 84. J. Chung *et al.*, Empirical evaluation of gated recurrent neural networks on sequence
  modeling (2014).
