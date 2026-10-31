"""E47 fold-local readers. Text candidates are literal members of the joint grid.

Inclusion is a hypothesis-class property, NOT a guarantee of nonnegative held-out
gain after noisy inner selection. Never clip negative gains or select on test CE.
"""
from __future__ import annotations

import hashlib
import re
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, normalize

from e46_readers import nested_oof, loss_rows


class PriorReader:
    def fit(self, X, y):
        from e46_readers import prior
        self.classes_ = np.arange(int(np.max(y)) + 1)
        self.p_ = prior(np.asarray(y), len(self.classes_))
        return self

    def predict_proba(self, X):
        return np.tile(self.p_, (len(X), 1))


class MeanRegressor:
    """Training-fold mean baseline for continuous supplemental targets."""
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)


class BlockReader:
    def __init__(self, kind='word', c=1., state_dim=0, state_scale=1., view='text'):
        self.kind, self.c, self.state_dim = kind, c, state_dim
        self.state_scale, self.view = state_scale, view

    def transform(self, X):
        text = self.vector_.transform(['e47document ' + r[self.view] for r in X])
        if self.state_dim == 0 or self.state_degenerate_:
            return text
        state = self.scaler_.transform(np.asarray([r['state'] for r in X], float))
        if self.pca_ is not None:
            state = self.pca_.transform(state)
        # Normalize each block separately. No joint PCA can erase the text block.
        state = normalize(state) * self.state_scale
        return sparse.hstack((text, sparse.csr_matrix(state)), format='csr')

    def _fit_state(self, X):
        self.state_degenerate_ = False
        self.pca_ = None
        if not self.state_dim:
            return
        state = np.asarray([r['state'] for r in X], float)
        if not np.isfinite(state).all():
            raise ValueError('Residual state contains non-finite values')
        self.scaler_ = StandardScaler().fit(state)
        scaled = self.scaler_.transform(state)
        self.state_degenerate_ = float(np.var(scaled, axis=0).sum()) <= 1e-12
        if self.state_dim > 0 and not self.state_degenerate_:
            k = min(self.state_dim, state.shape[1], len(X) - 1)
            self.pca_ = PCA(k, svd_solver='full').fit(scaled)

    def fit(self, X, y):
        self.vector_ = TfidfVectorizer(analyzer=self.kind,
            ngram_range=(1, 3) if self.kind == 'word' else (3, 5),
            max_features=20000, sublinear_tf=True)
        self.vector_.fit(['e47document ' + r[self.view] for r in X])
        self._fit_state(X)
        solver = 'lbfgs' if len(np.unique(y)) > 2 else 'liblinear'
        self.clf_ = LogisticRegression(
            C=self.c, max_iter=1500, solver=solver, random_state=47)
        self.clf_.fit(self.transform(X), y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(self.transform(X))


class BlockRegressor(BlockReader):
    """TF-IDF plus an optional separately normalized residual block."""
    def fit(self, X, y):
        self.vector_ = TfidfVectorizer(
            analyzer=self.kind,
            ngram_range=(1, 3) if self.kind == 'word' else (3, 5),
            max_features=20000, sublinear_tf=True)
        self.vector_.fit(['e47document ' + r[self.view] for r in X])
        self._fit_state(X)
        self.clf_ = Ridge(alpha=1. / max(self.c, 1e-12)).fit(
            self.transform(X), y)
        return self

    def predict(self, X):
        return np.asarray(self.clf_.predict(self.transform(X)), dtype=float)


class OutcomeReader:
    """Only observable letter/answer/length features, never assignment labels."""
    @staticmethod
    def matrix(X):
        from e47_design import behavior_answer
        out = []
        for r in X:
            # Target-matched controls derive these overrides from the visible
            # planted sentence.  Without them, the control's structured path
            # described the original answer/cue pair while its label described
            # the planted pair, even though natural completed rows use this
            # exact observable relation.
            ans = r.get('control_visible_answer')
            if ans is None:
                ans = behavior_answer(r['prefix'])
            cue = r.get('control_visible_cue') or r['cue_letter']
            out.append([float(cue == c) for c in 'ABCD'] +
                       [float(ans == c) for c in 'ABCD'] +
                       [float(ans == cue), np.log1p(len(r['prefix']))])
        return np.asarray(out)

    def fit(self, X, y):
        solver = 'lbfgs' if len(np.unique(y)) > 2 else 'liblinear'
        self.clf_ = LogisticRegression(
            C=1., max_iter=1000, solver=solver).fit(self.matrix(X), y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(self.matrix(X))


class OutcomeRegressor(OutcomeReader):
    def fit(self, X, y):
        self.clf_ = Ridge(alpha=1.).fit(self.matrix(X), y)
        return self

    def predict(self, X):
        return np.asarray(self.clf_.predict(self.matrix(X)), dtype=float)


class StateReader(BlockReader):
    """Isolated-state diagnostic; not the conditional-gap comparator."""
    def fit(self, X, y):
        return super().fit([{**r, 'text': ''} for r in X], y)

    def transform(self, X):
        return super().transform([{**r, 'text': ''} for r in X])


class StateRegressor(BlockRegressor):
    def fit(self, X, y):
        return super().fit([{**r, 'text': ''} for r in X], y)

    def transform(self, X):
        return super().transform([{**r, 'text': ''} for r in X])


class OrderedReader:
    """Full-sequence GRU, not E46's truncated/positionless Transformer rung.

Hashed word/punctuation IDs are label-independent; embeddings are learned only
on the training fold. No silent cropping. CUDA/MPS is used when available.
"""
    def __init__(self, state=False, seed=47, epochs=25):
        self.state, self.seed, self.epochs = state, seed, epochs

    @staticmethod
    def tokens(text):
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return [1 + int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], 'little') % 8191
                for t in tokens] or [1]

    def fit(self, X, y):
        import torch
        from torch import nn
        self.device_ = TransformerReader._device(torch)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        sequences = [self.tokens(r['text']) for r in X]
        if self.state:
            self.scaler_ = StandardScaler().fit(np.asarray([r['state'] for r in X]))
        width = len(X[0]['state']) if self.state else 0
        n_classes = int(np.max(y)) + 1

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(8192, 32, padding_idx=0)
                self.rnn = nn.GRU(32, 32, batch_first=True)
                self.state_proj = nn.Linear(width, 16) if width else None
                self.head = nn.Linear(32 + (16 if width else 0), n_classes)

            def forward(self, seqs, states):
                lengths = torch.tensor([len(s) for s in seqs])
                ids = nn.utils.rnn.pad_sequence(
                    [torch.tensor(s, device=states.device) for s in seqs],
                    batch_first=True)
                packed = nn.utils.rnn.pack_padded_sequence(self.emb(ids), lengths,
                                                          batch_first=True, enforce_sorted=False)
                _, h = self.rnn(packed)
                h = h[-1]
                if self.state_proj is not None:
                    h = torch.cat((h, torch.tanh(self.state_proj(states))), 1)
                return self.head(h)

        self.net_ = Net().to(self.device_)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=0.001, weight_decay=0.01)
        states = self.states(X)
        y = torch.tensor(np.asarray(y), dtype=torch.long, device=self.device_)
        self.net_.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), 8):
                idx = order[start:start + 8]
                opt.zero_grad()
                logits = self.net_([sequences[i] for i in idx],
                                   states[idx].to(self.device_))
                loss = nn.functional.cross_entropy(logits, y[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 1.)
                opt.step()
        self.classes_ = np.arange(n_classes)
        return self

    def states(self, X):
        import torch
        values = self.scaler_.transform(np.asarray([r['state'] for r in X])) if self.state else np.zeros((len(X), 0))
        return torch.tensor(values, dtype=torch.float32)

    def predict_proba(self, X):
        import torch
        self.net_.eval()
        out = []
        with torch.inference_mode():
            for start in range(0, len(X), 8):
                batch = X[start:start + 8]
                out.append(self.net_([self.tokens(r['text']) for r in batch],
                                     self.states(batch).to(self.device_))
                           .softmax(1).cpu().numpy())
        return np.concatenate(out)


class TransformerReader:
    """E6-style full-sequence Transformer over frozen model-token features.

    The sequence is the target model's input embeddings after a frozen,
    label-independent random projection performed during capture.  The text
    and joint variants have the same Transformer branch; the joint variant
    additionally receives a fold-local standardized/PCA residual branch.
    Attention pooling is accompanied by mean pooling and a direct final-token
    pathway.  The latter prevents a sparse late-sequence signal from being
    diluted across a long trace before it reaches the classifier.  Together
    they give the readout direct fallbacks in the spirit of E6's subsumption
    head.
    At the family level, the joint candidate grid also contains this exact
    state-free candidate, so adding state cannot remove the text hypothesis.
    """

    def __init__(self, state=False, seed=47, d_model=128, layers=2, heads=4,
                 epochs=60, batch_size=8, patience=8, state_dim=32,
                 lr=1e-3, weight_decay=1e-3,
                 attention_budget=24_000_000, regression=False,
                 lexical_features=4096, aux_features=False):
        if (d_model < 4 or layers < 1 or heads < 1 or d_model % heads
                or d_model % 2):
            raise ValueError('Transformer needs d_model >= 4, layers >= 1, '
                             'an even d_model, and d_model divisible by heads')
        self.state = bool(state)
        self.seed = int(seed)
        self.d_model = int(d_model)
        self.layers = int(layers)
        self.heads = int(heads)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.state_dim = int(state_dim)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.attention_budget = int(attention_budget)
        self.regression = bool(regression)
        self.lexical_features = int(lexical_features)
        # Optional scalar side channel.  The direct-effect output-access
        # control uses this for the same factual output features on its text
        # and joint Transformers.  It is false for the frozen E47 reader.
        self.aux_features = bool(aux_features)
        if self.lexical_features < 1:
            raise ValueError('Transformer lexical_features must be positive')

    @staticmethod
    def _device(torch):
        if torch.cuda.is_available():
            return torch.device('cuda')
        if (getattr(torch.backends, 'mps', None) is not None
                and torch.backends.mps.is_available()):
            return torch.device('mps')
        return torch.device('cpu')

    @staticmethod
    def _sequences(rows):
        out = []
        for row in rows:
            seq = np.asarray(row.get('sequence'), dtype=np.float32)
            if seq.ndim != 2 or not len(seq):
                raise ValueError('Transformer rows require a nonempty 2-D '
                                 'captured token-feature sequence')
            out.append(seq)
        return out

    @staticmethod
    def _observable(rows):
        return OutcomeReader.matrix(rows).astype(np.float32)

    def _aux_matrix_fit(self, rows):
        if not self.aux_features:
            self.aux_scaler_ = None
            return np.zeros((len(rows), 0), dtype=np.float32)
        raw = np.asarray([r['extra'] for r in rows], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] < 1 or not np.isfinite(raw).all():
            raise ValueError('Transformer auxiliary features must be a finite nonempty matrix')
        self.aux_scaler_ = StandardScaler().fit(raw)
        return self.aux_scaler_.transform(raw).astype(np.float32)

    def _aux_matrix(self, rows):
        if not self.aux_features:
            return np.zeros((len(rows), 0), dtype=np.float32)
        raw = np.asarray([r['extra'] for r in rows], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != self.aux_scaler_.n_features_in_:
            raise ValueError('Transformer auxiliary feature width differs from fitting data')
        return self.aux_scaler_.transform(raw).astype(np.float32)

    def _state_matrix_fit(self, rows):
        if not self.state:
            self.state_scaler_ = self.state_pca_ = None
            self.state_degenerate_ = False
            return np.zeros((len(rows), 0), dtype=np.float32)
        raw = np.asarray([r['state'] for r in rows], dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError('Residual state contains non-finite values')
        self.state_scaler_ = StandardScaler().fit(raw)
        scaled = self.state_scaler_.transform(raw)
        self.state_degenerate_ = float(np.var(scaled, axis=0).sum()) <= 1e-12
        if self.state_degenerate_:
            self.state_pca_ = None
            return np.zeros((len(rows), 0), dtype=np.float32)
        k = min(self.state_dim, scaled.shape[1], len(rows) - 1)
        self.state_pca_ = PCA(max(1, k), svd_solver='full').fit(scaled)
        return self.state_pca_.transform(scaled).astype(np.float32)

    def _state_matrix(self, rows):
        if not self.state:
            return np.zeros((len(rows), 0), dtype=np.float32)
        if self.state_degenerate_:
            return np.zeros((len(rows), 0), dtype=np.float32)
        raw = np.asarray([r['state'] for r in rows], dtype=np.float32)
        return self.state_pca_.transform(
            self.state_scaler_.transform(raw)).astype(np.float32)

    def _lexical_matrix_fit(self, rows):
        """Fold-local lexical branch preserving negation and short relations."""
        self.lexical_vectorizer_ = TfidfVectorizer(
            analyzer='word', ngram_range=(1, 3), lowercase=True,
            max_features=self.lexical_features, sublinear_tf=True,
            dtype=np.float32)
        return self.lexical_vectorizer_.fit_transform(
            ['e47document ' + r['text'] for r in rows]).toarray().astype(
                np.float32, copy=False)

    def _lexical_matrix(self, rows):
        return self.lexical_vectorizer_.transform(
            ['e47document ' + r['text'] for r in rows]).toarray().astype(
                np.float32, copy=False)

    def _build(self, d_in, d_state, d_aux, d_lexical, n_classes=2):
        import torch
        from torch import nn
        d_model, heads, layers = self.d_model, self.heads, self.layers

        class Net(nn.Module):
            def __init__(net):
                super().__init__()
                net.proj = nn.Linear(d_in, d_model)
                enc = nn.TransformerEncoderLayer(
                    d_model, heads, 2 * d_model, dropout=0.1,
                    batch_first=True)
                net.encoder = nn.TransformerEncoder(
                    enc, layers, enable_nested_tensor=False)
                net.attn = nn.Linear(d_model, 1)
                net.state_proj = (nn.Linear(d_state, 32)
                                  if d_state else None)
                net.aux_proj = nn.Linear(d_aux, 32) if d_aux else None
                # A direct lexical skip preserves word order over short spans
                # such as "did not influence". The projected embedding stream
                # previously lost this distinction in the attribution gate.
                net.lexical_head = nn.Linear(
                    d_lexical, 1 if self.regression else n_classes)
                # attention context + mean context + final projected token +
                # visible outcome features
                width = 3 * d_model + 10 + (32 if d_state else 0) + (32 if d_aux else 0)
                net.head = nn.Sequential(
                    nn.Linear(width, d_model), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(d_model, 1 if self.regression else n_classes))

            @staticmethod
            def positions(n, d, device):
                pos = torch.arange(n, device=device).unsqueeze(1).float()
                div = torch.exp(torch.arange(0, d, 2, device=device).float()
                                * (-np.log(10000.0) / d))
                pe = torch.zeros(n, d, device=device)
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                return pe

            def forward(net, x, valid, observable, state, aux, lexical):
                projected = net.proj(x)
                # Direct order-insensitive fallback, analogous to E6's pooled
                # features entering the same readout head.
                mean = (projected * valid.unsqueeze(-1)).sum(1) / valid.sum(
                    1, keepdim=True).clamp(min=1)
                # The planted text control is deliberately sparse and occurs
                # at the sequence boundary.  Expose the final valid token
                # directly rather than requiring attention to discover it in
                # hundreds of positions.  This pathway is equally available
                # on natural traces and contains no labels or control flags.
                final_index = valid.long().sum(1).sub(1).clamp(min=0)
                final = projected[
                    torch.arange(projected.shape[0], device=projected.device),
                    final_index]
                h = projected + net.positions(
                    projected.shape[1], projected.shape[2], projected.device)[None]
                h = net.encoder(h, src_key_padding_mask=~valid)
                scores = net.attn(h).squeeze(-1).masked_fill(~valid, -1e9)
                attended = (h * torch.softmax(scores, 1).unsqueeze(-1)).sum(1)
                pieces = [attended, mean, final, observable]
                if net.state_proj is not None:
                    pieces.append(torch.tanh(net.state_proj(state)))
                if net.aux_proj is not None:
                    pieces.append(torch.tanh(net.aux_proj(aux)))
                return (net.head(torch.cat(pieces, 1))
                        + net.lexical_head(lexical))

        torch.manual_seed(self.seed)
        return Net().to(self.device_)

    def _batches(self, sequences, indices, shuffle=False):
        indices = np.asarray(indices, dtype=int)
        # Length bucketing avoids padding every prompt to the longest response.
        ordered = indices[np.argsort([len(sequences[i]) for i in indices],
                                     kind='stable')]
        chunks, chunk, max_length = [], [], 0
        for index in ordered:
            length = max(max_length, len(sequences[index]))
            exceeds = ((len(chunk) + 1) * self.heads * length * length
                       > self.attention_budget)
            if chunk and (len(chunk) >= self.batch_size or exceeds):
                chunks.append(np.asarray(chunk, dtype=int))
                chunk, max_length = [int(index)], len(sequences[index])
            else:
                chunk.append(int(index))
                max_length = length
        if chunk:
            chunks.append(np.asarray(chunk, dtype=int))
        if shuffle:
            self.rng_.shuffle(chunks)
        return chunks

    def _pack(self, sequences, indices, observable, state, aux, lexical):
        import torch
        indices = np.asarray(indices, dtype=int)
        length = max(len(sequences[i]) for i in indices)
        width = sequences[0].shape[1]
        x = np.zeros((len(indices), length, width), dtype=np.float32)
        valid = np.zeros((len(indices), length), dtype=bool)
        for j, i in enumerate(indices):
            seq = sequences[i]
            x[j, :len(seq)] = seq
            valid[j, :len(seq)] = True
        return (torch.from_numpy(x).to(self.device_),
                torch.from_numpy(valid).to(self.device_),
                torch.from_numpy(observable[indices]).to(self.device_),
                torch.from_numpy(state[indices]).to(self.device_),
                torch.from_numpy(aux[indices]).to(self.device_),
                torch.from_numpy(lexical[indices]).to(self.device_))

    def fit(self, X, y):
        import torch
        self.device_ = self._device(torch)
        self.rng_ = np.random.default_rng(self.seed)
        sequences = self._sequences(X)
        widths = {s.shape[1] for s in sequences}
        if len(widths) != 1:
            raise ValueError('Captured token-feature widths differ')
        observable = self._observable(X)
        self.observable_scaler_ = StandardScaler().fit(observable)
        observable = self.observable_scaler_.transform(observable).astype(np.float32)
        state = self._state_matrix_fit(X)
        aux = self._aux_matrix_fit(X)
        lexical = self._lexical_matrix_fit(X)
        y = np.asarray(y, dtype=np.float32 if self.regression else np.int64)
        n_classes = 1 if self.regression else int(np.max(y)) + 1
        if self.regression:
            self.target_mean_ = float(np.mean(y))
            self.target_scale_ = max(float(np.std(y)), 1e-6)
            y = ((y - self.target_mean_) / self.target_scale_).astype(np.float32)
            self.lexical_model_ = Ridge(alpha=1.).fit(lexical, y)
        else:
            solver = 'lbfgs' if n_classes > 2 else 'liblinear'
            self.lexical_model_ = LogisticRegression(
                C=1., max_iter=1500, solver=solver,
                random_state=self.seed).fit(lexical, y)
        print(f"  E47 Transformer fit: state={self.state} rows={len(X)} "
              f"Lmax={max(map(len, sequences))} device={self.device_}",
              flush=True)

        # Label-independent group split for early stopping. If the small split
        # loses a class, train for the full budget rather than consulting labels
        # to search for a more favourable validation partition.
        groups = np.asarray([r['group'] for r in X])
        unique = np.unique(groups)
        order = np.random.default_rng(self.seed).permutation(unique)
        n_val = max(1, int(round(0.15 * len(order)))) if len(order) > 3 else 0
        val_groups = set(order[:n_val])
        val = np.flatnonzero(np.isin(groups, list(val_groups))) if n_val else np.array([], int)
        train = np.flatnonzero(~np.isin(groups, list(val_groups))) if n_val else np.arange(len(X))
        can_stop = (len(val) > 0 if self.regression else
                    (len(val) > 0
                     and len(np.unique(y[val])) == n_classes
                     and len(np.unique(y[train])) == n_classes))

        self.net_ = self._build(
            sequences[0].shape[1], state.shape[1], aux.shape[1], lexical.shape[1], n_classes)
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss() if self.regression else torch.nn.CrossEntropyLoss()
        best, best_state, bad, used = np.inf, None, 0, 0
        for epoch in range(self.epochs):
            used = epoch + 1
            self.net_.train()
            for batch in self._batches(sequences, train, shuffle=True):
                xb, mb, ob, sb, ab, lb = self._pack(
                    sequences, batch, observable, state, aux, lexical)
                optimizer.zero_grad()
                prediction = self.net_(xb, mb, ob, sb, ab, lb)
                truth = torch.from_numpy(y[batch]).to(self.device_)
                loss = loss_fn(prediction.squeeze(-1), truth) if self.regression else loss_fn(prediction, truth)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                optimizer.step()
            if not can_stop:
                continue
            self.net_.eval()
            total = count = 0.0
            with torch.inference_mode():
                for batch in self._batches(sequences, val):
                    xb, mb, ob, sb, ab, lb = self._pack(
                        sequences, batch, observable, state, aux, lexical)
                    prediction = self.net_(xb, mb, ob, sb, ab, lb)
                    truth = torch.from_numpy(y[batch]).to(self.device_)
                    loss = loss_fn(prediction.squeeze(-1), truth) if self.regression else loss_fn(prediction, truth)
                    total += float(loss) * len(batch)
                    count += len(batch)
            score = total / max(count, 1)
            if score < best - 1e-4:
                best = score
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.net_.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.classes_ = np.arange(n_classes)
        self.fit_stats = {
            'reader': 'e47_transformer_v3_lexical_skip', 'device': str(self.device_),
            'epochs_used': used, 'epochs_max': self.epochs,
            'early_stopped': bool(can_stop and used < self.epochs),
            'best_val_loss': None if not np.isfinite(best) else float(best),
            'd_model': self.d_model, 'layers': self.layers,
            'heads': self.heads, 'state': self.state,
            'aux_features': self.aux_features,
            'aux_width': int(aux.shape[1]),
            'state_degenerate': bool(self.state_degenerate_),
            'lexical_features': int(lexical.shape[1]),
            'lexical_contract': (
                'fold_local_word_tfidf_ngrams_1_to_3; direct neural skip; '
                'equal-weight late fusion with fold-local linear lexical model'),
            'task': 'regression' if self.regression else 'classification',
            'target_mean': self.target_mean_ if self.regression else None,
            'target_scale': self.target_scale_ if self.regression else None,
            'readout': ['attention', 'mean_projected', 'final_projected',
                        'observable', 'direct_lexical_ngram_skip'],
            'n_train_rows': len(X), 'sequence_max': max(map(len, sequences))}
        print(f"  E47 Transformer done: state={self.state} "
              f"epochs={used}/{self.epochs} early_stop={self.fit_stats['early_stopped']}",
              flush=True)
        return self

    def predict_proba(self, X):
        if self.regression:
            raise TypeError('Regression Transformer uses predict(), not predict_proba()')
        import torch
        sequences = self._sequences(X)
        observable = self.observable_scaler_.transform(
            self._observable(X)).astype(np.float32)
        state = self._state_matrix(X)
        aux = self._aux_matrix(X)
        lexical = self._lexical_matrix(X)
        out = np.zeros((len(X), len(self.classes_)), dtype=np.float64)
        self.net_.eval()
        with torch.inference_mode():
            for batch in self._batches(sequences, np.arange(len(X))):
                xb, mb, ob, sb, ab, lb = self._pack(
                    sequences, batch, observable, state, aux, lexical)
                out[batch] = self.net_(xb, mb, ob, sb, ab, lb).softmax(1).cpu().numpy()
        lexical_out = np.zeros_like(out)
        lexical_q = self.lexical_model_.predict_proba(lexical)
        lexical_out[:, np.asarray(self.lexical_model_.classes_, dtype=int)] = lexical_q
        return .5 * out + .5 * lexical_out

    def predict(self, X):
        if not self.regression:
            raise TypeError('Classification Transformer uses predict_proba()')
        import torch
        sequences = self._sequences(X)
        observable = self.observable_scaler_.transform(
            self._observable(X)).astype(np.float32)
        state = self._state_matrix(X)
        aux = self._aux_matrix(X)
        lexical = self._lexical_matrix(X)
        out = np.zeros(len(X), dtype=np.float64)
        self.net_.eval()
        with torch.inference_mode():
            for batch in self._batches(sequences, np.arange(len(X))):
                xb, mb, ob, sb, ab, lb = self._pack(
                    sequences, batch, observable, state, aux, lexical)
                out[batch] = self.net_(xb, mb, ob, sb, ab, lb).squeeze(-1).cpu().numpy()
        lexical_out = np.asarray(
            self.lexical_model_.predict(lexical), dtype=np.float64)
        out = .5 * out + .5 * lexical_out
        return out * self.target_scale_ + self.target_mean_


def transformer_factory(policy, state, aux_features=False):
    cfg = policy.get('transformer_config', {})
    return lambda: TransformerReader(
        state=state, seed=policy.get('seed', 47),
        d_model=cfg.get('d_model', 128), layers=cfg.get('layers', 2),
        heads=cfg.get('heads', 4), epochs=cfg.get('epochs', 60),
        batch_size=cfg.get('batch_size', 8), patience=cfg.get('patience', 8),
        state_dim=cfg.get('state_dim', 32),
        lexical_features=cfg.get('lexical_features', 4096),
        attention_budget=cfg.get('attention_budget', 24_000_000),
        aux_features=aux_features)


def transformer_regression_factory(policy, state, aux_features=False):
    cfg = policy.get('transformer_config', {})
    return lambda: TransformerReader(
        state=state, seed=policy.get('seed', 47), regression=True,
        d_model=cfg.get('d_model', 128), layers=cfg.get('layers', 2),
        heads=cfg.get('heads', 4), epochs=cfg.get('epochs', 60),
        batch_size=cfg.get('batch_size', 8), patience=cfg.get('patience', 8),
        state_dim=cfg.get('state_dim', 32),
        lexical_features=cfg.get('lexical_features', 4096),
        attention_budget=cfg.get('attention_budget', 24_000_000),
        aux_features=aux_features)


def candidates(joint=False, ordered=False, view='text', quick=False,
               transformer=False, policy=None, transformer_only=False,
               reader_scope='all'):
    out = [('prior', PriorReader)]
    if view == 'state':
        return out + [(f'state{dim}_C{c}', lambda dim=dim, c=c: StateReader(c=c, state_dim=dim))
                      for dim in ((-1,) if quick else (-1, 8, 32))
                      for c in ((1.,) if quick else (0.1, 1.))]
    if transformer_only:
        if view != 'text' or policy is None:
            raise ValueError('Transformer-only evaluation requires text rows and policy')
        out.append(('transformer_state0', transformer_factory(policy, False)))
        if joint:
            out.append(('transformer_state', transformer_factory(policy, True)))
        return out
    if reader_scope == 'ordered_only':
        if view != 'text':
            raise ValueError('Ordered-only evaluation requires text rows')
        out.append(('ordered_state0', lambda: OrderedReader(False)))
        if joint:
            out.append(('ordered_state', lambda: OrderedReader(True)))
        return out
    if reader_scope not in ('all', 'cheap'):
        raise ValueError(f'Unknown reader scope: {reader_scope}')
    if view == 'text':
        out.append(('observable_outcome', OutcomeReader))
        out.extend((f'prompt_C{c}', lambda c=c: BlockReader(c=c, view='prompt'))
                   for c in ((1.,) if quick else (0.1, 1., 10.)))
    for kind in (('word',) if quick else ('word', 'char')):
        for c in ((1.,) if quick else (0.1, 1., 10.)):
            out.append((f'{kind}_C{c}_state0', lambda k=kind, c=c:
                        BlockReader(k, c, 0, view=view)))
    if reader_scope == 'all' and ordered and view == 'text':
        out.append(('ordered_state0', lambda: OrderedReader(False)))
    if reader_scope == 'all' and transformer and view == 'text':
        out.append(('transformer_state0', transformer_factory(policy, False)))
    if joint:
        for dim in ((-1,) if quick else (-1, 8, 32)):
            for scale in ((1.,) if quick else (0.1, 1.)):
                for c in ((1.,) if quick else (0.1, 1.)):
                    out.append((f'word_C{c}_state{dim}_scale{scale}', lambda dim=dim, scale=scale, c=c:
                                BlockReader('word', c, dim, scale)))
        if reader_scope == 'all' and ordered:
            out.append(('ordered_state', lambda: OrderedReader(True)))
        if reader_scope == 'all' and transformer:
            out.append(('transformer_state', transformer_factory(policy, True)))
    return out


def evaluate(rows, y, groups, policy, joint=False, view='text', quick=False,
             include_transformer=False, transformer_only=False,
             reader_scope='all', n_classes=None):
    y = np.asarray(y)
    K = int(np.max(y)) + 1 if n_classes is None else int(n_classes)
    fit = nested_oof(rows, y, np.asarray(groups), K,
                     candidates(joint, policy.get('ordered', False), view, quick,
                                include_transformer, policy, transformer_only,
                                reader_scope),
                     seed=policy['seed'], n_folds=policy.get('folds', 5))
    fit['loss'] = loss_rows(y, fit['P'])
    fit['base_loss'] = loss_rows(y, fit['base'])
    return fit


def regression_candidates(joint=False, view='text', quick=False,
                          include_transformer=False, policy=None):
    """Prespecified continuous-target ladder with text-family subsumption."""
    out = [('mean', MeanRegressor)]
    if view == 'state':
        return out + [
            (f'state{dim}_a{alpha}',
             lambda dim=dim, alpha=alpha: StateRegressor(
                 c=1. / alpha, state_dim=dim))
            for dim in ((-1,) if quick else (-1, 8, 32))
            for alpha in ((1.,) if quick else (0.1, 1., 10.))]
    if view == 'text':
        out.append(('observable', OutcomeRegressor))
        out.extend(
            (f'prompt_a{alpha}',
             lambda alpha=alpha: BlockRegressor(
                 c=1. / alpha, state_dim=0, view='prompt'))
            for alpha in ((1.,) if quick else (0.1, 1., 10.)))
    for kind in (('word',) if quick else ('word', 'char')):
        for alpha in ((1.,) if quick else (0.1, 1., 10.)):
            out.append((
                f'{kind}_a{alpha}_state0',
                lambda kind=kind, alpha=alpha: BlockRegressor(
                    kind, 1. / alpha, 0, view=view)))
    if include_transformer and view == 'text':
        out.append(('transformer_state0',
                    transformer_regression_factory(policy, False)))
    if joint:
        for dim in ((-1,) if quick else (-1, 8, 32)):
            for scale in ((1.,) if quick else (0.1, 1.)):
                for alpha in ((1.,) if quick else (0.1, 1., 10.)):
                    out.append((
                        f'word_a{alpha}_state{dim}_scale{scale}',
                        lambda dim=dim, scale=scale, alpha=alpha:
                        BlockRegressor('word', 1. / alpha, dim, scale)))
        if include_transformer:
            out.append(('transformer_state',
                        transformer_regression_factory(policy, True)))
    return out


def _group_splits(groups, n_folds, seed):
    from e46_readers import splits
    return list(splits(np.asarray(groups), n_folds, seed))


def _stratum_means(y, strata, indices):
    """Training-only means with a global fallback for unseen strata."""
    y, strata, indices = np.asarray(y, float), np.asarray(strata), np.asarray(indices)
    fallback = float(np.mean(y[indices]))
    means = {
        stratum: float(np.mean(y[indices][strata[indices] == stratum]))
        for stratum in np.unique(strata[indices])}
    return means, fallback


def _stratum_baseline(strata, indices, means, fallback):
    strata = np.asarray(strata)
    return np.asarray([means.get(strata[i], fallback) for i in indices], float)


def nested_oof_regression(rows, y, groups, factories, seed=47, n_folds=5,
                          strata=None):
    """Grouped nested selection for continuous targets; no test-fold tuning.

    When ``strata`` is supplied, every fit predicts variation remaining after
    training-fold stratum means.  Test targets and baselines use only means
    estimated from the corresponding outer-training fold; inner candidate
    selection recomputes the means from each inner-training fold.  This makes
    the reported gain conditional on a known experimental manipulation without
    leaking held-out outcomes or privileging a particular reader family.
    """
    y, groups = np.asarray(y, float), np.asarray(groups)
    strata = None if strata is None else np.asarray(strata)
    if strata is not None and len(strata) != len(y):
        raise ValueError('Regression strata must align one-for-one with rows')
    prediction = np.zeros(len(y), dtype=float)
    baseline = np.zeros(len(y), dtype=float)
    fold_ids = np.full(len(y), -1, dtype=int)
    details = []
    outer = _group_splits(groups, n_folds, seed)
    for fold, (train, test) in enumerate(outer):
        candidates_scored = []
        inner_groups = groups[train]
        inner_folds = min(n_folds - 1, len(np.unique(inner_groups)))
        if inner_folds < 2:
            raise ValueError('Continuous target needs at least two inner group folds')
        for name, factory in factories:
            losses = []
            for inner_train, inner_test in _group_splits(
                    inner_groups, inner_folds, seed + 1009 + fold):
                fit_idx, val_idx = train[inner_train], train[inner_test]
                if strata is None:
                    fit_y, val_y = y[fit_idx], y[val_idx]
                    offset = np.zeros(len(val_idx), float)
                else:
                    means, fallback = _stratum_means(y, strata, fit_idx)
                    fit_offset = _stratum_baseline(
                        strata, fit_idx, means, fallback)
                    offset = _stratum_baseline(
                        strata, val_idx, means, fallback)
                    fit_y, val_y = y[fit_idx] - fit_offset, y[val_idx] - offset
                model = factory().fit([rows[i] for i in fit_idx], fit_y)
                pred = model.predict([rows[i] for i in val_idx])
                squared = np.square(val_y - pred)
                losses.extend(
                    float(squared[groups[val_idx] == group].mean())
                    for group in np.unique(groups[val_idx]))
            candidates_scored.append((float(np.mean(losses)), name, factory))
        score, name, factory = min(candidates_scored, key=lambda x: (x[0], x[1]))
        if strata is None:
            train_y = y[train]
            baseline[test] = float(np.mean(y[train]))
        else:
            means, fallback = _stratum_means(y, strata, train)
            train_offset = _stratum_baseline(strata, train, means, fallback)
            train_y = y[train] - train_offset
            baseline[test] = _stratum_baseline(strata, test, means, fallback)
        fitted = factory().fit([rows[i] for i in train], train_y)
        prediction[test] = (baseline[test]
                            + fitted.predict([rows[i] for i in test]))
        fold_ids[test] = fold
        detail = {
            'fold': fold, 'candidate': name, 'inner_mse': score,
            'baseline': ('training_fold_stratum_mean' if strata is not None
                         else 'training_fold_global_mean'),
            'train_groups': np.unique(groups[train]).tolist(),
            'test_groups': np.unique(groups[test]).tolist()}
        if isinstance(getattr(fitted, 'fit_stats', None), dict):
            detail['fit_stats'] = fitted.fit_stats
        details.append(detail)
    return {'prediction': prediction, 'base': baseline, 'fold': fold_ids,
            'details': details, 'squared_error': np.square(y - prediction),
            'base_squared_error': np.square(y - baseline)}


def evaluate_regression(rows, y, groups, policy, *, joint=False,
                        view='text', quick=False, include_transformer=False,
                        strata=None):
    return nested_oof_regression(
        rows, y, groups,
        regression_candidates(joint, view, quick, include_transformer, policy),
        seed=policy.get('seed', 47), n_folds=policy.get('folds', 5),
        strata=strata)


def fixed_evaluate(rows, y, groups, policy, *, family, joint=False):
    """Grouped OOF score for one prespecified expensive reader.

    Capacity gates do not select among model families, so an inner CV loop
    would only refit the same neural reader three extra times per outer fold.
    Each outer-training fold is instead split once, by group and without using
    labels, into fit and calibration subsets.  Temperature and prior mixture
    are selected on the held-out calibration groups, then applied to untouched
    outer-test groups. Natural-signal analysis continues to use ``evaluate``
    and nested model selection.
    """
    from e46_readers import (calibrate, choose_calibration, fit_predict, prior,
                             splits, take)
    y, groups = np.asarray(y, dtype=int), np.asarray(groups)
    if family == 'ordered':
        factory = lambda: OrderedReader(joint)
        name = 'ordered_state' if joint else 'ordered_state0'
    elif family == 'transformer':
        factory = transformer_factory(policy, joint)
        name = 'transformer_state' if joint else 'transformer_state0'
    else:
        raise ValueError(f'Unknown fixed reader family: {family}')
    P = np.zeros((len(y), 2), dtype=float)
    bases = np.zeros_like(P)
    fold_ids = np.full(len(y), -1, dtype=int)
    details = []

    def predict_fitted(fitted, indices):
        out = np.zeros((len(indices), 2), dtype=float)
        q = fitted.predict_proba(take(rows, indices))
        out[:, np.asarray(fitted.classes_, dtype=int)] = q
        return out

    for fold, (train, test) in enumerate(
            splits(groups, policy.get('folds', 5), policy.get('seed', 47))):
        bases[test] = prior(y[train], 2)
        train_groups = np.unique(groups[train])
        shuffled = np.random.default_rng(
            policy.get('seed', 47) + 1009 + fold).permutation(train_groups)
        n_cal = (max(1, int(round(0.15 * len(shuffled))))
                 if len(shuffled) > 3 else 0)
        cal_group_set = set(shuffled[:n_cal])
        is_cal = (np.isin(groups[train], list(cal_group_set))
                  if n_cal else np.zeros(len(train), bool))
        fit_indices, calibration = train[~is_cal], train[is_cal]
        can_calibrate = (
            len(calibration) > 0
            and len(np.unique(y[fit_indices])) == 2
            and len(np.unique(y[calibration])) == 2)
        if can_calibrate:
            calibration_raw, fitted = fit_predict(
                factory, rows, y, fit_indices, calibration, 2,
                return_estimator=True)
            fit_prior = prior(y[fit_indices], 2)
            calibration_bases = np.tile(fit_prior, (len(calibration), 1))
            calibration_ce, temperature, mixture = choose_calibration(
                calibration_raw, y[calibration], calibration_bases)
            raw = predict_fitted(fitted, test)
            P[test] = calibrate(
                raw, temperature, mixture,
                np.tile(fit_prior, (len(test), 1)))
        else:
            # A label-independent split can occasionally lose a class.  Do
            # not search for a favourable split; fit all outer-training rows
            # and use the prespecified identity calibration.
            raw, fitted = fit_predict(
                factory, rows, y, train, test, 2, return_estimator=True)
            P[test] = raw
            calibration_ce, temperature, mixture = None, 1.0, 0.0
            fit_indices, calibration = train, np.array([], dtype=int)
        fold_ids[test] = fold
        detail = {'fold': fold, 'candidate': name,
                  'temperature': temperature, 'prior_mixture': mixture,
                  'calibration_ce_bits': calibration_ce,
                  'train_groups': train_groups.tolist(),
                  'fit_groups': np.unique(groups[fit_indices]).tolist(),
                  'calibration_groups': np.unique(groups[calibration]).tolist(),
                  'test_groups': np.unique(groups[test]).tolist()}
        if fitted is not None and isinstance(
                getattr(fitted, 'fit_stats', None), dict):
            detail['fit_stats'] = fitted.fit_stats
        details.append(detail)
    return {'P': P, 'base': bases, 'fold': fold_ids, 'details': details,
            'loss': loss_rows(y, P), 'base_loss': loss_rows(y, bases),
            'evaluation_version': 'e47_fixed_grouped_oof_v2_calibrated'}


def interval(values, groups, seed=47, n_boot=2000):
    values, groups = np.asarray(values), np.asarray(groups)
    means = np.array([values[groups == g].mean() for g in np.unique(groups)])
    rng = np.random.default_rng(seed)
    draws = rng.choice(means, (n_boot, len(means)), replace=True).mean(1)
    return {'mean': float(means.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist(),
            'n_items': len(means), 'scope': 'item bootstrap of fixed OOF losses; not refit uncertainty'}
