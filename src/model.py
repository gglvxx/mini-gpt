"""Arhitectura Mini-GPT: un Transformer decoder-only, la scară mică."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config


class CausalSelfAttention(nn.Module):
    """Fiecare caracter "se uită" la caracterele ANTERIOARE și decide care contează."""

    def __init__(self, config: Config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Un singur strat calculează Query, Key, Value pentru toate capetele deodată:
        #   Query = "ce caut?"   Key = "ce conțin?"   Value = "ce informație ofer?"
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)  # recombină rezultatele capetelor
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Masca triunghiulară: poziția i vede doar pozițiile 0..i, NU viitorul.
        # Fără ea, modelul ar "trișa" uitându-se la răspuns.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        # register_buffer = tensor care se mută automat pe GPU odată cu modelul,
        # dar nu e parametru antrenabil. persistent=False -> nu e salvat în checkpoint.
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape  # Batch, Time (lungimea secvenței), Channels (n_embd)

        q, k, v = self.qkv(x).split(C, dim=2)
        # [B, T, C] -> [B, n_head, T, head_dim]: fiecare cap lucrează independent
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scor de atenție: cât de relevant e fiecare caracter pentru fiecare altul.
        # Împărțirea la sqrt(head_dim) ține valorile stabile numeric.
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # ascunde viitorul
        att = F.softmax(att, dim=-1)  # scoruri -> probabilități (suma pe rând = 1)
        att = self.attn_dropout(att)

        out = att @ v  # medie ponderată a informațiilor (Value)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # lipește capetele la loc
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    """După ce caracterele "au comunicat" prin atenție, fiecare "gândește" individual."""

    def __init__(self, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # extinde de 4x (standard GPT)
            nn.GELU(),                                     # non-liniaritate
            nn.Linear(4 * config.n_embd, config.n_embd),  # comprimă la loc
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Un bloc Transformer = Attention + FeedForward, cu normalizare și conexiuni reziduale."""

    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        # "x + ..." = conexiune reziduală: informația originală trece mai departe nealterată,
        # iar stratul adaugă doar o corecție. Esențial pentru a antrena rețele adânci.
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.block_size = config.block_size

        # Embedding = tabel care transformă fiecare ID într-un vector antrenabil
        self.token_emb = nn.Embedding(vocab_size, config.n_embd)        # CE caracter e
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)   # UNDE se află
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        # Stratul final: vector -> un scor pentru fiecare caracter posibil din vocabular
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Greutăți inițiale mici și aleatorii (std=0.02, ca în GPT-2) -> antrenare stabilă
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """idx: [B, T] ID-uri. Returnează logits [B, T, vocab_size] și loss (dacă avem targets)."""
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Secvența are {T} tokeni, maximul este {self.block_size}")

        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)  # sensul caracterului + poziția lui
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # scoruri brute (nenormalizate) pentru următorul caracter

        loss = None
        if targets is not None:
            # cross_entropy = cât de "surprins" e modelul de caracterul corect. Mai mic = mai bine.
            # .view(-1, ...) aplatizează batch-ul: tratăm toate pozițiile ca un singur șir lung
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()  # la generare nu antrenăm -> fără calcul de gradienți (mai rapid)
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        """Generează text caracter cu caracter. Apelează model.eval() înainte."""
        for _ in range(max_new_tokens):
            # Păstrăm doar ultimele block_size caractere (limita de context a modelului)
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)

            # Ne interesează doar predicția pentru ULTIMA poziție.
            # temperature < 1 -> text mai sigur/repetitiv; > 1 -> mai creativ/haotic
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                # Păstrează doar cele mai probabile top_k caractere, restul devin imposibile
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # extrage aleator după probabilități
            idx = torch.cat((idx, next_id), dim=1)              # adaugă la secvență și repetă
        return idx