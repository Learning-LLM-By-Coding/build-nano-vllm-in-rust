# The restaurant: an analogy to help you build the right mental model

Every day of this book adds one scene to a single running story: **a
little restaurant, built one scene at a time.** Each piece of the engine
you build is one thing in that restaurant — a jar in the pantry, a guest at
a table, a chef with a recipe book. Keep the restaurant in your head and
you can always picture the engine, even on the days the code gets dense.

This page keeps the whole story in one place:

- **Before you start**, read it through once. A few minutes here give you
  a map of where the course is going, and every later chapter will feel
  like a place you have already visited.

- **While you work**, come back whenever a name from an earlier day stops
  making sense. You rarely need to leave the chapter to do that: a concept
  name written like [this](#token) is a link to this page, and clicking it
  opens that concept's reminder right where you are reading. (Press Esc or
  click anywhere else to close it.)

- **The page grows with the book.** Every new day adds its scene here, so
  it always covers exactly the days you can read.

Each reminder has the same three parts: *what it is* in plain words, *in
the story* — the thing it is in the restaurant — and the day that
introduced it, if you want the full explanation again.

| Day | Scene | Concepts it adds |
|-----|-------|------------------|
| 1 | [The kitchen](#day-1--the-kitchen) | token, tensor, matmul, logits, greedy pick |

## Day 1 — the kitchen

The restaurant starts with its kitchen. Your text arrives as an **order
slip**. The kitchen has a pantry of numbered jars, a recipe board for
cooking, and a judge who picks the winning dish. Together they do the one
thing every language model does: read some text, score every possible
next piece, and choose one.

<div class="concept" id="token">

**Token — a pantry jar** · [Day 1](day-01-next-token.md)

*What it is:* the unit of text a model reads and writes. A model never sees letters, only token ids — whole numbers. The full list of tokens it knows is its *vocabulary*.

*In the story:* the pantry has one numbered jar for every token. Our first pantry has 256 jars, one for each possible byte, so any order slip in any language becomes a list of jar numbers: the text `hi` is jars 104 and 105.

</div>

<div class="concept" id="tensor">

**Tensor — numbers with a shape** · [Day 1](day-01-next-token.md)

*What it is:* a grid of numbers plus its *shape*: `[1, 256]` means 1 row of 256 numbers. Every value inside the engine is a tensor.

*In the story:* the numbers written on the recipe board and on every dish score.

</div>

<div class="concept" id="matmul">

**Matrix multiplication — the recipe board** · [Day 1](day-01-next-token.md)

*What it is:* multiply-and-sum between two tensors; the shapes must line up (`[1, 3] × [3, 2] → [1, 2]`). It is the operation a model spends nearly all its time on.

*In the story:* a recipe is a column of amounts: this much from jar A, this much from jar B. One matmul cooks every recipe on the board at once.

</div>

<div class="concept" id="logits">

**Logits — a score for every dish** · [Day 1](day-01-next-token.md)

*What it is:* the model's output: one score for every token in the vocabulary. A higher score means the model thinks that token is a likelier next token.

*In the story:* every dish the kitchen could serve next gets a score.

</div>

<div class="concept" id="greedy">

**Greedy pick — the judge** · [Day 1](day-01-next-token.md)

*What it is:* choose the token with the highest logit. The same input always gives the same choice, which makes the engine easy to test.

*In the story:* the judge always picks the highest-scoring dish. No dice involved.

</div>

