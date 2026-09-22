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
| 2 | [The cook learns](#day-2--the-cook-learns) | bigram model, training, autoregression |
| 3 | [The dining room opens](#day-3--the-dining-room-opens) | guest, one-hot, embedding, RMSNorm, Linear, RoPE |

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


## Day 2 — the cook learns

On Day 1 the kitchen followed a made-up rulebook. On Day 2 the cook
**learns from old order tickets** and then **serves course after course**,
each one chosen from the one before — your first complete language model.

<div class="concept" id="bigram">

**Bigram model — the tally sheet** · [Day 2](day-02-toy-language-model.md)

*What it is:* a model that predicts the next token from only the one token before it, using counts of which token followed which in real text.

*In the story:* the cook keeps a tally sheet with one row per jar just used and one column per jar that came next, and adds a mark for every pair seen in old order tickets.

</div>

<div class="concept" id="training">

**Training — learning from old tickets** · [Day 2](day-02-toy-language-model.md)

*What it is:* adjusting a model's numbers based on example text. Real training uses gradient descent, a smarter version of the same idea.

*In the story:* the cook reading old order tickets and updating the tally sheet: “+1 per sighting” is training in miniature.

</div>

<div class="concept" id="autoregression">

**Autoregression — course after course** · [Day 2](day-02-toy-language-model.md)

*What it is:* generating text one token at a time, feeding each chosen token back in as input for the next step.

*In the story:* the cook serves a course, looks at what was just served, and uses it to choose the next one — again and again.

</div>


## Day 3 — the dining room opens

The kitchen alone can only look one token back. So the **dining room**
opens: your prompt becomes tonight's table of **guests**, and a host gets
every guest ready before anyone talks — swapping number badges for
profile cards, leveling voices, rewriting cards, and stamping seat
numbers.

<div class="concept" id="guest">

**Guest — one token at the table** · [Day 3](day-03-layers-before-attention.md)

*What it is:* one token of your prompt, seen from inside the model. A prompt of five tokens seats five guests. With the byte tokenizer a guest is one byte — one letter of plain English; from Day 7 it is a word piece.

*In the story:* each guest carries a card of numbers describing them, and from Day 4 the guests talk to each other.

</div>

<div class="concept" id="one-hot">

**One-hot — the numbered badge** · [Day 3](day-03-layers-before-attention.md)

*What it is:* a row of zeros with a single 1 at the token's id. It says which token this is and nothing else.

*In the story:* a guest walks in wearing only a numbered badge (`a` is badge 97). Two guests with the same badge are indistinguishable.

</div>

<div class="concept" id="embedding">

**Embedding — the profile card** · [Day 3, section A](day-03-layers-before-attention.md)

*What it is:* a lookup table that turns a token id into a short list of numbers describing that token. In a trained model, tokens used in similar ways carry similar numbers.

*In the story:* at the door, the host swaps badge 97 for card 97 from a filing cabinet. The same badge always gets the same card.

</div>

<div class="concept" id="rmsnorm">

**RMSNorm — the volume knob** · [Day 3, section B](day-03-layers-before-attention.md)

*What it is:* rescales each token's numbers so their overall size is 1.0, keeping their proportions. It runs before every major step so numbers never grow or shrink out of control.

*In the story:* every guest is turned to the same speaking volume before each activity. Loud cards get quieter, faint cards get louder, and nobody's message changes.

</div>

<div class="concept" id="linear">

**Linear — the translator** · [Day 3, section B](day-03-layers-before-attention.md)

*What it is:* a matmul by a weight matrix: each output number is a weighted blend of all the input numbers. The weights are what training learns.

*In the story:* before each activity, a translator rewrites every card into the form that activity needs, following a recipe of how much of each old number goes into each new one.

</div>

<div class="concept" id="rope">

**RoPE — the seat stamp** · [Day 3, section C](day-03-layers-before-attention.md)

*What it is:* rotary position embedding: encodes each token's position by rotating pairs of its numbers by an angle that grows with the position.

*In the story:* each pair of numbers on a card is a little dial, and sitting in seat N turns every dial N clicks. The first dial moves like a second hand, later ones like minute and hour hands, so comparing two guests' dials tells how many seats apart they sit.

</div>

