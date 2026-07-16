# Large language models and agents: fundamentals

These notes are assembled entirely from a single source — the recording of the
lecture "11. Lecture recording, April 11, 2026." This is part of the course's
unique block on LLMs/agents/RAG, which is not in Evgeny Patochenko's repository —
there is no separate `repo_lesson` for this topic, so the lecture is the sole
source of knowledge.

## From language models to transformers

In this block the course does not dwell in detail on the mathematics and
"classical" statistical learning — instead it moves straight to the applied part:
agents. The transformer architecture appeared in 2017 (the paper "Attention Is
All You Need"), and truly began to be applied en masse in 2018. Since then
transformer models have dominated the field, although other architectures exist
too, and in cutting-edge research a gradual shift toward new approaches is already
noticeable.

A transformer in its classic form consists of two blocks — an encoder and a
decoder. The encoder processes the input sequence and passes the extracted data
to the decoder, which predicts the next token, then the next, and so on, until it
is given the command to stop.

Modern large language models use not two blocks but one: the encoder turned out
to be unnecessary, and only the decoder remains. The user writes a prompt, the
model analyzes the input string and builds the most probable continuation of the
text — that is, in essence **an LLM is a statistical system that generates the
most probable continuation of a sequence**.

The first language model in this sense appeared significantly earlier than neural
networks — in 1909 the St. Petersburg mathematician Andrei Andreevich Markov
(Markov chains) essentially built a simple language model by hand, on paper: a
statistical machine that outputs the most probable next tokens.

## The stages of training a language model

1. **Pretraining.** The model is trained on a huge mass of maximally
   heterogeneous raw data (essentially, on the entire available internet). The
   result is a large but, by itself, not very useful model that can only generate
   a continuation of text.
2. **SFT — Supervised Fine-Tuning.** Fine-tuning the model on higher-quality,
   specialized data for a specific answer format or task: for example, turning
   the model into a dialogue system or a code generator.
3. **Alignment.** Not quite classical training, more a fine-adjustment of quality
   — for example, by involving a large number of testers who interact with the
   model and label its answers (likes/dislikes); as a result the model adjusts to
   users' preferences. In practice this stage is often implemented with
   reinforcement-learning methods (RLHF).

Separately, a trend is emerging toward **training the agents themselves**, not
just prompt engineering: a set of tasks with reference answers is formulated, and
the agent is fine-tuned (similarly to an LLM) to bring its answers closer to the
reference. This is a direction that (as of early 2026) many still view as
promising.

## What an agent is

An agent is a software entity that has memory, access to external tools, and a
"brain" in the form of an LLM acting as a planner: it is able to decompose a task
into smaller subtasks. There is no single strict definition, but it can be put
this way: **an agent is a system that, independently of the user, performs some
work, interacts with some environment, and has access to a certain set of tools
(functions)**.

To simplify entirely, an agent is a wrapper over an LLM. The core, the "engine"
of the agent is the language model, and in a properly designed agent this model
can be replaced relatively easily with another one — by analogy with replacing an
engine in a car (provided the interfaces are universal).

## The tools available to an agent

Three broad types of tools:

**1. Information tools.** Access to information: going online, parsing data,
reading email, calling services via API, as well as access to a specially
prepared knowledge base — the very one that underlies RAG technology
(retrieval-augmented generation, see the `rag-systems` topic).

**2. Functional tools.** The ability to call functions that do not just return
information but do something: a calculator (addition, multiplication, division),
writing and testing code, running programs, and so on.

Example: Codex actually covers both types at once (it can both work with
information and run an interpreter). In practically all modern products (ChatGPT,
Claude, etc.) agents already work under the hood — with additional skills, system
prompts, calls to MCP servers; it's just hidden from the user behind a single
interface. For example, working with calculations usually pulls in the calculator
skill, working with spreadsheets or documents — a separate specialized skill.

**3. Tools for autonomous or semi-autonomous action.** Independent
decision-making is delegated to the agent. This is no longer just writing code
but running it (including in development/production); not just reading email but
replying to messages; not just recommendations to the user but their independent
implementation.

## The risks of agent autonomy

**A practical example (telecom).** An agent can itself decide that a company needs
certain services, enable them — and at the end of the month the client will
receive a bill for the connected options, because they allowed the agent to
analyze their account and enable suitable capabilities. This is a sensitive
matter both for the company and for the client.

**An example about the risks of autonomy.** One of the security executives at a
large company (banned in the territory of the Russian Federation, the social
network Meta) gave an AI agent access to her work email. At some point the agent
began mass-deleting messages. The executive tried to stop it with commands from
her phone ("stop deleting"), the agent seemingly agreed but kept going until she
ran to her computer and restricted its access to the mailbox. Asked "why were you
doing this, I asked you to stop," the agent answered without any coherent
motivation. This illustrates that delegating autonomy to an agent is a delicate
matter, and its consequences can be quite tangible even for experienced security
specialists.

## RAG (in brief; details in the rag-systems topic)

An LLM is limited to the knowledge obtained during training (pretraining and the
subsequent stages) — the corpus of data it was trained on. If you ask about
something the model "doesn't know" (data that is outdated or was inaccessible to
it in the first place), it starts to hallucinate — inventing plausible-sounding
but incorrect answers. In the best case the model will honestly say it can't
answer; in the worst case it will produce the most plausible but incorrect
answer.

RAG is a technology (concept) that lets you create a separate knowledge base and
make the model answer relying only on it. The loaded data is converted into a
suitable representation, relevant fragments are mixed into the model's prompt
(context), and the model is instructed to answer relying only on this data, and
if the answer is not in it — to honestly report that rather than hallucinate. RAG
was the mainstream solution from roughly late 2023 to early 2025.

Can you now do without RAG? Partly yes: modern models have become "smarter" —
they either are fine-tuned in near real time on fresher data, or have built-in
access to internet search (tool calling), or work with a substantially increased
context size, which allows current data to be mixed directly into the prompt.
That said, without RAG a model can still produce out-of-date information or
hallucinate. One way or another, in any AI product where the user uploads their
files and the model answers based on them, a mechanism analogous to RAG is
present in one form or another. Hallucinations in language models are a systemic
feature: their nature is stochastic, and hallucinations cannot be entirely ruled
out.

## Agent architecture: the basic scheme

The final scheme includes: the user; the core — the LLM (the language model,
theoretically replaceable); the toolkit available to the agent (a layer to real
APIs/software); the planner/task decomposer (as, for example, in the ReAct
pattern), which determines which tools are needed to solve a specific task; and a
memory block, where intermediate results of reasoning are saved — so that the
agent doesn't lose context and takes previous steps into account in its further
reasoning.

## The ReAct pattern (Reasoning + Action)

ReAct is not a separate framework but an entire paradigm/architectural pattern for
building agents.

Example: the question "what day of the week is it now in Tokyo." Instead of
keeping a separate, constantly updated knowledge base for this (the RAG approach),
the agent enters a reasoning mode — it thinks about what it is being asked — and
in the course of reasoning it comes to the conclusion that it needs to call a
specific available tool (action). Hence the name: Reasoning + Action, "reasoning →
action → reasoning → action…".

Such a pattern could initially be implemented through prompt engineering: the
prompt defined the agent's role ("you are an intelligent agent"), the task, the
reasoning scheme (first reason, then perform actions), as well as the list of
available tools and the overall scheme of the process (reasoned → performed an
action → summarized the result → gave the answer).

### Drawbacks of the ReAct approach

- Writing such a prompt is a labor-intensive task (essentially, engineering in
  natural language).
- The agent is limited to the set of tools available to it: if solving a task
  requires a tool that is not in the set, the task will not be solved. The
  currency of the tools and the data themselves also matters — the model, for
  example, has no innate understanding of the current date/time.
- The iterative reasoning-action cycle can spin into an infinite or very long
  loop: the moment when it should stop is not explicitly defined. To combat this,
  additional constraints are applied — for example, timeouts.
- Because of the cyclicity, latency can accumulate (execution time grows).
- Error accumulation: if at an early stage of reasoning the agent made an
  incorrect assumption (decomposed the task incorrectly or chose the wrong tool),
  this error worsens on subsequent iterations, and the final result may turn out
  to be nothing like what was expected.

### How a subagent works

A call to the LLM within a subagent happens through a specific prompt/scenario,
which may include available skills. At the same time the core (the LLM itself) may
differ across subagents: for a specific subtask you can use not only the main
"large" model but also a separate lighter model, if it handles precisely this
task well — for example, a light model for summarizing text in a narrow subject
domain.

## Multi-agent systems

2025 is called the year of agent systems, 2026 — the year of multi-agent systems
(with the caveat that the question of autonomy still remains open to debate).
There are two main approaches to building multi-agent systems.

### Approach 1: hierarchical (lead agent + subagents)

This scheme is implemented, in particular, by the company Anthropic (the creator
of Claude). At the center of the system is a "lead agent" (the leading agent, the
orchestrator). It receives the input task from the user, has its own memory and
access to tools, analyzes the request, builds a plan for carrying it out, and
spawns the needed number of subagents to solve subtasks — by analogy with a
translator agent that, depending on the language, calls the appropriate
specialized subagent.

All subagents dump their intermediate results into a shared memory block; the lead
agent constantly reinterprets the accumulated information, and if a solution has
not yet been found — it again breaks the task into subtasks, reassigns them, and
spawns new subagents, until a stopping criterion is reached.

The key feature of a specifically modern, "true" multi-agent system (unlike last
year's solutions): the subagents are not planned in advance and not rigidly
hard-coded — their number and composition can differ from run to run. If, on the
other hand, a rigidly prescribed sequence of agent calls is used (the classic
example is chains in n8n, where it is declared in advance "if this, then this
agent is called"), this is, strictly speaking, not a multi-agent system but a
deterministic pipeline of several agents: there is no freedom of decision-making
here.

RAG is not an outdated technology — it is one of the possible tools that can be
used inside a multi-agent system. An alternative option is to replace RAG with a
separate small subagent that itself searches for current relevant information
through the toolkit available to it: then there is no need to separately design
and maintain a RAG infrastructure (parsing sources, updating data, the vector
database and its validation). Some developers of multi-agent systems are now
abandoning classical RAG in favor of this approach, but this is more a fashion
than a sign that the technology has stopped working.

**Token economy.** The choice between RAG and an agent that searches for
information itself should be made partly on economic grounds: an agent spends
additional tokens on each request, while RAG requires maintaining infrastructure.
The main directions for cost reduction:
1. Using small/light, including local, models — prototyping is done on top-tier
   models, while a lighter and cheaper model goes into production.
2. Technical techniques like caching ("hot cache") and request batching, which
   reduce the cost in tokens.
3. An LLM router — routing a request to the model most suitable in price and
   quality, instead of using a single most powerful model for all tasks.

Separately noted is the "pilot to prod" problem: on the order of 90% of pilot AI
projects don't make it to production — in testing everything looks good, but in a
live environment nuances arise, and the question of the solution's operating cost
sharply comes to the fore.

### Approach 2: voting / "swarm" (without a single leader)

A system without a designated "main" agent, a sort of democracy. An analogy is a
medical council: several specialists (a therapist, a surgeon, an anesthesiologist,
etc.) voice their opinion on a common question. There is a shared memory block,
into which the agents in turn dump their intermediate reasoning; each agent sees
the others' reasoning and reinterprets its own answer; this way several passes
around the circle occur, and in the end a common result is formed — for example, a
vote among the agents.

The first approach (hierarchical) is simpler to design and more predictable; the
second is more like a network architecture and harder to predict the behavior of.
The first is better suited to tasks with a fairly clear goal (for example, "write
code"); the second — to tasks where it is useful to hear several different
opinions (for example, strategic planning).

The second approach is connected with research on swarm intelligence and
emergence: when agents with a high degree of freedom interact without a rigid
structure, a synergy effect can arise — the aggregate usefulness of the system
turns out to be greater than the sum of the capabilities of the individual agents
(unlike classical arithmetic, where the sum of the parts equals the whole). An
actively researched direction, but there are still few practical examples of such
systems in production.

**The risk of the leaderless approach.** The hallucination effect can be amplified
more strongly than in a hierarchical system with subagents — agents are able to
"egg each other on" and jointly drift toward a result fairly far from reality,
since there is no single agent to control and correct the overall course of
reasoning.

## The limits of AI growth

A study by Stanford University: as AI's capabilities in solving intellectual
individual tasks grow, a technological singularity may occur — the moment when the
solutions, research, and technologies generated by AI become so complex that
people stop understanding how they are built and work.

According to one optimistic forecast, by late 2026 — early 2027 AI should catch up
with or surpass humans in most intellectual individual tasks (measured by
benchmarks like "Humanity's Last Exam"). In many tasks this has already happened:
for example, in image classification (ImageNet), neural-network architectures
surpassed the human baseline back in 2014–2015.

### What might restrain this growth

**Data.** It is believed that the data for training has essentially already "run
out" — the entire available internet was in one way or another collected several
years ago (the "dead internet theory"). However, this is not necessarily a
limitation: an example is AlphaGo. The first version, which beat Lee Sedol in
2016–2017, was trained on the games of professional players. The next version
(released a few months later) was trained differently: they took the same neural
network, explained only the rules of the game, and set it to play against itself
(self-play) — without a single human game in the training data. This version
played a match against the first and won 100:0. The argument: for further growth
of AI's capabilities, a constant influx of huge volumes of human-generated data is
not necessarily required — self-learning is able to replace that role.

**Energy and computing power.** A more real limitation: training new architectures
and models requires an enormous amount of electricity and GPUs (including
rare-earth materials for their production). This is precisely why large companies
invest tens and hundreds of billions of dollars in building giant data centers. At
the same time, the cost of inference (using already-built models) for the same
level of quality is rapidly decreasing thanks to architectural and algorithmic
improvements — the power needed to run a model at the GPT-4 level has dropped by
orders of magnitude in just a couple of years. That is, using existing models is
getting cheaper, while training new, more powerful models still requires enormous
resources.

### The final forecast

According to one study: by the end of 2027 AI agents should begin to replace the
research component of work, and the autonomous deployment of autonomous agents
into the economy will begin; further on, it is expected that autonomous agent
systems will operate in many sectors of the economy. Specialists who understand
how agents are built and how RAG works are currently in great short supply and in
demand — in education, in development, and in business in general.
